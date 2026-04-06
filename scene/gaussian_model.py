import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import open3d as o3d
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from random import randint
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.deformation import deform_network
from scene.regulation import compute_plane_smoothness
import random


class MotionStatePredictor(nn.Module):

    def __init__(self, hidden_dim=512, num_layers=5, time_dim=128, fourier_dim=30, dropout=0.1):

        super().__init__()

        self.time_dim = time_dim
        self.fourier_dim = fourier_dim

        self.time_proj = nn.Linear(fourier_dim, time_dim)
        self.time_encoder = nn.Linear(time_dim, hidden_dim)

        layers = [nn.ReLU(), nn.Dropout(dropout)]

        for i in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))  # 每个隐藏层后都加 dropout

        self.feature_layers = nn.Sequential(*layers)
        self.output_layer = None

    def fourier_encode_time(self, t, max_freq=15):
        freqs = torch.arange(0, max_freq, device=t.device).float()
        freqs = 2.0 ** freqs * np.pi

        t_expanded = t.unsqueeze(-1)  # [batch_size, 1, 1]
        freqs_expanded = freqs.unsqueeze(0).unsqueeze(0)  # [1, 1, max_freq]

        encoded = torch.cat([
            torch.sin(t_expanded * freqs_expanded),
            torch.cos(t_expanded * freqs_expanded)
        ], dim=-1).squeeze(1)  # [batch_size, 2*max_freq=30]

        return encoded

    def update_output_dim(self, num_gaussians, copy_weights=None, parent_indices=None):

        hidden_dim = None
        for layer in reversed(self.feature_layers):
            if isinstance(layer, nn.Linear):
                hidden_dim = layer.out_features
                break

        if hidden_dim is None:
            hidden_dim = 512

        old_layer = self.output_layer
        self.output_layer = nn.Linear(hidden_dim, num_gaussians).cuda()

        nn.init.xavier_uniform_(self.output_layer.weight)

        if old_layer is not None and copy_weights is not None:
            with torch.no_grad():
                if copy_weights == "expand":
                    old_num = old_layer.out_features

                    self.output_layer.weight.data[:old_num].copy_(old_layer.weight.data)
                    self.output_layer.bias.data[:old_num].copy_(old_layer.bias.data)

                    if parent_indices is not None and old_num < num_gaussians:
                        new_point_indices = torch.arange(old_num, num_gaussians, device='cuda')
                        for new_idx in new_point_indices:
                            parent_idx = parent_indices[new_idx - old_num]
                            if parent_idx < old_num:
                                self.output_layer.weight.data[new_idx] = old_layer.weight.data[parent_idx]
                                self.output_layer.bias.data[new_idx] = old_layer.bias.data[parent_idx]
                            else:
                                self.output_layer.bias.data[new_idx] = -2.0
                    elif old_num < num_gaussians:
                        self.output_layer.bias.data[old_num:] = -2.0

                elif isinstance(copy_weights, torch.Tensor):
                    selected_weights = old_layer.weight.data[copy_weights]
                    selected_bias = old_layer.bias.data[copy_weights]
                    if selected_weights.shape[0] == num_gaussians:
                        self.output_layer.weight.data.copy_(selected_weights)
                        self.output_layer.bias.data.copy_(selected_bias)
                    else:
                        print(
                            f"Warning: Weight dimension mismatch. Expected {num_gaussians}, got {selected_weights.shape[0]}")
        else:
            with torch.no_grad():
                self.output_layer.bias.data[:] = -2.0

    def forward(self, t, normalize_time=True, time_range=(0, 1)):

        if t.dim() == 0:
            t = t.unsqueeze(0).unsqueeze(0)
        elif t.dim() == 1:
            t = t.unsqueeze(1)

        if normalize_time:
            t = (t - time_range[0]) / (time_range[1] - time_range[0] + 1e-8)
            t = torch.clamp(t, 0, 1)


        t_encoded = self.fourier_encode_time(t, max_freq=15)


        t_projected = self.time_proj(t_encoded)


        features = self.time_encoder(t_projected)
        features = self.feature_layers(features)


        if self.output_layer is not None:
            logits = self.output_layer(features)
            return torch.sigmoid(logits)
        else:
            return None


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree: int, args):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._deformation = deform_network(args)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self._deformation_table = torch.empty(0)
        self.setup_functions()


        self.motion_predictor = MotionStatePredictor().cuda()
        self.last_deformation_output = None
        self.time_range = None

        self.cached_motion_states = None

        self.use_full_deformation = False

    def predict_motion_states(self, time):

        if self.motion_predictor is None or self.motion_predictor.output_layer is None:
            return torch.ones((self._xyz.shape[0],), device="cuda")

        with torch.no_grad():
            time_input = torch.tensor([time], dtype=torch.float32, device="cuda")
            motion_states = self.motion_predictor(
                time_input,
                normalize_time=True,
                time_range=self.time_range if self.time_range is not None else (0, 1)
            )

            return (motion_states.squeeze() > 0.5).float()

    def selective_deformation(self, means3D, scales, rotations, opacity, shs, time):

        if hasattr(self, 'use_full_deformation') and self.use_full_deformation:
            print("    - Performing full deformation (bypass MLP mask)")
            return self._deformation(means3D, scales, rotations, opacity, shs, time)

        motion_states = self.predict_motion_states(time[0, 0].item())
        self.cached_motion_states = motion_states

        means3D_out = means3D.clone()
        scales_out = scales.clone() if scales is not None else None
        rotations_out = rotations.clone() if rotations is not None else None
        opacity_out = opacity.clone()
        shs_out = shs.clone()

        deform_mask = motion_states > 0.5  # bool mask

        if deform_mask.any():

            deform_indices = torch.where(deform_mask)[0]

            means3D_deform = means3D[deform_mask]
            scales_deform = scales[deform_mask] if scales is not None else None
            rotations_deform = rotations[deform_mask] if rotations is not None else None
            opacity_deform = opacity[deform_mask]
            shs_deform = shs[deform_mask]
            time_deform = time[deform_mask]

            means3D_deformed, scales_deformed, rotations_deformed, opacity_deformed, shs_deformed = self._deformation(
                means3D_deform, scales_deform, rotations_deform, opacity_deform, shs_deform, time_deform
            )

            means3D_out[deform_mask] = means3D_deformed
            if scales is not None:
                scales_out[deform_mask] = scales_deformed
            if rotations is not None:
                rotations_out[deform_mask] = rotations_deformed
            opacity_out[deform_mask] = opacity_deformed
            shs_out[deform_mask] = shs_deformed

        return means3D_out, scales_out, rotations_out, opacity_out, shs_out

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._deformation.state_dict(),
            self._deformation_table,
            # self.grid,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,

            self.motion_predictor.state_dict() if self.motion_predictor is not None else None,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
         self._xyz,
         deform_state,
         self._deformation_table,

         # self.grid,
         self._features_dc,
         self._features_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         xyz_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale,
         motion_predictor_state) = model_args[:15] if len(model_args) >= 15 else model_args + (None,)

        self._deformation.load_state_dict(deform_state)
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

        if motion_predictor_state is not None:
            self.motion_predictor.load_state_dict(motion_predictor_state)
            self.motion_predictor.update_output_dim(self._xyz.shape[0])

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float, time_line: int):
        self.spatial_lr_scale = spatial_lr_scale
        # breakpoint()
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._deformation = self._deformation.to("cuda")
        # self.grid = self.grid.to("cuda")
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._deformation_table = torch.gt(torch.ones((self.get_xyz.shape[0]), device="cuda"), 0)


        self.motion_predictor.update_output_dim(fused_point_cloud.shape[0])

        self.time_range = (0, time_line - 1) if time_line > 0 else (0, 1)

    def compute_motion_labels(self, time):

        if self.last_deformation_output is None:
            return torch.zeros((self._xyz.shape[0],), device="cuda")

        deformed_xyz = self.last_deformation_output[0]


        motion_magnitude = torch.norm(deformed_xyz - self._xyz, dim=1)


        d_np = motion_magnitude.detach().cpu().numpy().reshape(-1, 1)

        try:
            from sklearn.mixture import GaussianMixture


            gmm = GaussianMixture(n_components=2, random_state=0, max_iter=50)
            gmm.fit(d_np)

            means = gmm.means_.flatten()
            weights = gmm.weights_.flatten()


            order = np.argsort(means)
            means = means[order]
            weights = weights[order]


            pi_1 = weights[0]


            tau = max(0.5, pi_1)


            posteriors = gmm.predict_proba(d_np)  # [N, 2]
            gamma_i2 = posteriors[:, order[1]]  # 动态分量的后验


            motion_labels_np = (gamma_i2 >= tau).astype(np.float32)
            motion_labels = torch.from_numpy(motion_labels_np).to("cuda")


            if random.random() < 0.01:
                active_ratio = motion_labels.mean().item()
                print(f"[GMM Motion] mean_disp={motion_magnitude.mean().item():.4f}, "
                      f"max_disp={motion_magnitude.max().item():.4f}, "
                      f"π_1={pi_1:.3f}, τ={tau:.3f}, "
                      f"active_ratio={active_ratio:.3f}")

        except ImportError:

            print("[Warning] sklearn not available, using simplified GMM fallback")
            d_flat = motion_magnitude.detach()
            median_d = d_flat.median()


            mask_low = d_flat <= median_d
            mask_high = d_flat > median_d

            if mask_high.sum() > 0 and mask_low.sum() > 0:
                mu_1 = d_flat[mask_low].mean()
                mu_2 = d_flat[mask_high].mean()
                sigma_1 = d_flat[mask_low].std() + 1e-8
                sigma_2 = d_flat[mask_high].std() + 1e-8
                pi_1 = mask_low.float().mean().item()
                pi_2 = 1.0 - pi_1


                log_p1 = -0.5 * ((d_flat - mu_1) / sigma_1) ** 2 - torch.log(sigma_1) + np.log(pi_1 + 1e-8)
                log_p2 = -0.5 * ((d_flat - mu_2) / sigma_2) ** 2 - torch.log(sigma_2) + np.log(pi_2 + 1e-8)
                log_sum = torch.logsumexp(torch.stack([log_p1, log_p2], dim=-1), dim=-1)
                gamma_i2 = torch.exp(log_p2 - log_sum)

                tau = max(0.5, pi_1)
                motion_labels = (gamma_i2 >= tau).float()
            else:
                motion_labels = torch.zeros((self._xyz.shape[0],), device="cuda")

        return motion_labels

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': list(self._deformation.get_mlp_parameters()),
             'lr': training_args.deformation_lr_init * self.spatial_lr_scale, "name": "deformation"},
            {'params': list(self._deformation.get_grid_parameters()),
             'lr': training_args.grid_lr_init * self.spatial_lr_scale, "name": "grid"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}

        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.deformation_scheduler_args = get_expon_lr_func(
            lr_init=training_args.deformation_lr_init * self.spatial_lr_scale,
            lr_final=training_args.deformation_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.deformation_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps)
        self.grid_scheduler_args = get_expon_lr_func(lr_init=training_args.grid_lr_init * self.spatial_lr_scale,
                                                     lr_final=training_args.grid_lr_final * self.spatial_lr_scale,
                                                     lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                     max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr
            if "grid" in param_group["name"]:
                lr = self.grid_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr
            elif param_group["name"] == "deformation":
                lr = self.deformation_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def compute_deformation(self, time):

        deform = self._deformation[:, :, :time].sum(dim=-1)
        xyz = self._xyz + deform
        return xyz

    def load_model(self, path):
        print("loading model from exists{}".format(path))
        weight_dict = torch.load(os.path.join(path, "deformation.pth"), map_location="cuda")
        self._deformation.load_state_dict(weight_dict)
        self._deformation = self._deformation.to("cuda")
        self._deformation_table = torch.gt(torch.ones((self.get_xyz.shape[0]), device="cuda"), 0)
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        if os.path.exists(os.path.join(path, "deformation_table.pth")):
            self._deformation_table = torch.load(os.path.join(path, "deformation_table.pth"), map_location="cuda")
        if os.path.exists(os.path.join(path, "deformation_accum.pth")):
            self._deformation_accum = torch.load(os.path.join(path, "deformation_accum.pth"), map_location="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        if os.path.exists(os.path.join(path, "motion_predictor.pth")):
            motion_state = torch.load(os.path.join(path, "motion_predictor.pth"), map_location="cuda")
            self.motion_predictor.load_state_dict(motion_state)
            self.motion_predictor.update_output_dim(self.get_xyz.shape[0])

    def save_deformation(self, path):
        torch.save(self._deformation.state_dict(), os.path.join(path, "deformation.pth"))
        torch.save(self._deformation_table, os.path.join(path, "deformation_table.pth"))
        torch.save(self._deformation_accum, os.path.join(path, "deformation_accum.pth"))

        if self.motion_predictor is not None:
            torch.save(self.motion_predictor.state_dict(), os.path.join(path, "motion_predictor.pth"))

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree

        if self.motion_predictor is not None:
            self.motion_predictor.update_output_dim(xyz.shape[0])

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) > 1:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._deformation_accum = self._deformation_accum[valid_points_mask]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self._deformation_table = self._deformation_table[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        if self.motion_predictor is not None and self.motion_predictor.output_layer is not None:
            old_weights = self.motion_predictor.output_layer.weight.data.clone()  # [old_num_gaussians, hidden_dim]
            old_bias = self.motion_predictor.output_layer.bias.data.clone()  # [old_num_gaussians]

            new_num_gaussians = self._xyz.shape[0]
            hidden_dim = old_weights.shape[1]
            self.motion_predictor.output_layer = nn.Linear(hidden_dim, new_num_gaussians).cuda()

            with torch.no_grad():
                valid_weights = old_weights[valid_points_mask]  # [new_num_gaussians, hidden_dim]
                valid_bias = old_bias[valid_points_mask]  # [new_num_gaussians]

                self.motion_predictor.output_layer.weight.data = valid_weights
                self.motion_predictor.output_layer.bias.data = valid_bias

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) > 1: continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                              new_rotation, new_deformation_table, parent_indices=None):

        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation,

             }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self._deformation_table = torch.cat([self._deformation_table, new_deformation_table], -1)
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        if self.motion_predictor is not None:
            self.motion_predictor.update_output_dim(
                self._xyz.shape[0],
                copy_weights="expand",
                parent_indices=parent_indices
            )

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):

        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)

        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self.percent_dense * scene_extent)
        if not selected_pts_mask.any():
            return

        # 原有的分裂逻辑
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_deformation_table = self._deformation_table[selected_pts_mask].repeat(N)

        selected_indices = torch.where(selected_pts_mask)[0]
        parent_indices_for_new = selected_indices.repeat_interleave(N)

        old_mlp_weights = None
        old_mlp_bias = None
        if self.motion_predictor is not None and self.motion_predictor.output_layer is not None:
            old_mlp_weights = self.motion_predictor.output_layer.weight.data.clone()
            old_mlp_bias = self.motion_predictor.output_layer.bias.data.clone()

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation,
            new_deformation_table, parent_indices=parent_indices_for_new
        )

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, density_threshold=20, displacement_scale=20,
                          model_path=None, iteration=None, stage=None):

        grads_accum_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)

        selected_pts_mask = torch.logical_and(grads_accum_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_deformation_table = self._deformation_table[selected_pts_mask]

        parent_indices = torch.where(selected_pts_mask)[0]

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
            new_rotation, new_deformation_table, parent_indices=parent_indices
        )

    @property
    def get_aabb(self):
        return self._deformation.get_aabb

    def get_displayment(self, selected_point, point, perturb):
        xyz_max, xyz_min = self.get_aabb
        displacements = torch.randn(selected_point.shape[0], 3).to(selected_point) * perturb
        final_point = selected_point + displacements

        mask_a = final_point < xyz_max
        mask_b = final_point > xyz_min
        mask_c = mask_a & mask_b
        mask_d = mask_c.all(dim=1)
        final_point = final_point[mask_d]

        return final_point, mask_d

    def add_point_by_mask(self, selected_pts_mask, perturb=0):
        selected_xyz = self._xyz[selected_pts_mask]
        new_xyz, mask = self.get_displayment(selected_xyz, self.get_xyz.detach(), perturb)

        new_features_dc = self._features_dc[selected_pts_mask][mask]
        new_features_rest = self._features_rest[selected_pts_mask][mask]
        new_opacities = self._opacity[selected_pts_mask][mask]

        new_scaling = self._scaling[selected_pts_mask][mask]
        new_rotation = self._rotation[selected_pts_mask][mask]
        new_deformation_table = self._deformation_table[selected_pts_mask][mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rotation, new_deformation_table)
        return selected_xyz, new_xyz

    def prune(self, max_grad, min_opacity, extent, max_screen_size):
        prune_mask = (self.get_opacity < min_opacity).squeeze()

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(prune_mask, big_points_vs)

            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def densify(self, max_grad, min_opacity, extent, max_screen_size, density_threshold, displacement_scale,
                model_path=None, iteration=None, stage=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent, density_threshold, displacement_scale, model_path, iteration,
                               stage)
        self.densify_and_split(grads, max_grad, extent)

    def standard_constaint(self):

        means3D = self._xyz.detach()
        scales = self._scaling.detach()
        rotations = self._rotation.detach()
        opacity = self._opacity.detach()
        time = torch.tensor(0).to("cuda").repeat(means3D.shape[0], 1)
        means3D_deform, scales_deform, rotations_deform, _ = self._deformation(means3D, scales, rotations, opacity,
                                                                               time)
        position_error = (means3D_deform - means3D) ** 2
        rotation_error = (rotations_deform - rotations) ** 2
        scaling_erorr = (scales_deform - scales) ** 2
        return position_error.mean() + rotation_error.mean() + scaling_erorr.mean()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor[update_filter, :2], dim=-1,
                                                             keepdim=True)
        self.denom[update_filter] += 1

    @torch.no_grad()
    def update_deformation_table(self, threshold):
        # print("origin deformation point nums:",self._deformation_table.sum())
        self._deformation_table = torch.gt(self._deformation_accum.max(dim=-1).values / 100, threshold)

    def print_deformation_weight_grad(self):
        for name, weight in self._deformation.named_parameters():
            if weight.requires_grad:
                if weight.grad is None:

                    print(name, " :", weight.grad)
                else:
                    if weight.grad.mean() != 0:
                        print(name, " :", weight.grad.mean(), weight.grad.min(), weight.grad.max())
        print("-" * 50)

    def _plane_regulation(self):
        multi_res_grids = self._deformation.deformation_net.grid.grids
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids = [0, 1, 3]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return total

    def _time_regulation(self):
        multi_res_grids = self._deformation.deformation_net.grid.grids
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids = [2, 4, 5]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return total

    def _l1_regulation(self):
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        multi_res_grids = self._deformation.deformation_net.grid.grids

        total = 0.0
        for grids in multi_res_grids:
            if len(grids) == 3:
                continue
            else:
                # These are the spatiotemporal grids
                spatiotemporal_grids = [2, 4, 5]
            for grid_id in spatiotemporal_grids:
                total += torch.abs(1 - grids[grid_id]).mean()
        return total

    def compute_regulation(self, time_smoothness_weight, l1_time_planes_weight, plane_tv_weight):
        return plane_tv_weight * self._plane_regulation() + time_smoothness_weight * self._time_regulation() + l1_time_planes_weight * self._l1_regulation()