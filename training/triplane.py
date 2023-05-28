# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch
from torch_utils import persistence
from training.networks_stylegan2 import ToRGBLayer, SynthesisNetwork, MappingNetwork
from training.networks_stylegan2 import Generator as StyleGAN2Backbone
from training.volumetric_rendering.renderer import ImportanceRenderer
from training.volumetric_rendering.ray_sampler import RaySampler
import dnnlib
import math
from training.networks_stylegan2 import ToRGBLayer, SynthesisNetwork
import numpy as np

@persistence.persistent_class
class TriPlaneGenerator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        sr_num_fp16_res     = 0,
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        rendering_kwargs    = {},
        sr_kwargs = {},
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()        
        bcg_synthesis_kwargs = synthesis_kwargs.copy()
        bcg_synthesis_kwargs["channel_base"] = bcg_synthesis_kwargs["channel_base"] // 2
        self.z_dim=z_dim
        self.c_dim=c_dim
        self.w_dim=w_dim
        self.img_resolution=img_resolution
        self.img_channels=img_channels
        im3dmm_path = rendering_kwargs.get('im3dmm_path', '')
        # proj_mode = rendering_kwargs.get('proj_mode', 'axis')
        self.num_biplanes = 0 if im3dmm_path else 0
        self.num_volplanes = 3 * rendering_kwargs['triplane_depth']
        
        self.feature_dim = 0 if not im3dmm_path else rendering_kwargs.get('feature_dim', 206)
        self.pose_dim = 0 if not im3dmm_path else 106
        self.renderer = ImportanceRenderer(rendering_kwargs)
        self.ray_sampler = RaySampler()
        self.backbone = StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=256, img_channels=32 * (self.num_volplanes + self.num_biplanes), mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        #self.backbone = StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=256, img_channels=32 * self.num_planes, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        self.superresolution = dnnlib.util.construct_class_by_name(class_name=rendering_kwargs['superresolution_module'], channels=32, img_resolution=img_resolution, sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], img_channels=img_channels, **sr_kwargs)
        self.decoder = OSGDecoder(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 32, 'decoder_activation': rendering_kwargs['decoder_activation'], 'pose_perturb_magtinude': rendering_kwargs['pose_perturb_magtinude'], 'pose_embed': rendering_kwargs.get('pose_embed', False)}, self.pose_dim)
        self.bcg_synthesis = None
        if rendering_kwargs.get('use_background', False):
            self.bcg_synthesis = SynthesisNetwork(w_dim, img_resolution=self.superresolution.input_resolution, img_channels=32, **bcg_synthesis_kwargs)
        self.neural_rendering_resolution = 64
        self.rendering_kwargs = rendering_kwargs
        
        self.return_mask = (self.img_channels == 4)
        
        self._last_planes = None

        if rendering_kwargs.get('with_torgb', False):
            self.torgb = ToRGBLayer(32, 3, w_dim)
        else:
            self.torgb = None

    def mapping(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, need_repeat=True):
        cam_label = c[...,:self.c_dim]
        if self.rendering_kwargs['c_gen_conditioning_zero']:
            cam_label = torch.zeros_like(cam_label)
        return self.backbone.mapping(z, cam_label * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas, need_repeat = need_repeat)

    def synthesis(self, ws, c, sr_ws=None, res_out=None, res_skip=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, return_eikonal=False, ws_bcg=None, return_prior=False, return_weighted_geom_delta=False, **synthesis_kwargs):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)

        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(ws, res_out=res_out,res_skip=res_skip, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        # Reshape output into three 32-channel planes
        
        vol_planes = planes[:, :self.num_volplanes*32].view(len(planes), 3, 32 * self.rendering_kwargs['triplane_depth'], planes.shape[-2], planes.shape[-1])
        bi_planes = planes[:, self.num_volplanes*32:].view(len(planes), self.num_biplanes, 32, planes.shape[-2], planes.shape[-1])
        
        # Perform volume rendering
        feature_samples, depth_samples, acc_mask, _, _, _, _, _ = self.renderer(vol_planes, bi_planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs, return_eikonal, features_3dmm=c[...,-self.feature_dim:], return_prior=return_prior, return_weighted_geom_delta=return_weighted_geom_delta) # channels last
            
        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution
        acc_mask = acc_mask.permute(0, 2, 1).reshape(N, -1, H, W) 

        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        # Generate Background
        if self.bcg_synthesis:
            ws_bcg = ws[:,:self.bcg_synthesis.num_ws] if ws_bcg is None else ws_bcg[:,:self.bcg_synthesis.num_ws]
            bcg_image = self.bcg_synthesis(ws_bcg, update_emas=update_emas, **synthesis_kwargs)
            bcg_image = torch.nn.functional.interpolate(bcg_image, size=feature_image.shape[2:],
                    mode='bilinear', align_corners=False, antialias=self.rendering_kwargs['sr_antialias'])
            feature_image = feature_image + (1-acc_mask) * bcg_image

        # Run superresolution to get final image
        if self.decoder.activation == "sigmoid":
            feature_image = feature_image * 2 - 1

        if self.torgb:
            rgb_image = self.torgb(feature_image, ws[:,-1], fused_modconv=False)
            rgb_image = rgb_image.to(dtype=torch.float32, memory_format=torch.contiguous_format)
        else:
            rgb_image = feature_image[:, :3]
        
        if sr_ws is None:
            sr_ws = ws
        sr_image = self.superresolution(rgb_image, feature_image, sr_ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})

        acc_mask = acc_mask*(1 + 2*0.001) - 0.001

        return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image, 'image_mask': acc_mask}
    
    def sample(self, coordinates, directions, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, return_prior=False, return_weighted_geom_delta=False, **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes. 
        ws = self.mapping(z, c[...,:self.c_dim], truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        vol_planes = planes[:, :self.num_volplanes*32].view(len(planes), 3, 32 * self.rendering_kwargs['triplane_depth'], planes.shape[-2], planes.shape[-1])
        bi_planes = planes[:, self.num_volplanes*32:].view(len(planes), self.num_biplanes, 32, planes.shape[-2], planes.shape[-1])
        
        return self.renderer.run_model(vol_planes, bi_planes, self.decoder, coordinates, directions, self.rendering_kwargs, features_3dmm=c[...,-self.feature_dim:], return_prior=return_prior, return_weighted_geom_delta=return_weighted_geom_delta)

    def sample_mixed(self, coordinates, directions, ws, truncation_psi=1, truncation_cutoff=None, update_emas=False, features_3dmm=None, return_prior=False, return_weighted_geom_delta=False, **synthesis_kwargs):
        # Same as sample, but expects latent vectors 'ws' instead of Gaussian noise 'z'
        planes = self.backbone.synthesis(ws, update_emas = update_emas, **synthesis_kwargs)
        #planes = planes.view(len(planes), self.num_planes, 32 * self.rendering_kwargs['triplane_depth'], planes.shape[-2], planes.shape[-1])
        vol_planes = planes[:, :self.num_volplanes*32].view(len(planes), 3, 32 * self.rendering_kwargs['triplane_depth'], planes.shape[-2], planes.shape[-1])
        bi_planes = planes[:, self.num_volplanes*32:].view(len(planes), self.num_biplanes, 32, planes.shape[-2], planes.shape[-1])
        if features_3dmm is None and self.feature_dim != 0:
            features_3dmm = torch.zeros([coordinates.shape[0], self.feature_dim], device=planes.device)
        return self.renderer.run_model(vol_planes, bi_planes, self.decoder, coordinates, directions, self.rendering_kwargs, features_3dmm=features_3dmm, return_prior=return_prior, return_weighted_geom_delta=return_weighted_geom_delta)

    def sample_canonical(self, coordinates, directions, ws, truncation_psi=1, truncation_cutoff=None, update_emas=False, return_prior=False, return_weighted_geom_delta=False, **synthesis_kwargs):
        # Same as sample, but expects latent vectors 'ws' instead of Gaussian noise 'z'
        planes = self.backbone.synthesis(ws, update_emas = update_emas, **synthesis_kwargs)
        #planes = planes.view(len(planes), self.num_planes, 32 * self.rendering_kwargs['triplane_depth'], planes.shape[-2], planes.shape[-1])
        vol_planes = planes[:, :self.num_volplanes*32].view(len(planes), 3, 32 * self.rendering_kwargs['triplane_depth'], planes.shape[-2], planes.shape[-1])
        bi_planes = planes[:, self.num_volplanes*32:].view(len(planes), self.num_biplanes, 32, planes.shape[-2], planes.shape[-1])
        
        return self.renderer.run_model(vol_planes, bi_planes, self.decoder, coordinates, directions, self.rendering_kwargs, features_3dmm=None, return_prior=return_prior, return_weighted_geom_delta=return_weighted_geom_delta)        
    
    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, return_prior=False, return_weighted_geom_delta=False, **synthesis_kwargs):
        # Render a batch of generated images.
        ws = self.mapping(z, c[...,:self.c_dim], truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return self.synthesis(ws, c, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, return_prior=return_prior, return_weighted_geom_delta=return_weighted_geom_delta, **synthesis_kwargs)


from training.networks_stylegan2 import FullyConnectedLayer

class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options, n_pose_features=0):
        super().__init__()
        self.hidden_dim = 64

        self.n_pose_features = n_pose_features
        if n_pose_features > 0:
            self.pose_embed = options.get('pose_embed', False)
            if self.pose_embed:
                n_pose_embedding = 3 * n_pose_features
            else:
                n_pose_embedding = n_pose_features
            self.pose_net = torch.nn.Sequential(
                FullyConnectedLayer(n_pose_embedding, n_features, lr_multiplier=options['decoder_lr_mul']),
                torch.nn.Softplus(),
            )
            n_pose_embedding = n_features
        else:
            self.pose_net = None
            n_pose_embedding = 0

        self.sigma_net = torch.nn.Sequential(
            FullyConnectedLayer(n_features + n_pose_embedding, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1+options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        )

        self.activation = 'sigmoid'
        if options.get('decoder_activation', False):
            self.activation = options['decoder_activation']

        self.pose_perturb_magtinude = options.get('pose_perturb_magtinude', 0)

    def forward(self, sampled_features, sampled_poses):
        # Aggregate features
        if sampled_features.shape[1] != 3:
            split_size = sampled_features.shape[1] // 2
            sampled_features = sampled_features[:, :split_size] * sampled_features[:, split_size:]
        sampled_features = sampled_features.mean(1)

        x = sampled_features
        N, M, C = x.shape
        x = x.view(N*M, C)

        if self.pose_net is not None:
            sampled_poses = sampled_poses[..., :1, -self.n_pose_features:]
            sampled_poses[..., :100] += torch.randn_like(sampled_poses[:, :1, :100]) * self.pose_perturb_magtinude
            sampled_poses[..., 100:103] += torch.randn_like(sampled_poses[:, :1, 100:103]) * self.pose_perturb_magtinude * 0.2
            sampled_poses[..., 103:] += torch.randn_like(sampled_poses[:, :1, 103:]) * self.pose_perturb_magtinude
            N, _, C = sampled_poses.shape
            sampled_poses = sampled_poses.view(N, C)
            if self.pose_embed:
                sampled_poses = torch.cat([torch.sin(torch.pi*sampled_poses), torch.cos(torch.pi*sampled_poses), sampled_poses], dim = -1)
            pose_embedding = self.pose_net(sampled_poses)
            pose_embedding = pose_embedding.repeat_interleave(M, 0)
            x_pose = torch.cat([x, pose_embedding], axis = -1)
        else:
            x_pose = x

        out = self.sigma_net(x_pose)
        out = out.view(N, M, -1)
        sigma, rgb = out[..., :1], out[..., 1:]

        if self.activation == 'sigmoid':
            rgb = torch.sigmoid(rgb)*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        elif self.activation == 'lrelu':
            rgb = torch.nn.functional.leaky_relu(rgb, 0.2, inplace=True) * math.sqrt(2)

        # sigma = x[..., 0:1]
        return {'rgb': rgb, 'sigma': sigma}
