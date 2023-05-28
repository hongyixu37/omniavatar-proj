# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
The renderer is a module that takes in rays, decides where to sample along each
ray, and computes pixel colors using the volume rendering equation.
"""

import math
import torch
import torch.nn as nn

from training.volumetric_rendering.ray_marcher import MipRayMarcher2
from training.volumetric_rendering import math_utils
from training.im3dmm_network import IGRModel

import trimesh
import numpy as np

def generate_planes():
    """
    Defines planes by the three vectors that form the "axes" of the
    plane. Should work with arbitrary number of planes and planes of
    arbitrary orientation.
    """
    return torch.tensor([[[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]],
                            [[1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]],
                            [[0, 1, 0],
                            [0, 0, 1],
                            [1, 0, 0]]], dtype=torch.float32)

def project_onto_planes(n_planes, coordinates, vol_proj=True):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.

    Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    """
    N, M, C = coordinates.shape
    
    if vol_proj:
        xyz, yzx, xzy = coordinates[...,[0,1,2]], coordinates[...,[1,2,0]], coordinates[...,[0,2,1]]
        projections = torch.stack([xyz, yzx, xzy], dim = 1).view(N * n_planes, M, 3)
    else:
        # zs, ys, xs = coordinates[...,[0,1]], coordinates[...,[1,2]], coordinates[...,[0,2]]
        zs, ys, xs = coordinates[...,[2,3]], coordinates[...,[1,3]], coordinates[...,[0,3]]
        projections = torch.stack([zs, ys, xs], dim = 1).view(N * n_planes, M, 2)
    
    return projections

def grid_sample(image, optical):
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW-1);
    iy = ((iy + 1) / 2) * (IH-1);
    with torch.no_grad():
        ix_nw = torch.floor(ix);
        iy_nw = torch.floor(iy);
        ix_ne = ix_nw + 1;
        iy_ne = iy_nw;
        ix_sw = ix_nw;
        iy_sw = iy_nw + 1;
        ix_se = ix_nw + 1;
        iy_se = iy_nw + 1;

    nw = (ix_se - ix)    * (iy_se - iy)
    ne = (ix    - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix)    * (iy    - iy_ne)
    se = (ix    - ix_nw) * (iy    - iy_nw)

    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW-1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH-1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW-1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH-1, out=iy_ne)

        torch.clamp(ix_sw, 0, IW-1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH-1, out=iy_sw)

        torch.clamp(ix_se, 0, IW-1, out=ix_se)
        torch.clamp(iy_se, 0, IH-1, out=iy_se)

    image = image.view(N, C, IH * IW)


    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) + 
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val

def sample_from_volplanes(plane_features, coordinates, mode='bilinear', padding_mode='zeros', box_warp=None, triplane_depth=1, custimized_sample_op=False):
    assert padding_mode == 'zeros'
    N, n_planes, CD, H, W = plane_features.shape
    _, M, _ = coordinates.shape
    C, D = CD // triplane_depth, triplane_depth
    plane_features = plane_features.reshape(N*n_planes, C, D, H, W)

    coordinates = (2/torch.tensor(box_warp, device=coordinates.device, dtype=torch.float32)) * coordinates # TODO: add specific box bounds

    projected_coordinates = project_onto_planes(n_planes, coordinates).unsqueeze(1).unsqueeze(2)
    if custimized_sample_op:
        output_features = grid_sample(plane_features, projected_coordinates.float()).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    else:
        output_features = torch.nn.functional.grid_sample(plane_features, projected_coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False).permute(0, 4, 3, 2, 1).reshape(N, n_planes, M, C)

    return output_features

def sample_from_biplanes(plane_features, coordinates, mode='bilinear', padding_mode='zeros', box_warp=None, custimized_sample_op=False):
    assert padding_mode == 'zeros'
    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape
    plane_features = plane_features.reshape(N*n_planes, C, H, W)

    coordinates = (2/torch.tensor(box_warp, device=coordinates.device, dtype=torch.float32)) * coordinates # TODO: add specific box bounds

    projected_coordinates = project_onto_planes(n_planes, coordinates, vol_proj=False).unsqueeze(1)
    if custimized_sample_op:
        output_features = grid_sample(plane_features, projected_coordinates.float()).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    else:
        output_features = torch.nn.functional.grid_sample(plane_features, projected_coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)

    # output_features = torch.nn.functional.grid_sample(plane_features, projected_coordinates.float()).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    return output_features

def sample_from_3dgrid(grid, coordinates):
    """
    Expects coordinates in shape (batch_size, num_points_per_batch, 3)
    Expects grid in shape (1, channels, H, W, D)
    (Also works if grid has batch size)
    Returns sampled features of shape (batch_size, num_points_per_batch, feature_channels)
    """
    batch_size, n_coords, n_dims = coordinates.shape
    sampled_features = torch.nn.functional.grid_sample(grid.expand(batch_size, -1, -1, -1, -1),
                                                       coordinates.reshape(batch_size, 1, 1, -1, n_dims),
                                                       mode='bilinear', padding_mode='zeros', align_corners=False)
    N, C, H, W, D = sampled_features.shape
    sampled_features = sampled_features.permute(0, 4, 3, 2, 1).reshape(N, H*W*D, C)
    return sampled_features

class ImportanceRenderer(torch.nn.Module):
    def __init__(self, rendering_kwargs):
        super().__init__()
        self.ray_marcher = MipRayMarcher2()
        self.plane_axes = generate_planes()
        self.with_sdf = rendering_kwargs['with_sdf']
        self.delta_geom = rendering_kwargs['delta_geom']
        self.template_sdf = self.load_template_sdf(rendering_kwargs)
        self.im3dmm = self.load_im3dmm(rendering_kwargs)
        if self.with_sdf or self.im3dmm is not None:
            self.sigmoid_beta = nn.Parameter(0.001 * torch.ones(1))

    def load_template_sdf(self, rendering_kwargs):
        template_sdf_path = rendering_kwargs.get('template_sdf_path', '')
        if not template_sdf_path:
            return None
        
        bbox = torch.tensor(rendering_kwargs['bbox'])
        template_sdf_network = IGRModel(
            bbox=bbox, 
            output_dim=1, 
            intermediate_dims=rendering_kwargs['template_intermediate_dims'], 
            skip_connection_layers=rendering_kwargs['skip_connection_layers'],
            feature_dim=0)
        saved_model_state = torch.load(template_sdf_path)
        template_sdf_network.load_state_dict(saved_model_state['model_state_dict'])
        for param in template_sdf_network.parameters():
            param.requires_grad = False

        return template_sdf_network

    def load_im3dmm(self, rendering_kwargs):
        im3dmm_path = rendering_kwargs.get('im3dmm_path', '')
        if not im3dmm_path:
            return None
        
        bbox = torch.tensor(rendering_kwargs['bbox'])
        output_dim = 4

        im3dmm = IGRModel(
            intermediate_dims=rendering_kwargs['intermediate_dims'], 
            skip_connection_layers=rendering_kwargs['skip_connection_layers'],
            bbox=bbox, 
            feature_dim=rendering_kwargs['feature_dim'],
            output_dim=output_dim,
            template_sdf=self.template_sdf)

        saved_model_state = torch.load(im3dmm_path)
        im3dmm.load_state_dict(saved_model_state['model_state_dict'])
        for param in im3dmm.parameters():
            param.requires_grad = False
        
        return im3dmm

    def forward(self, vol_planes, bi_planes, decoder, ray_origins, ray_directions, rendering_options, return_eikonal=False, features_3dmm=None, return_prior=False, return_weighted_geom_delta=False):
        self.plane_axes = self.plane_axes.to(ray_origins.device)

        det = rendering_options.get('det_sampling', False)
        if rendering_options['ray_start'] == rendering_options['ray_end'] == 'auto':
            ray_start, ray_end = math_utils.get_ray_limits_box(ray_origins, ray_directions, box_side_length=rendering_options['box_warp'])
            is_ray_valid = ray_end > ray_start
            if torch.any(is_ray_valid).item():
                ray_start[~is_ray_valid] = ray_start[is_ray_valid].min()
                ray_end[~is_ray_valid] = ray_start[is_ray_valid].max()
            depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end, rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'], det=det)
        else:
            # Create stratified depth samples
            depths_coarse = self.sample_stratified(ray_origins, rendering_options['ray_start'], rendering_options['ray_end'], rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'], det=det)

        batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape

        # Coarse Pass
        sample_coordinates = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
        sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)

        out = self.run_model(vol_planes, bi_planes, decoder, sample_coordinates, sample_directions, rendering_options, return_eikonal, features_3dmm, return_prior, return_weighted_geom_delta)
        colors_coarse = out['rgb']
        densities_coarse = out['sigma']
        colors_coarse = colors_coarse.reshape(batch_size, num_rays, samples_per_ray, colors_coarse.shape[-1])
        densities_coarse = densities_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)

        eikonal_term = None
        if return_eikonal:
            eikonal_term_coarse = out['eikonal']
            eikonal_term_coarse = eikonal_term_coarse.reshape(batch_size, num_rays, samples_per_ray, eikonal_term_coarse.shape[-1])

        if self.with_sdf:
            geom_densities = out['sdf'].reshape(batch_size, num_rays, samples_per_ray, 1)
        else:
            geom_densities = densities_coarse

        # Fine Pass
        N_importance = rendering_options['depth_resolution_importance']
        if N_importance > 0:
            _, _, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)

            # depths_fine = self.sample_importance(depths_coarse, alpha_entropy, N_importance)
            depths_fine = self.sample_importance(depths_coarse, weights, N_importance, det=det)

            sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, N_importance, -1).reshape(batch_size, -1, 3)
            sample_coordinates = (ray_origins.unsqueeze(-2) + depths_fine * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)

            out = self.run_model(vol_planes, bi_planes, decoder, sample_coordinates, sample_directions, rendering_options, return_eikonal, features_3dmm, return_prior, return_weighted_geom_delta)
            colors_fine = out['rgb']
            densities_fine = out['sigma']
            colors_fine = colors_fine.reshape(batch_size, num_rays, N_importance, colors_fine.shape[-1])
            densities_fine = densities_fine.reshape(batch_size, num_rays, N_importance, 1)
            
            if return_eikonal:
                eikonal_term_fine = out['eikonal']
                eikonal_term_fine = eikonal_term_fine.reshape(batch_size, num_rays, N_importance, eikonal_term_fine.shape[-1])
                eikonal_term = torch.cat([eikonal_term_coarse, eikonal_term_fine], dim = -2)

            all_depths, all_colors, all_densities = self.unify_samples(depths_coarse, colors_coarse, densities_coarse,
                                                                  depths_fine, colors_fine, densities_fine)

            # Aggregate
            rgb_final, depth_final, weights = self.ray_marcher(all_colors, all_densities, all_depths, rendering_options)
            if self.with_sdf:
                geom_densities_fine = out['sdf'].reshape(batch_size, num_rays, N_importance, 1)
                geom_densities = torch.cat([geom_densities, geom_densities_fine], dim=-2)
            else:
                geom_densities = all_densities

        else:
            rgb_final, depth_final, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)
            if return_eikonal:
                eikonal_term = eikonal_term_coarse

        weights_mask = torch.sum(weights, dim=-2)
         
        if not return_eikonal:
            eikonal_term = geom_densities

        return rgb_final, depth_final, weights_mask, eikonal_term, None, None, None, None

    def sdf_activation(self, input):
        sigma = torch.sigmoid(input / self.sigmoid_beta) / self.sigmoid_beta

        return sigma

    def get_eikonal_term(self, pts, sdf):
        eikonal_term = torch.autograd.grad(outputs=sdf, inputs=pts,
                                     grad_outputs=torch.ones_like(sdf, requires_grad=False),
                                     create_graph=True,
                                     retain_graph=True,
                                     only_inputs=True)[0]

        return eikonal_term

    def run_model(self, vol_planes, bi_planes, decoder, sample_coordinates, sample_directions, options, return_eikonal=False, features_3dmm=None, return_prior=False, return_weighted_geom_delta=False):
        if return_eikonal and self.with_sdf:
            sample_coordinates.requires_grad = True

        im3dmm_features = None
        residual_magnitude = 0
        if self.im3dmm is not None and features_3dmm is not None:
            feature_dim = features_3dmm.shape[-1]
            if feature_dim != 0:
                with torch.no_grad():
                    num_points = sample_coordinates.shape[-2]
                    query_coordinates = sample_coordinates
                    im3dmm_features = features_3dmm[:, None, :].repeat([1, num_points, 1]).view(*sample_coordinates.shape[:-1],feature_dim)
                    query = {
                        'points': query_coordinates,
                        'return_distance_only': False,
                        'return_warping_only': True,
                        'features': im3dmm_features,
                    }
                    sample_coordinates, _, _ = self.im3dmm(query)
                    if sample_coordinates.shape[-1] > 3:
                        sample_coordinates = sample_coordinates[..., [1, 2, 3, 0]]

                
        sampled_features = sample_from_volplanes(
            vol_planes, sample_coordinates[...,:3], padding_mode='zeros', box_warp=options['box_warp'], triplane_depth=options['triplane_depth'], custimized_sample_op=self.with_sdf)

        out = decoder(sampled_features, im3dmm_features)
        
        out['residual'] = residual_magnitude
        if self.with_sdf:
            sdf_delta = out['sigma']
            if sample_coordinates.shape[-1] > 3 and self.delta_geom:
                sdf = sample_coordinates[..., -1:] + sdf_delta
            else:
                sdf = sdf_delta

            out['sdf'] = sdf
            out['sigma'] = self.sdf_activation(-sdf)

            if return_eikonal:
                eikonal_term = self.get_eikonal_term(sample_coordinates, sdf)
                out['eikonal'] = torch.cat([eikonal_term, sdf], dim = -1)
            else:
                out['eikonal'] = None
        else:
            if sample_coordinates.shape[-1] > 3:    
                prior_sdf = sample_coordinates[..., -1:]
                sdf_sigma = self.sdf_activation(-prior_sdf)
                if return_prior:
                    out['sigma'] = sdf_sigma
                    if self.delta_geom:
                        out['geom_delta'] = out['sigma']
                    else:
                        out['geom_delta'] = out['sigma'] - sdf_sigma    
                elif self.delta_geom:
                    out['sigma'] = sdf_sigma + out['sigma']
                    if return_weighted_geom_delta:
                        out['geom_delta'] = out['sigma']
                else:
                    out['geom_delta'] = out['sigma'] - sdf_sigma                          

            if options.get('density_noise', 0) > 0:
                out['sigma'] += torch.randn_like(out['sigma']) * options['density_noise']
            out['eikonal'] = None
        
        return out

    def sort_samples(self, all_depths, all_colors, all_densities):
        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))
        return all_depths, all_colors, all_densities

    def unify_samples(self, depths1, colors1, densities1, depths2, colors2, densities2):
        all_depths = torch.cat([depths1, depths2], dim = -2)
        all_colors = torch.cat([colors1, colors2], dim = -2)
        all_densities = torch.cat([densities1, densities2], dim = -2)

        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))

        return all_depths, all_colors, all_densities

    def sample_stratified(self, ray_origins, ray_start, ray_end, depth_resolution, disparity_space_sampling=False, det=False):
        """
        Return depths of approximately uniformly spaced samples along rays.
        """
        N, M, _ = ray_origins.shape
        if disparity_space_sampling:
            depths_coarse = torch.linspace(0,
                                    1,
                                    depth_resolution,
                                    device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
            depth_delta = 1/(depth_resolution - 1)
            if not det:
                depths_coarse += torch.rand_like(depths_coarse) * depth_delta
            depths_coarse = 1./(1./ray_start * (1. - depths_coarse) + 1./ray_end * depths_coarse)
        else:
            if type(ray_start) == torch.Tensor:
                depths_coarse = math_utils.linspace(ray_start, ray_end, depth_resolution).permute(1,2,0,3)
            else:
                depths_coarse = torch.linspace(ray_start, ray_end, depth_resolution, device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)

            if not det:    
                depth_delta = (ray_end - ray_start)/(depth_resolution - 1)
                depths_coarse += torch.rand_like(depths_coarse) * depth_delta

        return depths_coarse

    def sample_importance(self, z_vals, weights, N_importance, det=False):
        """
        Return depths of importance sampled points along rays. See NeRF importance sampling for more.
        """
        with torch.no_grad():
            batch_size, num_rays, samples_per_ray, _ = z_vals.shape

            z_vals = z_vals.reshape(batch_size * num_rays, samples_per_ray)
            weights = weights.reshape(batch_size * num_rays, -1) # -1 to account for loss of 1 sample in MipRayMarcher

            # smooth weights
            weights = torch.nn.functional.max_pool1d(weights.unsqueeze(1).float(), 2, 1, padding=1)
            weights = torch.nn.functional.avg_pool1d(weights, 2, 1).squeeze()
            weights = weights + 0.01

            z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
            importance_z_vals = self.sample_pdf(z_vals_mid, weights[:, 1:-1],
                                             N_importance, det=det).detach().reshape(batch_size, num_rays, N_importance, 1)
        return importance_z_vals

    def sample_pdf(self, bins, weights, N_importance, det=False, eps=1e-5):
        """
        Sample @N_importance samples from @bins with distribution defined by @weights.
        Inputs:
            bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
            weights: (N_rays, N_samples_)
            N_importance: the number of samples to draw from the distribution
            det: deterministic or not
            eps: a small number to prevent division by zero
        Outputs:
            samples: the sampled samples
        """
        N_rays, N_samples_ = weights.shape
        weights = weights + eps # prevent division by zero (don't do inplace op!)
        pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
        cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
        cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1)
                                                                   # padded to 0~1 inclusive

        if det:
            u = torch.linspace(0, 1, N_importance, device=bins.device)
            u = u.expand(N_rays, N_importance)
        else:
            u = torch.rand(N_rays, N_importance, device=bins.device)
        u = u.contiguous()

        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp_min(inds-1, 0)
        above = torch.clamp_max(inds, N_samples_)

        inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
        cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
        bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

        denom = cdf_g[...,1]-cdf_g[...,0]
        denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                             # anyway, therefore any value for it is fine (set to 1 here)

        samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
        return samples