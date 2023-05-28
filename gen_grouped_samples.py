
# The code was adapted from EG3D (https://github.com/NVlabs/eg3d)
import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import torch
from tqdm import tqdm

import legacy

from camera_utils import LookAtPoseSampler
import glob
import numpy as np
import PIL.Image

#----------------------------------------------------------------------------

def layout_grid(img, grid_w=None, grid_h=1, float_to_uint8=True, chw_to_hwc=True, to_numpy=True):
    batch_size, channels, img_h, img_w = img.shape
    if grid_w is None:
        grid_w = batch_size // grid_h
    assert batch_size == grid_w * grid_h
    if float_to_uint8:
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img.reshape(grid_h, grid_w, channels, img_h, img_w)
    img = img.permute(2, 0, 3, 1, 4)
    img = img.reshape(channels, grid_h * img_h, grid_w * img_w)
    if chw_to_hwc:
        img = img.permute(1, 2, 0)
    if to_numpy:
        img = img.cpu().numpy()
    return img

def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size

def generate_group_images(G, psi=1, truncation_cutoff=14, cfg='FFHQ',device=torch.device('cuda'), label_pool='', outdir='', num_groups=1000, num_samples=10):

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
    cam_radius = G.rendering_kwargs.get('avg_camera_radius', 1.0)
    conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)

    feature_dim = G.rendering_kwargs.get('feature_dim', 0)
    if cfg == 'ffhq':
        intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    else:
        intrinsics = torch.tensor([[1015./224, 0.0, 0.5],[0.0, 1015./224, 0.5], [0.0, 0.0, 1.0]], device=device)

    frame_3dmms = np.load(label_pool)

    for frame_idx in tqdm(range(num_groups)):
        param_seed = np.random.randint(0, len(frame_3dmms))
        frame_3dmm = frame_3dmms[param_seed, :]

        frame_3dmm = torch.from_numpy(frame_3dmm).to(device).float()
        shape = frame_3dmm[..., 25:125]
        conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9), frame_3dmm[...,-feature_dim:].reshape(-1, feature_dim)], 1)
        
        zs =  torch.from_numpy(np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in [param_seed]])).to(device)
        ws = G.mapping(zs, conditioning_params, truncation_psi=psi, truncation_cutoff=truncation_cutoff)

        img_outdir = os.path.join(outdir, 'group%04d' % frame_idx)
        if not os.path.exists(img_outdir):
            os.mkdir(img_outdir)

        for sample_idx in range(num_samples):
            exp_seed =  np.random.randint(0, len(frame_3dmms))         
            exp = np.copy(frame_3dmms[exp_seed, :])
            
            # add random jaw motions
            exp[..., -3] += np.random.normal() * 0.1
            exp[..., -2] += np.random.normal() * 0.15
            exp[..., -1] += np.random.normal() * 0.1
            new_frame_3dmm = torch.from_numpy(exp).to(device).float()
            new_frame_3dmm[..., 25:125] = shape
                
            pitch = 0.1 * np.random.uniform(-1, 1)
            yaw = 0.25 * np.random.uniform(-1, 1)

            cam2world_pose = LookAtPoseSampler.sample(3.14/2 + yaw,
                                                3.14/2 -0.05 + pitch,
                                                cam_pivot, radius=cam_radius, device=device)

            conditioning_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9),new_frame_3dmm[...,-feature_dim:].reshape(-1, feature_dim)], 1)

            out = G.synthesis(ws, conditioning_params.repeat([len(ws), 1]), noise_mode='const')
            img = out['image'][0].permute(1, 2, 0)
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            output_path = os.path.join(img_outdir, 'seed%06d_%06d_%02d.png' % (param_seed, exp_seed, sample_idx))
            PIL.Image.fromarray(img.cpu().numpy(), 'RGB').save(output_path)

def parse_range(s: Union[str, List[int]]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        if m := range_re.match(p):
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_tuple(s: Union[str, Tuple[int,int]]) -> Tuple[int, int]:
    '''Parse a 'M,N' or 'MxN' integer tuple.

    Example:
        '4x2' returns (4,2)
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    if m := re.match(r'^(\d+)[x,](\d+)$', s):
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f'cannot parse tuple {s}')

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--outdir', help='Output directory', type=str, required=True, metavar='DIR')
@click.option('--cfg', help='Config', type=click.Choice(['ffhq', 'cats', 'ffhq_3dmm']), required=False, metavar='STR', default='ffhq', show_default=True)
@click.option('--sample_mult', 'sampling_multiplier', type=float, help='Multiplier for depth sampling in volume rendering', default=1, show_default=True)
@click.option('--nrr', type=int, help='Neural rendering resolution override', default=None, show_default=True)
@click.option('--label-pool', help='label pool', type=str, required=False, default='')
@click.option('--num-groups', 'num_groups', type=int, help='number of groups', default=1000, show_default=True)
@click.option('--num-samples', 'num_samples', type=int, help='number of samples per group', default=10, show_default=True)

def generate_images(
    network_pkl: str,
    truncation_psi: float,
    truncation_cutoff: int,
    outdir: str,
    cfg: str,
    sampling_multiplier: float,
    nrr: Optional[int],
    label_pool: str,
    num_groups: int,
    num_samples: int,
):
    """Render a latent vector interpolation video.

    Examples:

    \b
    # Render a 4x2 grid of interpolations for seeds 0 through 31.
    python gen_video.py --output=lerp.mp4 --trunc=1 --seeds=0-31 --grid=4x2 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl

    Animation length and seed keyframes:

    The animation length is either determined based on the --seeds value or explicitly
    specified using the --num-keyframes option.

    When num keyframes is specified with --num-keyframes, the output video length
    will be 'num_keyframes*w_frames' frames.

    If --num-keyframes is not specified, the number of seeds given with
    --seeds must be divisible by grid size W*H (--grid).  In this case the
    output video length will be '# seeds/(w*h)*w_frames' frames.
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        network = legacy.load_network_pkl(f)
        G = network['G_ema'].to(device) # type: ignore

    G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * sampling_multiplier)
    G.rendering_kwargs['depth_resolution_importance'] = int(G.rendering_kwargs['depth_resolution_importance'] * sampling_multiplier)
    G.rendering_kwargs['det_sampling'] = True
    G.decoder.pose_perturb_magtinude = 0
    if nrr is not None: G.neural_rendering_resolution = nrr

    if truncation_cutoff == 0:
        truncation_psi = 1.0 # truncation cutoff of 0 means no truncation anyways
    if truncation_psi == 1.0:
        truncation_cutoff = 14 # no truncation so doesn't matter where we cutoff

    generate_group_images(G=G, psi=truncation_psi, truncation_cutoff=truncation_cutoff, cfg=cfg, label_pool=label_pool, outdir=outdir, num_groups=num_groups, num_samples=num_samples)
 
if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter


#----------------------------------------------------------------------------
