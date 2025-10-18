# coding=utf-8
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import math

from scene.gaussian_model import GaussianModel
from ever.splinetracers.fast_ellipsoid_splinetracer import trace_rays
# from splinetracer.splinetracers.ellipsoid_splinetracer import trace_rays
MAX_ITERS = 400
from ever.eval_sh import eval_sh as eval_sh2
from utils.sh_utils import eval_sh, RGB2SH, SH2RGB
from kornia import create_meshgrid
import numpy as np
from icecream import ic
from scene.dataset_readers import ProjectionType
from plyfile import PlyData, PlyElement

def get_ray_directions(H, W, focal, center=None, random=True):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        H, W, focal: image height, width and focal length
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]# + 0.5
    if random:
        grid = grid + torch.rand_like(grid)
    else:
        grid = grid + 0.5

    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    cent = center if center is not None else [W / 2, H / 2]
    directions = torch.stack(
        [(i - cent[0]) / focal[0], (j - cent[1]) / focal[1], torch.ones_like(i)], -1
    )  # (H, W, 3)

    return directions

def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate
    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:3, :3].T  # (H, W, 3)
    # rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d

def camera2rays_full(view, **kwargs):
    w = view.image_width  # // 4
    h = view.image_height  # // 4
    # y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    device = torch.device('cuda')

    x, y = torch.meshgrid(torch.arange(w, device=device), torch.arange(h, device=device), indexing='xy')

    fx = 0.5 * w / np.tan(0.5 * view.FoVx)  # original focal length
    fy = 0.5 * h / np.tan(0.5 * view.FoVy)  # original focal length
    pixtocams = torch.eye(3, device=device)
    pixtocams[0, 0] = 1/fx
    pixtocams[1, 1] = 1/fy
    pixtocams[0, 2] = -w/2/fx
    pixtocams[1, 2] = -h/2/fy

    T = torch.linalg.inv(view.world_view_transform.T).to(device)
    origins, _, directions, _, _ = camera_utils_zipnerf.pixels_to_rays(
        x.reshape(-1), y.reshape(-1),
        pixtocams.reshape(1, 3, 3),
        T[:3].reshape(1, 3, 4),
        camtype=view.model,
        distortion_params=view.distortion_params,
        xnp=torch
    )
    origins = origins.float().cuda().contiguous()
    directions = directions.float().cuda().contiguous()
    # ic(camera2rays(view)[1])
    # ic(directions)
    return origins, directions

def camera2rays(view, **kwargs):
    w = view.image_width
    h = view.image_height

    fx = 0.5 * w / math.tan(0.5 * view.FoVx)  # original focal length
    fy = 0.5 * h / math.tan(0.5 * view.FoVy)  # original focal length

    directions = get_ray_directions(h, w, [fx, fy], **kwargs).cuda()  # (h, w, 3)
    directions = (directions / torch.norm(directions, dim=-1, keepdim=True))

    T = torch.linalg.inv(view.world_view_transform.T.cuda())
    rays_o, rays_d = get_rays(
        directions,
        T,
    )  # both (h*w, 3)
    rays_o = (rays_o).contiguous()
    return rays_o, rays_d

def output_ply(point_cloud, colors, filename):
    point_cloud = point_cloud.detach().cpu().numpy()
    colors = colors.detach().cpu().numpy()

    vertex = np.zeros(point_cloud.shape[0], dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ])

    vertex['x'] = point_cloud[:, 0]
    vertex['y'] = point_cloud[:, 1]
    vertex['z'] = point_cloud[:, 2]
    vertex['red'] = (colors[:, 0] * 255).astype(np.uint8)
    vertex['green'] = (colors[:, 1] * 255).astype(np.uint8)
    vertex['blue'] = (colors[:, 2] * 255).astype(np.uint8)

    ply_data = PlyData([PlyElement.describe(vertex, 'vertex')], text=True)

    ply_data.write(filename)

def splinerender(
    view,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    random=False,
    tmin=None,
    tmax=1e7,
    deformation=None,
    stage="coarse",
    now_iteration=0
):
    if stage == "coarse":
        return splinerender_coarse(
            view,
            pc,
            pipe,
            bg_color,
            scaling_modifier,
            override_color,
            random,
            tmin,
            tmax,
            deformation,
            stage,
            now_iteration
        )
    else:
        return splinerender_fine(
            view,
            pc,
            pipe,
            bg_color,
            scaling_modifier,
            override_color,
            random,
            tmin,
            tmax,
            deformation,
            stage,
            now_iteration
        )

def splinerender_coarse(
    view,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    random=False,
    tmin=None,
    tmax=1e7,
    deformation=None,
    stage="coarse",
    now_iteration=0
):
    device = pc.get_xyz.device
    if view.model == ProjectionType.PERSPECTIVE:
        rays_o, rays_d = camera2rays(view, random=random)
    else:
        rays_o, rays_d = camera2rays_full(view, random=False)

    means2D = torch.zeros_like(pc.get_xyz[..., :2])
    means2D.requires_grad = True

    w = view.image_width  # // 4
    h = view.image_height  # // 4

    fx = 0.5 * w / np.tan(0.5 * view.FoVx)  # original focal length
    fy = 0.5 * h / np.tan(0.5 * view.FoVy)  # original focal length
    K = torch.tensor([
        [fx, 0, w/2, 0],
        [0, fy, h/2, 0],
        [0, 0, 1, 0],
    ], device="cuda").float()
    invK = torch.tensor([
        [1/fx, 0, -w/2/fx],
        [0, 1/fy, -h/2/fy],
        [0, 0, 1],
        [0, 0, 0],
    ], device="cuda").float()
    device = "cuda"
    
    wct = view.world_view_transform.cuda().float()
    full_wct = torch.eye(4, device="cuda")
    full_wct[:, :3] = wct @ K.T

    shs = pc.get_features
    # shs[:, (pc.active_sh_degree+1)**2:] = 0
    # ic(shs.shape, shs[:, :(pc.active_sh_degree+1)**2].shape)
    if pipe.enable_GLO:
        if view.glo_vector is not None:
            glo_vector = view.glo_vector
        else:
            glo_vector = torch.zeros((1, 64), device='cuda')
        shs = pc.glo_network(
            glo_vector.reshape(1, -1), shs.reshape(shs.shape[0], -1)
        ).reshape(shs.shape)

    cam_pos = view.camera_center.to(device)
    T = torch.linalg.inv(wct.T)
    v = T[:3, 2]
    net_color = eval_sh2(pc.get_xyz, shs, cam_pos, pc.active_sh_degree)
    # ic(net_color, SH2RGB(features))
    net_color = torch.nn.functional.softplus(net_color, beta=10)
    features = RGB2SH(net_color).reshape(-1, 1, 3)

    per_point_2d_filter_scale = torch.zeros(pc._xyz.shape[0], device=pc._xyz.device)

    if trace_rays.uses_density:
        scales, density = pc.get_scale_and_density_for_rendering(per_point_2d_filter_scale, scaling_modifier)
    else:
        scales, density = pc.get_scale_and_opacity_for_rendering(per_point_2d_filter_scale, scaling_modifier)
    tmin = pc.tmin if tmin is None else tmin

    ######################################### Deformation
    time = torch.tensor(view.time).to(pc.get_xyz.device).repeat(pc.get_xyz.shape[0],1)

    if stage[:4] == "fine":
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = deformation(pc.get_xyz, pc._scaling, 
                                                                    pc._rotation, density, features,
                                                                    time, stage, now_iteration)
    else:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = \
            pc.get_xyz, pc._scaling, pc._rotation, density, shs

    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)

    out, extras = trace_rays(
        means3D_final, # pc.get_xyz,
        scales_final,
        rotations_final,
        density,
        features,
        rays_o,
        rays_d,
        tmin,
        tmax,
        100,
        means2D,
        full_wct.reshape(1, 4, 4),
        max_iters=MAX_ITERS,
        return_extras=True,
    )

    torch.cuda.synchronize()
    radii = torch.ones_like(means2D[..., 0])

    rendered_image = out[:, :3].T.reshape(3, view.image_height, view.image_width)
    num_pixels = (extras['touch_count'] // 2)

    # aspect_ratio = scales.max(dim=-1).values / scales.min(dim=-1).values
    side_length = (num_pixels).float().sqrt() #/ aspect_ratio # mul by 2 to get to rect, then sqrt
    radii = side_length / 2 * np.sqrt(2) * 2.5 * 5

    return {
        "render": rendered_image,
        "viewspace_points": means2D,
        "visibility_filter": num_pixels >= 4,
        "touch_count": extras['touch_count'],
        "radii": radii, # match gaussian radius
        "iters": extras["iters"].reshape(view.image_height, view.image_width),
        "opacity": out[:, 3].reshape(-1, 1),
        "distortion_loss": out[:, 4].reshape(-1, 1),
    }

def splinerender_fine(
    view,
    pc,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    random=False,
    tmin=None,
    tmax=1e7,
    deformation=None,
    stage="coarse",
    now_iteration=0
):
    device = pc[0].get_xyz.device
    if view.model == ProjectionType.PERSPECTIVE:
        rays_o, rays_d = camera2rays(view, random=random)
    else:
        rays_o, rays_d = camera2rays_full(view, random=False)

    means2D = torch.zeros(pc[0].get_xyz.shape[0] + pc[1].get_xyz.shape[0], 2).cuda()
    means2D.requires_grad = True

    w = view.image_width  # // 4
    h = view.image_height  # // 4

    fx = 0.5 * w / np.tan(0.5 * view.FoVx)  # original focal length
    fy = 0.5 * h / np.tan(0.5 * view.FoVy)  # original focal length
    K = torch.tensor([
        [fx, 0, w/2, 0],
        [0, fy, h/2, 0],
        [0, 0, 1, 0],
    ], device="cuda").float()
    invK = torch.tensor([
        [1/fx, 0, -w/2/fx],
        [0, 1/fy, -h/2/fy],
        [0, 0, 1],
        [0, 0, 0],
    ], device="cuda").float()
    device = "cuda"
    
    wct = view.world_view_transform.cuda().float()
    full_wct = torch.eye(4, device="cuda")
    full_wct[:, :3] = wct @ K.T

    features_list = []
    for i in range(2):
        shs = pc[i].get_features

        cam_pos = view.camera_center.to(device)
        net_color = eval_sh2(pc[i].get_xyz, shs, cam_pos, pc[i].active_sh_degree)
        net_color = torch.nn.functional.softplus(net_color, beta=10)
        features_list.append(RGB2SH(net_color).reshape(-1, 1, 3))

    scales_list = []
    density_list = []
    for i in range(2):
        per_point_2d_filter_scale = torch.zeros(pc[i]._xyz.shape[0], device=pc[i]._xyz.device)
        if trace_rays.uses_density:
            scales, density = pc[i].get_scale_and_density_for_rendering(per_point_2d_filter_scale, scaling_modifier)
        else:
            scales, density = pc[i].get_scale_and_opacity_for_rendering(per_point_2d_filter_scale, scaling_modifier)
        scales_list.append(scales)
        density_list.append(density)
    tmin = min(pc[0].tmin, pc[1].tmin) if tmin is None else tmin

    ######################################### Deformation
    means3D_final_list = []
    scales_final_list = []
    rotations_final_list = []
    opacity_final_list = []
    shs_final_list = []
    for i in range(2):
        view_time = view.time if i == 0 else 1.0 - view.time
        time_tensor = torch.tensor(view_time).to("cuda").repeat(pc[i].get_xyz.shape[0],1)
            
        if stage[:4] == "fine":
            means3D_final_i, scales_final_i, rotations_final_i, opacity_final_i, shs_final_i = deformation[i].deform(
                                                                        pc[i].get_xyz, pc[i]._scaling, 
                                                                        pc[i]._rotation, density_list[i], features_list[i],
                                                                        time_tensor, stage, now_iteration)
        else:
            means3D_final_i, scales_final_i, rotations_final_i, opacity_final_i, shs_final_i = \
                pc[i].get_xyz, scales_list[i], pc[i].get_rotation, density_list[i], features_list[i]
            
        means3D_final_list.append(means3D_final_i)
        scales_final_list.append(scales_final_i)
        rotations_final_list.append(rotations_final_i)
        opacity_final_list.append(opacity_final_i)
        shs_final_list.append(shs_final_i)
    
    means3D_final = torch.cat(means3D_final_list, 0)
    scales_final = torch.cat(scales_final_list, 0)
    scales_final = pc[0].scaling_activation(scales_final)
    rotations_final = torch.cat(rotations_final_list, 0)
    rotations_final = pc[0].rotation_activation(rotations_final)
    opacity_final = torch.cat(opacity_final_list, 0)
    shs_final = torch.cat(shs_final_list, 0)

    average_scale = scales_final.detach().cpu().numpy().flatten() #.mean(0).mean()
    from scipy.stats import trim_mean
    print(trim_mean(average_scale, 0.3))
    exit(0)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.hist(average_scale, bins=100, color='skyblue', alpha=0.7, edgecolor='black')
    plt.xlabel('数值')
    plt.ylabel('频数')
    plt.title('数据分布直方图')
    plt.grid(True, alpha=0.3)
    plt.savefig('output/figure.png')
    plt.show()
    exit(0)
    

    out, extras = trace_rays(
        means3D_final,
        scales_final,
        rotations_final,
        opacity_final,
        shs_final,
        rays_o,
        rays_d,
        tmin,
        tmax,
        100,
        means2D,
        full_wct.reshape(1, 4, 4),
        max_iters=MAX_ITERS,
        return_extras=True,
    )

    torch.cuda.synchronize()
    radii = torch.ones_like(means2D[..., 0])

    rendered_image = out[:, :3].T.reshape(3, view.image_height, view.image_width)
    num_pixels = (extras['touch_count'] // 2)

    # aspect_ratio = scales.max(dim=-1).values / scales.min(dim=-1).values
    side_length = (num_pixels).float().sqrt() #/ aspect_ratio # mul by 2 to get to rect, then sqrt
    radii = side_length / 2 * np.sqrt(2) * 2.5 * 5

    return {
        "render": rendered_image,
        "viewspace_points": means2D,
        "visibility_filter": num_pixels >= 4,  # 每个高斯碰到的像素个数//2大于等于4时，则为可见
        "touch_count": extras['touch_count'],
        "radii": radii, # match gaussian radius
        "iters": extras["iters"].reshape(view.image_height, view.image_width),
        "opacity": out[:, 3].reshape(-1, 1),
        "distortion_loss": out[:, 4].reshape(-1, 1),
    }, average_scale
