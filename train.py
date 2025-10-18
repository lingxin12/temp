#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
from gaussian_renderer.ever import splinerender
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, get_diff_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from icecream import ic
import random
import math
import cv2
import numpy as np

import readline
import ipdb
import torchvision
from scene.deformation import DeformationModel
from datetime import datetime
from pytorch_msssim import ms_ssim
from lpipsPyTorch import lpips
import json
import itertools
import copy

renderFunc = splinerender
# renderFunc = render
from scene.dataset_readers import ProjectionType

PREVIEW_RES_FACTOR = 1

###############################################################
def get_all_camera_pose(scene):
    #############################################
    # train_cameras = scene.getTrainCameras()
    # poses_list = []
    # fovs_list = []
    # pp_list = []
    # for id, cam in enumerate(train_cameras):
    #     pose = torch.from_numpy(cam.R).float()
    #     pose = torch.cat([pose, torch.from_numpy(cam.T).float().unsqueeze(-1)], -1)
    #     pose = torch.cat([pose, torch.tensor([[0,0,0,1]]).float()], 0)
    #     poses_list.append(pose)
    #     w, h = cam.image_width, cam.image_height
    #     fovs_list.append(torch.tensor([cam.FoVx, cam.FoVy]))
    #     pp_list.append(torch.tensor([w/2, h/2]))
    # poses = torch.stack(poses_list, 0).cuda()
    # fovs = torch.stack(fovs_list, 0).cuda()
    # pp = torch.stack(pp_list, 0).cuda()
    # torch.save((poses, fovs, pp), "datasets/hypernerf/vrig/broom2/ex_in_mat.pth")
    # exit(0)
    #############################################
    pass

def save_image_and_test(iteration, use_deformation, deformation_model, camera_inds, gaussians, pipe, bg, opt, stage, now_iteration):
    print("\n\n[ITER {}] Saving Images".format(iteration))
    # stage = "fine" if use_deformation else "coarse"
    if not os.path.exists(os.path.join(scene.model_path, f"{stage}_{iteration}_train")):
        os.makedirs(os.path.join(scene.model_path, f"{stage}_{iteration}_train"))
    if not os.path.exists(os.path.join(scene.model_path, f"{stage}_{iteration}_test")):
        os.makedirs(os.path.join(scene.model_path, f"{stage}_{iteration}_test"))
    
    psnrs = 0.0
    ssims = 0.0
    ms_ssims = 0.0
    lpipss = 0.0
    # for cam in tqdm(scene.getTrainCameras(), desc="saving train images"):
    #     set_glo_vector(cam, scene.gaussians, camera_inds)
    #     render_image = renderFunc(cam, gaussians, pipe, bg, random=not opt.center_pixel, deformation=deformation_model, stage=stage, now_iteration=now_iteration)["render"].clip(0,1)
    #     gt_image = cam.original_image.to("cuda").clip(0,1)
    #     diff_image = get_diff_image(render_image, gt_image)
    #     temp_psnr = psnr(render_image, gt_image).mean().double()
    #     if cam.uid not in range(0, len(scene.getTrainCameras()), 20):
    #         continue
    #     torchvision.utils.save_image(torch.cat([render_image, gt_image, diff_image], dim=2), os.path.join(scene.model_path, f"{stage}_{iteration}_train/{cam.uid}_PSNR{temp_psnr:.2f}.png"))
    psnrs_list = []
    for cam in tqdm(scene.getTestCameras(), desc="saving test images"):
        # set_glo_vector(cam, scene.gaussians, camera_inds)
        render_image = renderFunc(cam, gaussians, pipe, bg, random=False, deformation=deformation_model, stage=stage, now_iteration=now_iteration)["render"].clip(0,1)
        gt_image = cam.original_image.to("cuda").clip(0,1)
        diff_image = get_diff_image(render_image, gt_image)
        temp_psnr = psnr(render_image, gt_image).mean().double()
        psnrs += temp_psnr
        ssims += ssim(render_image, gt_image).mean()
        ms_ssims += ms_ssim(render_image.unsqueeze(0), gt_image.unsqueeze(0),data_range=1,size_average=True).mean()
        # lpipss += lpips(render_image, gt_image).mean()
        psnrs_list.append(temp_psnr.item())
        if cam.uid not in range(0, len(scene.getTestCameras()), 20):
            continue
        torchvision.utils.save_image(torch.cat([render_image, gt_image, diff_image], dim=2), os.path.join(scene.model_path, f"{stage}_{iteration}_test/{cam.uid}_PSNR{temp_psnr:.2f}.png"))
    
    psnrs /= len(scene.getTestCameras())
    ssims /= len(scene.getTestCameras())
    ms_ssims /= len(scene.getTestCameras())
    lpipss /= len(scene.getTestCameras())
    with open(os.path.join(scene.model_path, f"{stage}_{iteration}_test/metrics.txt"), "w") as f:
        f.write(f"PSNR: {psnrs}\n")
        f.write(f"SSIM: {ssims}\n")
        f.write(f"MS-SSIM: {ms_ssims}\n")
        f.write(f"LPIPS: {lpipss}\n")

def project(xyz, wct):
    p_hom = torch.cat([xyz, torch.ones((xyz.shape[0], 1), device="cuda")], dim=1)
    p_view = (p_hom @ wct)
    pix2d = p_view[:, :2] / p_view[:, 2:3]
    return pix2d, p_view[:, 2]

def inv_project(xy, dist, inv_wvt):
    N = xy.shape[0]
    pad = torch.ones((N, 1), device="cuda")
    p_hom = torch.cat([xy * dist.reshape(-1, 1), dist.reshape(-1, 1), pad], dim=1) @ inv_wvt
    return p_hom[:, :3]

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def set_glo_vector(viewpoint_cam, gaussians, camera_inds):
    camera_ind = camera_inds[viewpoint_cam.uid]
    viewpoint_cam.glo_vector = torch.cat(
        [gaussians.glo[camera_ind], torch.tensor([
                math.log(
                viewpoint_cam.iso * viewpoint_cam.exposure / 1000),
            ], device=gaussians.glo.device)
         ]
    )

def training_coarse(scene, gaussians, tb_writer,
             dataset, opt, pipe, hyper,
             testing_iterations, saving_iterations,
             checkpoint_iterations, checkpoint, deform_checkpoint, debug_from,
             use_deformation=False, stage="", deformation_model=None):

    first_iter = 0

    ################## Deformation
    if deformation_model is None:
        deformation_model = DeformationModel(hyper, opt, scene)
    if not use_deformation:
        deformation_model.deform = None   ################## Open deformation or not
    #################################################################################################

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    if deform_checkpoint and deformation_model.deform is not None:
        deformation_params = torch.load(deform_checkpoint)
        deformation_model.deform.load_state_dict(deformation_params)


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    ema_loss_for_log = 0.0
    train_iterations = opt.fine_iterations_1 if stage == "fine1" else \
        opt.fine_iterations_2 if stage == "fine2" else \
        opt.fine_iterations_3 if stage == "fine3" else \
        opt.coarse_iterations
    progress_bar = tqdm(range(first_iter, train_iterations), desc="Training progress")
    first_iter += 1
    camera_inds = {view.uid: i for i, view in enumerate(viewpoint_stack)}
    gaussians.initialize_glo(len(viewpoint_stack), dataset.glo_latent_dim) ## useless
    train_cameras = scene.getTrainCameras()

    clone_grad_threshold = opt.clone_grad_threshold
    densify_grad_threshold = opt.densify_grad_threshold

    gaussians.training_setup(opt)

    # #############################################################################
    # ################################# initialize dx
    # if stage == "fine2":# or stage == "fine3":
    if stage == "fine1":
        cams = scene.getTrainCameras()
        init_iteration = 3000
        init_pb = tqdm(range(init_iteration), desc="Init progress")
        for iter in range(init_iteration):
            view = cams[iter%len(cams)]
            time = torch.tensor(view.time).to("cuda").repeat(gaussians.get_xyz.shape[0],1)
            dx_list = deformation_model.deform.deformation_net.get_dx_for_init(gaussians.get_xyz, time)
            # loss = (dx_list[1]**2).mean() if stage == "fine2" else (dx_list[2]**2).mean()
            loss = (dx_list[1]**2).mean() + (dx_list[2]**2).mean()
            loss.backward()

            with torch.no_grad():
                if iter%10 == 0:
                    init_pb.set_postfix({"Loss": f"{loss:.5f}"})
                    init_pb.update(10)
                if iter == init_iteration:
                    init_pb.close()
                deformation_model.optimizer.step()
                deformation_model.optimizer.zero_grad(set_to_none = True)
    
    if stage == "fine1":
        first_iter = 1
        train_iterations = 10000
    if stage == "fine2":
        first_iter = 10001
        train_iterations = 20000
    if stage == "fine3":
        first_iter = 20001
        train_iterations = 30000
    for iteration in range(first_iter, train_iterations+1):

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        deformation_model.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % opt.sh_up_interval == 0:
            gaussians.oneupSHdegree()

        # Pick random camera and render
        images = []
        gt_images = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        distortion_loss_list = []
        for _ in range(opt.batch_size):
            if not viewpoint_stack:
                viewpoint_stack = train_cameras.copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            # set_glo_vector(viewpoint_cam, gaussians, camera_inds)
            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True

            bg = torch.rand((3), device="cuda") if opt.random_background else background  #####

            render_pkg = renderFunc(viewpoint_cam, gaussians, pipe, bg, random=not opt.center_pixel, deformation=deformation_model, stage=stage, now_iteration=iteration)  # self.center_pixel = False
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            if viewpoint_cam.alpha_mask is not None:
                alpha_mask = viewpoint_cam.alpha_mask.cuda()
                image *= alpha_mask
        
            gt_image = viewpoint_cam.original_image.cuda()

            ########################### batchsize content
            images.append(image)
            gt_images.append(gt_image)
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)
            distortion_loss_list.append(render_pkg['distortion_loss'])

        image = torch.cat(images, dim=-1)
        gt_image = torch.cat(gt_images, dim=-1)
        radii = torch.cat(radii_list,0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        #########################################################

        # Loss
        Ll1 = l1_loss(image, gt_image)

        scaling = gaussians.get_scaling
        anisotropic_loss = ((1-gaussians.get_opacity.detach()).reshape(-1)*((scaling.max(dim=-1).values - scaling.min(dim=-1).values)))[visibility_filter].mean()
        size_loss = (scaling.sqrt()).mean()
        lambda_dssim = opt.lambda_dssim
        
        loss = 0.0
        loss += (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim(image, gt_image)).clip(min=0, max=1)
        loss += opt.lambda_distortion * sum([d_l.mean() for d_l in distortion_loss_list])# if iteration > 2000 else 0
        loss += opt.lambda_anisotropic * anisotropic_loss 
        if torch.isnan(loss).any():
            print("nan")
            continue

        ######################################################## Deformation Loss
        if use_deformation:
            tv_loss = deformation_model.compute_regulation(hyper)
            loss += tv_loss

        loss.backward()

        ################ batchsize content: set viewspace_point_tensor_grad
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f} Num Prim: {gaussians.get_xyz.shape[0]} I: {render_pkg['iters'].float().mean()}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, gaussians, renderFunc, (pipe, background), camera_inds, deformation_model, stage=stage, now_iteration=iteration)
            # exit(0)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            if iteration in checkpoint_iterations:
                save_image_and_test(iteration, use_deformation, deformation_model, camera_inds, gaussians, pipe, bg, opt, stage=stage, now_iteration=iteration)

            # Densification
            if iteration > opt.densify_from_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)  ######### modify

            if iteration < opt.densify_until_iter and iteration > opt.densify_from_iter:

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 \
                    and iteration % opt.opacity_reset_interval > opt.densification_interval:#  \
                    #and gaussians.get_xyz.shape[0]<600000:

                    gaussians.densify_and_prune(densify_grad_threshold, opt.min_opacity, scene.cameras_extent, 1000, clone_grad_threshold, opt.min_split_opacity)
                    torch.cuda.empty_cache()
                
                #################### reset opacity
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    print(f"Reset opacity")
                    gaussians.reset_opacity(0.005)
                    torch.cuda.empty_cache()
            else:
                if (   #################### 考虑是否删除
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                    and iteration % opt.opacity_reset_interval > opt.densification_interval
                ):
                    print(f"Prune with min_opacity in iteration {iteration}")
                    gaussians.update_death_mark()
                    prune_mask = ((gaussians.get_minor_axis_opacity < opt.min_opacity).squeeze())
                    gaussians.prune_points(prune_mask)
                    # print(f"Pruned {prune_mask.sum()} primitives. Mean Opacity: {gaussians.get_opacity.mean()}")
                    torch.cuda.empty_cache()


            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

                deformation_model.optimizer.step()
                deformation_model.optimizer.zero_grad(set_to_none = True)
                

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                if not os.path.exists(scene.model_path+"/chkpnts"):
                    os.makedirs(scene.model_path+"/chkpnts")
                if use_deformation:
                    torch.save((gaussians.capture(), iteration), scene.model_path + f"/chkpnts/{stage}_chkpnt" + str(iteration) + ".pth")
                    torch.save(deformation_model.deform.state_dict(), scene.model_path + f"/chkpnts/{stage}_chkpnt_deform" + str(iteration) + ".pth")
                else:
                    torch.save((gaussians.capture(), 0), scene.model_path + "/chkpnts/coarse_chkpnt" + str(iteration) + ".pth")



def training_fine(scene, gaussians, tb_writer,
             dataset, opt, pipe, hyper,
             testing_iterations, saving_iterations,
             checkpoint_iterations, checkpoint, deform_checkpoint, debug_from,
             use_deformation=False, stage="", deformation_model=None):

    first_iter = 0 if stage=="fine1" else opt.fine_iterations_1 if stage=="fine2" else opt.fine_iterations_1+opt.fine_iterations_2

    if checkpoint:
        if "coarse" in checkpoint:
            (model_params_0, first_iter) = torch.load(checkpoint)
            gaussians[0].restore(model_params_0, opt)
            gaussians[1] = copy.deepcopy(gaussians[0])
        else:
            (model_params_0, model_params_1, first_iter) = torch.load(checkpoint)
            gaussians[0].restore(model_params_0, opt)
            gaussians[1].restore(model_params_1, opt)
    if deform_checkpoint:
        (deformation_params_0, deformation_params_1) = torch.load(deform_checkpoint)
        deformation_model[0].deform.load_state_dict(deformation_params_0)
        deformation_model[1].deform.load_state_dict(deformation_params_1)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    ema_loss_for_log = 0.0
    train_iterations = opt.fine_iterations_1 if stage == "fine1" else \
        opt.fine_iterations_1+opt.fine_iterations_2 if stage == "fine2" else \
        opt.fine_iterations_1+opt.fine_iterations_2+opt.fine_iterations_3 if stage == "fine3" else \
        opt.coarse_iterations
    progress_bar = tqdm(range(first_iter, train_iterations), desc="Training progress")
    first_iter += 1
    camera_inds = {view.uid: i for i, view in enumerate(viewpoint_stack)}
    train_cameras = scene.getTrainCameras()

    clone_grad_threshold = opt.clone_grad_threshold
    densify_grad_threshold = opt.densify_grad_threshold

    gaussians[0].training_setup(opt)
    gaussians[1].training_setup(opt)

    all_cams = scene.getTrainCameras()
    bg = torch.rand((3), device="cuda") if opt.random_background else background
    all_average_scale = []
    for cam in all_cams:
        render_pkg, average_scale = renderFunc(cam, gaussians, pipe, bg, random=not opt.center_pixel, deformation=deformation_model, stage=stage, now_iteration=20000)
        all_average_scale.append((cam.time, average_scale.item()))
        exit(0)
    print(all_average_scale)
    exit(0)


    # #############################################################################
    # ################################# initialize dx
    if stage == "fine2":# or stage == "fine3":
    # if stage == "fine1":
        cams = scene.getTrainCameras()
        init_iteration = 3000
        init_pb = tqdm(range(init_iteration), desc="Init progress")
        for iter in range(init_iteration):
            view = cams[iter%len(cams)]
            time_0 = torch.tensor(view.time).to("cuda").repeat(gaussians[0].get_xyz.shape[0],1)
            time_1 = torch.tensor(1.0 - view.time).to("cuda").repeat(gaussians[1].get_xyz.shape[0],1)
            dx_list_0 = deformation_model[0].deform.deformation_net.get_dx_for_init(gaussians[0].get_xyz, time_0)
            dx_list_1 = deformation_model[1].deform.deformation_net.get_dx_for_init(gaussians[1].get_xyz, time_1)
            loss = (dx_list_0[1]**2).mean() + (dx_list_0[2]**2).mean() + (dx_list_1[1]**2).mean() + (dx_list_1[2]**2).mean()
            loss.backward()

            with torch.no_grad():
                if iter%10 == 0:
                    init_pb.set_postfix({"Loss": f"{loss:.5f}"})
                    init_pb.update(10)
                if iter == init_iteration:
                    init_pb.close()
                deformation_model[0].optimizer.step()
                deformation_model[0].optimizer.zero_grad(set_to_none = True)
                deformation_model[1].optimizer.step()
                deformation_model[1].optimizer.zero_grad(set_to_none = True)

    for iteration in range(first_iter, train_iterations+1):

        iter_start.record()

        gaussians[0].update_learning_rate(iteration)
        gaussians[1].update_learning_rate(iteration)
        deformation_model[0].update_learning_rate(iteration)
        deformation_model[1].update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % opt.sh_up_interval == 0:
            gaussians[0].oneupSHdegree()
            gaussians[1].oneupSHdegree()

        # Pick random camera and render
        images = []
        gt_images = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        distortion_loss_list = []
        for _ in range(opt.batch_size):
            if not viewpoint_stack:
                viewpoint_stack = train_cameras.copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            # set_glo_vector(viewpoint_cam, gaussians, camera_inds)
            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True

            bg = torch.rand((3), device="cuda") if opt.random_background else background  #####

            render_pkg = renderFunc(viewpoint_cam, gaussians, pipe, bg, random=not opt.center_pixel, deformation=deformation_model, stage=stage, now_iteration=iteration)  # self.center_pixel = False
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            if viewpoint_cam.alpha_mask is not None:
                alpha_mask = viewpoint_cam.alpha_mask.cuda()
                image *= alpha_mask
        
            gt_image = viewpoint_cam.original_image.cuda()

            ########################### batchsize content
            images.append(image)
            gt_images.append(gt_image)
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)
            distortion_loss_list.append(render_pkg['distortion_loss'])
        
        gs_segment_list = list(itertools.accumulate([0, gaussians[0].get_xyz.shape[0], gaussians[1].get_xyz.shape[0]]))
        image = torch.cat(images, dim=-1)
        gt_image = torch.cat(gt_images, dim=-1)
        radii = torch.cat(radii_list,0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        visibility_filter_seg_list = [visibility_filter[:gaussians[0].get_xyz.shape[0]], visibility_filter[gaussians[0].get_xyz.shape[0]:]]
        radii_seg_list = [radii[:gaussians[0].get_xyz.shape[0]], radii[gaussians[0].get_xyz.shape[0]:]]
        #########################################################

        # Loss
        Ll1 = l1_loss(image, gt_image)

        scaling_0 = gaussians[0].get_scaling
        scaling_1 = gaussians[1].get_scaling
        anisotropic_loss_0 = ((1-gaussians[0].get_opacity.detach()).reshape(-1)*((scaling_0.max(dim=-1).values - scaling_0.min(dim=-1).values)))[visibility_filter_seg_list[0]].mean()
        anisotropic_loss_1 = ((1-gaussians[1].get_opacity.detach()).reshape(-1)*((scaling_1.max(dim=-1).values - scaling_1.min(dim=-1).values)))[visibility_filter_seg_list[1]].mean()
        lambda_dssim = opt.lambda_dssim
        
        loss = 0.0
        loss += (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim(image, gt_image)).clip(min=0, max=1)
        loss += opt.lambda_distortion * distortion_loss_list[0].mean() + opt.lambda_distortion * distortion_loss_list[1].mean()
        loss += opt.lambda_anisotropic * anisotropic_loss_0 + opt.lambda_anisotropic * anisotropic_loss_1

        ######################################################## Deformation Loss
        if use_deformation:
            loss += deformation_model[0].compute_regulation(hyper)
            loss += deformation_model[1].compute_regulation(hyper)

        if torch.isnan(loss).any():
            print("nan")
            continue

        loss.backward()

        ################ batchsize content: set viewspace_point_tensor_grad
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        viewspace_point_tensor_grad_list = [viewspace_point_tensor_grad[:gaussians[0].get_xyz.shape[0]], viewspace_point_tensor_grad[gaussians[0].get_xyz.shape[0]:]]

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f} Num Prim: {[gaussians[0].get_xyz.shape[0], gaussians[1].get_xyz.shape[0]]}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, gaussians, renderFunc, (pipe, background), camera_inds, deformation_model, stage=stage, now_iteration=iteration)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            if iteration in checkpoint_iterations:
                save_image_and_test(iteration, use_deformation, deformation_model, camera_inds, gaussians, pipe, bg, opt, stage=stage, now_iteration=iteration)

            # Densification
            if iteration > opt.densify_from_iter:
                gaussians[0].max_radii2D[visibility_filter_seg_list[0]] = torch.max(gaussians[0].max_radii2D[visibility_filter_seg_list[0]], radii_seg_list[0][visibility_filter_seg_list[0]])
                gaussians[1].max_radii2D[visibility_filter_seg_list[1]] = torch.max(gaussians[1].max_radii2D[visibility_filter_seg_list[1]], radii_seg_list[1][visibility_filter_seg_list[1]])
                gaussians[0].add_densification_stats(viewspace_point_tensor_grad_list[0], visibility_filter_seg_list[0])
                gaussians[1].add_densification_stats(viewspace_point_tensor_grad_list[1], visibility_filter_seg_list[1])

            if iteration < opt.densify_until_iter and iteration > opt.densify_from_iter:

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 \
                    and iteration % opt.opacity_reset_interval > opt.densification_interval:

                    gaussians[0].densify_and_prune(densify_grad_threshold, opt.min_opacity, scene.cameras_extent, 1000, clone_grad_threshold, opt.min_split_opacity, gaussians[0].get_xyz.shape[0]+gaussians[1].get_xyz.shape[0])
                    gaussians[1].densify_and_prune(densify_grad_threshold, opt.min_opacity, scene.cameras_extent, 1000, clone_grad_threshold, opt.min_split_opacity, gaussians[0].get_xyz.shape[0]+gaussians[1].get_xyz.shape[0])
                    torch.cuda.empty_cache()
                
                #################### reset opacity
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    print(f"Reset opacity")
                    gaussians[0].reset_opacity(0.005)
                    gaussians[1].reset_opacity(0.005)
                    torch.cuda.empty_cache()
            else:
                if (   #################### 考虑是否删除
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                    and iteration % opt.opacity_reset_interval > opt.densification_interval
                ):
                    print(f"Prune with min_opacity in iteration {iteration}")
                    gaussians[0].update_death_mark()
                    gaussians[1].update_death_mark()
                    prune_mask_0 = ((gaussians[0].get_minor_axis_opacity < opt.min_opacity).squeeze())
                    prune_mask_1 = ((gaussians[1].get_minor_axis_opacity < opt.min_opacity).squeeze())
                    gaussians[0].prune_points(prune_mask_0)
                    gaussians[1].prune_points(prune_mask_1)
                    # print(f"Pruned {prune_mask.sum()} primitives. Mean Opacity: {gaussians.get_opacity.mean()}")
                    torch.cuda.empty_cache()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians[0].optimizer.step()
                gaussians[0].optimizer.zero_grad(set_to_none = True)
                gaussians[1].optimizer.step()
                gaussians[1].optimizer.zero_grad(set_to_none = True)

                deformation_model[0].optimizer.step()
                deformation_model[0].optimizer.zero_grad(set_to_none = True)
                deformation_model[1].optimizer.step()
                deformation_model[1].optimizer.zero_grad(set_to_none = True)
                

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                if not os.path.exists(scene.model_path+"/chkpnts"):
                    os.makedirs(scene.model_path+"/chkpnts")
                if stage[:4] == "fine":
                    torch.save((gaussians[0].capture(), gaussians[1].capture(), iteration), scene.model_path + f"/chkpnts/{stage}_chkpnt" + str(iteration) + ".pth")
                    torch.save((deformation_model[0].deform.state_dict(), deformation_model[1].deform.state_dict()), scene.model_path + f"/chkpnts/{stage}_chkpnt_deform" + str(iteration) + ".pth")
                elif stage[:4] == "coar":
                    torch.save((gaussians.capture(), 0), scene.model_path + "/chkpnts/coarse_chkpnt" + str(iteration) + ".pth")




def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path+"/tbs/"+datetime.now().strftime("%m-%d---%H-%M-%S"))
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, gaussians, renderFunc, renderArgs, camera_inds, deformation_model, stage, now_iteration):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' :
                               [scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                                for idx in range(5, 30, 5)]
                               })

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    # set_glo_vector(viewpoint, scene.gaussians, camera_inds)
                    image = torch.clamp(renderFunc(viewpoint, gaussians, *renderArgs, random=False, deformation=deformation_model, stage=stage, now_iteration=now_iteration)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                with open(os.path.join(scene.model_path, f"psnr_record.txt"), "a") as f:
                    f.write(f"{iteration} : {psnr_test}\n")

        if tb_writer:
            if stage[:4] == "fine":
                tb_writer.add_scalar('total_points', gaussians[0].get_xyz.shape[0]+gaussians[1].get_xyz.shape[0], iteration)
            else:
                tb_writer.add_scalar('total_points', gaussians.get_xyz.shape, iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[1, 3000, 7000, 10000, 14000, 20000, 30000])
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[3000, 7000, 10000, 14000, 20000, 30000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[i for i in range(3000, 30001, 1000)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[3000, 7000, 10000, 14000, 20000, 30000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--start_deform_checkpoint", type=str, default=None)
    parser.add_argument("--skip_coarse", action="store_true")
    parser.add_argument("--configs", type=str, default = "")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    # args.checkpoint_iterations.append(args.iterations)

    ############################ modify model path
    args.model_path = args.model_path + "/" + datetime.now().strftime("%m-%d---%H-%M-%S")

    ############################ add params
    print("Load parameters and save configs")
    import mmengine
    config = mmengine.Config.fromfile(args.configs)
    ParamsGroupName = ["OptimizationParams", "ModelHiddenParams", "ModelParams", "PipelineParams"]
    for param in ParamsGroupName:
        if param in config.keys():
            for key, value in config[param].items():
                if hasattr(args, key):
                    setattr(args, key, value)
    all_params = {"ModelParams": lp.extract(args).__dict__, "OptimizationParams": op.extract(args).__dict__, "PipelineParams": pp.extract(args).__dict__, "ModelHiddenParams": hp.extract(args).__dict__, "args": args.__dict__}
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    with open(os.path.join(args.model_path, "params.json"), "w") as f:
        json.dump(all_params, f, indent=4)
    ##############################
    
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    lp, op, pp, hp = lp.extract(args), op.extract(args), pp.extract(args), hp.extract(args)
    gaussians = GaussianModel(lp.sh_degree, lp.use_neural_network, lp.max_opacity, lp.tmin, op.max_primitives)
    scene = Scene(lp, gaussians)
    
    tb_writer = prepare_output_and_logger(lp)
    if not args.skip_coarse:
        print("----------------------------run coarse stage----------------------------")
        training_coarse(scene, gaussians, None,
                lp, op, pp, hp,
                args.test_iterations, args.save_iterations, args.checkpoint_iterations,
                args.start_checkpoint, args.start_deform_checkpoint, args.debug_from,
                use_deformation=False, stage="coarse")
    
    gaussians_list = [gaussians]
    gaussians_list.append(copy.deepcopy(gaussians))
    deformation_model_list = [DeformationModel(hp, op, scene), DeformationModel(hp, op, scene)]
    
    # print("----------------------------run fine stage 1----------------------------")
    # training_fine(scene, gaussians_list, tb_writer,
    #          lp, op, pp, hp,
    #          args.test_iterations, args.save_iterations, args.checkpoint_iterations,
    #          args.start_checkpoint, args.start_deform_checkpoint, args.debug_from,
    #          use_deformation=True, stage="fine1",
    #          deformation_model=deformation_model_list)
    
    print("----------------------------run fine stage 2----------------------------")
    training_fine(scene, gaussians_list, tb_writer,
             lp, op, pp, hp,
             args.test_iterations, args.save_iterations, args.checkpoint_iterations,
             args.start_checkpoint, args.start_deform_checkpoint, args.debug_from,
             use_deformation=True, stage="fine2",
             deformation_model=deformation_model_list)

    # print("----------------------------run fine stage 3----------------------------")
    # training(scene, gaussians, tb_writer,
    #          lp, op, pp, hp,
    #          args.test_iterations, args.save_iterations, args.checkpoint_iterations,
    #          args.start_checkpoint, args.start_deform_checkpoint, args.debug_from,
    #          use_deformation=True, stage="fine3",
    #          deformation_model=deformation_model)

    # print("----------------------------run fine stage 4----------------------------")
    # training(scene, gaussians, tb_writer,
    #          lp, op, pp, hp,
    #          args.test_iterations, args.save_iterations, args.checkpoint_iterations,
    #          args.start_checkpoint, args.start_deform_checkpoint, args.debug_from,
    #          use_deformation=True, stage="fine4")
    


