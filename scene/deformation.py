import functools
import math
import os
import time
from tkinter import W

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from utils.graphics_utils import apply_rotation, batch_quaternion_multiply
from scene.hexplane import HexPlaneField
from scene.grid import DenseGrid
from utils.general_utils import get_expon_lr_func

# from scene.grid import HashHexPlane
class Deformation(nn.Module):
    def __init__(self, D=8, W=256, input_ch=27, input_ch_time=9, grid_pe=0, skips=[], args=None):
        super(Deformation, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_time = input_ch_time
        self.skips = skips
        self.grid_pe = grid_pe
        self.no_grid = args.no_grid
        self.multires = args.multires
        self.multires_num = len(self.multires)
        self.use_attention = True
        self.is_encode_value = False
        self.is_expand_atten_weight = True
        # self.front_attention = False

        self.grid = HexPlaneField(args.bounds, args.kplanes_config, args.multires, self.use_attention)
        # breakpoint()
        self.args = args
        # self.args.empty_voxel=True
        if self.args.empty_voxel:
            self.empty_voxel = DenseGrid(channels=1, world_size=[64,64,64])
        if self.args.static_mlp:
            self.static_mlp = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
        
        self.ratio=0
        self.create_net()
    @property
    def get_aabb(self):
        return self.grid.get_aabb
    def set_aabb(self, xyz_max, xyz_min):
        print("Deformation Net Set aabb",xyz_max, xyz_min)
        self.grid.set_aabb(xyz_max, xyz_min)
        if self.args.empty_voxel:
            self.empty_voxel.set_aabb(xyz_max, xyz_min)
    def create_net(self):
        grid_out_dim = self.grid.feat_dim
        
        self.feature_out_list = nn.ModuleList()
        for i in range(self.multires_num):
            # feature_out = [nn.Linear(grid_out_dim//3 ,self.W)]
            # feature_out = [nn.Linear(grid_out_dim*(3-i) ,self.W)]
            feature_out = [nn.Linear(grid_out_dim*3,self.W)]
            # feature_out = [nn.Linear(grid_out_dim//2*3,self.W)]
            for j in range(self.D-1):
                feature_out.append(nn.ReLU())
                feature_out.append(nn.Linear(self.W,self.W))
            feature_out = nn.Sequential(*feature_out)
            self.feature_out_list.append(feature_out)

        if self.use_attention:
            self.atten_encoder_list = nn.ModuleList()
            # for i in range(self.multires_num):
            #     # self.atten_encoder_list.append(nn.Sequential(nn.ReLU(), nn.Linear(self.W*(self.multires_num-1), self.W)))
            #     self.atten_encoder_list.append(nn.Sequential(nn.ReLU(), nn.Linear(self.W*self.multires_num, self.W)))
            self.atten_encoder_list.append(nn.Sequential(nn.ReLU(), nn.Linear(grid_out_dim*3, self.W)))
            self.atten_encoder_list.append(nn.Sequential(nn.ReLU(), nn.Linear(grid_out_dim*3, self.W)))

        if self.is_encode_value:
            self.value_encoder_list = nn.ModuleList()
            for i in range(self.multires_num):
                self.value_encoder_list.append(nn.Sequential(nn.Linear(self.W,self.W), nn.ReLU()))

        self.pos_deform_list = nn.ModuleList()
        self.scales_deform_list = nn.ModuleList()
        self.rotations_deform_list = nn.ModuleList()
        self.opacity_deform_list = nn.ModuleList()
        self.shs_deform_list = nn.ModuleList()
        for i in range(self.multires_num):
            if self.use_attention:
                self.pos_deform_list.append(nn.Sequential(nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3)))
                self.scales_deform_list.append(nn.Sequential(nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3)))
                self.rotations_deform_list.append(nn.Sequential(nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 4)))
                self.opacity_deform_list.append(nn.Sequential(nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1)))
                self.shs_deform_list.append(nn.Sequential(nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 16*3)))
            else:
                self.pos_deform_list.append(nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3)))
                self.scales_deform_list.append(nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3)))
                self.rotations_deform_list.append(nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 4)))
                self.opacity_deform_list.append(nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1)))
                self.shs_deform_list.append(nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 16*3)))

    def mutual_attention_process(self, hidden_list):
        # atten_hidden_list = []
        # atten_hidden_list.append(torch.relu(hidden_list[1])*(2.0 * torch.sigmoid(self.atten_encoder_list[0](torch.cat(hidden_list[:1]+hidden_list[2:], -1)))-1) )
        # return hidden_list

        atten_hidden_list = []
        total_feature = torch.cat(hidden_list, -1)
        total_atten = self.atten_encoder_list[0](total_feature)
        for i in range(len(hidden_list)):
            ############################################
            v = torch.relu(hidden_list[i])
            if self.is_encode_value:
                v = self.value_encoder_list[i](v)
            qk = self.atten_encoder_list[i](total_feature)
            qk = 2.0 * torch.sigmoid(qk) - 1.0 if self.is_expand_atten_weight else torch.sigmoid(qk)
            atten_hidden_list.append(v*qk)

            atten_hidden_list.append(torch.relu(hidden_list[i]) * (2.0 * torch.sigmoid(self.atten_encoder_list[i](total_feature)) - 1.0))

            ############################################
            # deform_feature = hidden_list[i][:self.W]
            # atten_feature = torch.cat([h_f[self.W:] for h_f in hidden_list], -1)
            # atten_hidden_list.append(torch.relu(deform_feature) * (2.0 * torch.sigmoid(atten_feature) - 1.0))

            ############################################
            atten_hidden_list.append(torch.relu(hidden_list[i]) * (2.0 * torch.sigmoid(total_atten) - 1.0))


        return atten_hidden_list

    def query_time(self, rays_pts_emb, scales_emb, rotations_emb, time_feature, time_emb, stage, now_iteration):
        
        if self.no_grid:
            h = torch.cat([rays_pts_emb[:,:3],time_emb[:,:1]],-1)
        else:

            grid_feature = self.grid(rays_pts_emb[:,:3], time_emb[:,:1])  # return a grid feature list

            if self.grid_pe > 1: # useless
                grid_feature = poc_fre(grid_feature,self.grid_pe)
        
        hidden_list = []
        if not self.use_attention:
            concat_grid_feature = torch.cat(grid_feature, -1)
            for i in range(self.multires_num):
                hidden_list.append(self.feature_out_list[i](concat_grid_feature))
                # hidden_list.append(self.feature_out_list[i](grid_feature[i]))
        else:
            # # hidden_list.append(torch.relu(self.feature_out_list[0](torch.cat(grid_feature, -1))))
            # # hidden_list.append(torch.relu(self.feature_out_list[1](torch.cat(grid_feature[1:], -1))))
            # # hidden_list.append(torch.relu(self.feature_out_list[2](grid_feature[-1])))

            # breakpoint()
            if stage == "fine1":
                hidden_list.append(torch.relu(self.feature_out_list[0](torch.cat(grid_feature, -1))))

            if stage == "fine2":
                # hidden_list.append(torch.relu(self.feature_out_list[0](torch.cat(grid_feature, -1))))
                # hidden_list.append(torch.relu(self.feature_out_list[1](torch.cat(grid_feature[1:], -1))))
                hidden_list.append(torch.relu(self.feature_out_list[0](torch.cat(grid_feature, -1))))
                hidden_list.append(torch.relu(self.feature_out_list[1](torch.cat(grid_feature, -1))))

                # v = self.feature_out_list[1](torch.cat(grid_feature[1:], -1))
                # qk = self.atten_encoder_list[0](grid_feature[0].detach())
                # hidden_list.append(torch.relu(v) * (2.0 * torch.sigmoid(qk) - 1.0))
            
            # if stage == "fine3":
            #     # hidden_list.append(torch.relu(self.feature_out_list[0](torch.cat(grid_feature, -1))))
            #     # hidden_list.append(torch.relu(self.feature_out_list[1](torch.cat(grid_feature[1:], -1))))
            #     # hidden_list.append(torch.relu(self.feature_out_list[2](grid_feature[-1])))
            #     hidden_list.append(torch.relu(self.feature_out_list[0](torch.cat(grid_feature, -1))))
            #     hidden_list.append(torch.relu(self.feature_out_list[1](torch.cat(grid_feature, -1))))
            #     hidden_list.append(torch.relu(self.feature_out_list[2](torch.cat(grid_feature, -1))))

            #     # v = self.feature_out_list[1](torch.cat(grid_feature[1:], -1).detach())
            #     # qk = self.atten_encoder_list[0](grid_feature[0].detach())
            #     # hidden_list.append(torch.relu(v) * (2.0 * torch.sigmoid(qk) - 1.0))

            # if stage == "fine1":
            #     hidden_list.append(torch.relu(self.feature_out_list[0](torch.cat(grid_feature, -1))))
            # if stage == "fine2" or stage == "fine3":
            #     if now_iteration % 400 >= 0 and now_iteration % 400 < 200:
            #         hidden_list.append(torch.relu(self.feature_out_list[0](torch.cat(grid_feature, -1))))
            #         hidden_list.append(torch.relu(self.feature_out_list[1](torch.cat(grid_feature, -1).detach()).detach()))
            #         # v_0 = torch.relu(self.feature_out_list[0](torch.cat(grid_feature, -1)))
            #         # v_1 = torch.relu(self.feature_out_list[1](torch.cat(grid_feature, -1).detach()).detach())
            #         # qk_0 = self.atten_encoder_list[0](torch.cat(grid_feature, -1))
            #         # qk_1 = self.atten_encoder_list[1](torch.cat(grid_feature, -1).detach()).detach()
            #         # hidden_list.append(v_0 * (2.0 * torch.sigmoid(qk_0) - 1.0))
            #         # hidden_list.append(v_1.detach() * (2.0 * torch.sigmoid(qk_1.detach()) - 1.0))

            #     else:
            #         hidden_list.append(torch.relu(self.feature_out_list[0](torch.cat(grid_feature, -1).detach()).detach()))
            #         hidden_list.append(torch.relu(self.feature_out_list[1](torch.cat(grid_feature, -1))))
            #         # v_0 = torch.relu(self.feature_out_list[0](torch.cat(grid_feature, -1).detach()).detach())
            #         # v_1 = torch.relu(self.feature_out_list[1](torch.cat(grid_feature, -1)))
            #         # qk_0 = self.atten_encoder_list[0](torch.cat(grid_feature, -1).detach()).detach()
            #         # qk_1 = self.atten_encoder_list[1](torch.cat(grid_feature, -1))
            #         # hidden_list.append(v_0.detach() * (2.0 * torch.sigmoid(qk_0.detach()) - 1.0))
            #         # hidden_list.append(v_1 * (2.0 * torch.sigmoid(qk_1) - 1.0))
                
            # if stage == "fine3":
            #     if now_iteration % 600 >= 0 and now_iteration % 600 < 200:
            #         hidden_list.append(torch.relu(self.feature_out_list[0](torch.cat(grid_feature, -1))))
            #         hidden_list.append(torch.relu(self.feature_out_list[1](torch.cat(grid_feature, -1).detach()).detach()))
            #         hidden_list.append(torch.relu(self.feature_out_list[2](torch.cat(grid_feature, -1).detach()).detach()))
            #     elif now_iteration % 600 >= 200 and now_iteration % 600 < 400:
            #         hidden_list.append(torch.relu(self.feature_out_list[0](torch.cat(grid_feature, -1).detach()).detach()))
            #         hidden_list.append(torch.relu(self.feature_out_list[1](torch.cat(grid_feature, -1))))
            #         hidden_list.append(torch.relu(self.feature_out_list[2](torch.cat(grid_feature, -1).detach()).detach()))
            #     else:
            #         hidden_list.append(torch.relu(self.feature_out_list[0](torch.cat(grid_feature, -1).detach()).detach()))
            #         hidden_list.append(torch.relu(self.feature_out_list[1](torch.cat(grid_feature, -1).detach()).detach()))
            #         hidden_list.append(torch.relu(self.feature_out_list[2](torch.cat(grid_feature, -1))))

            # g1_list = []
            # g2_list = []
            # for i in range(len(grid_feature)):
            #     g1_list.append(grid_feature[i][...,:16])
            #     g2_list.append(grid_feature[i][...,16:])
            # hidden_list.append(torch.relu(self.feature_out_list[0](torch.cat(g1_list, -1))))
            # hidden_list.append(torch.relu(self.feature_out_list[1](torch.cat(g2_list, -1))))

            # hidden_list.append(torch.relu(self.feature_out_list[0](torch.cat(grid_feature, -1))))
            # hidden_list.append(torch.relu(self.feature_out_list[1](torch.cat(grid_feature, -1))))
            
        return hidden_list
    
    @property
    def get_empty_ratio(self):
        return self.ratio
    def forward(self, rays_pts_emb, scales_emb=None, rotations_emb=None, opacity = None,shs_emb=None, time_feature=None, time_emb=None, stage="coarse", now_iteration=0):
        if time_emb is None:
            return self.forward_static(rays_pts_emb[:,:3])
        else:
            return self.forward_dynamic(rays_pts_emb, scales_emb, rotations_emb, opacity, shs_emb, time_feature, time_emb, stage, now_iteration)

    def forward_static(self, rays_pts_emb):
        grid_feature = self.grid(rays_pts_emb[:,:3])
        dx = self.static_mlp(grid_feature)
        return rays_pts_emb[:, :3] + dx
    
    def deformation_group_process(self, is_no, embs, deforms, hidden_list, stage, now_iteration):
        deform_res_list = []
        for i in range(len(is_no)):
            if is_no[i]:
                deform_res_list.append(embs[i])
            else:
                deform_res = torch.zeros_like(embs[i])
                deform_res += embs[i]
                # deform_res += deforms[i][0](hidden_list[0])
                # deform_res += deforms[i][1](hidden_list[1])
                if stage == "fine1":
                    deform_res += deforms[i][0](hidden_list[0])
                if stage == "fine2":
                    deform_res += deforms[i][0](hidden_list[0])
                    deform_res += deforms[i][1](hidden_list[1])
                # if stage == "fine3":
                #     deform_res += deforms[i][0](hidden_list[0])
                #     deform_res += deforms[i][1](hidden_list[1])
                #     deform_res += deforms[i][2](hidden_list[2])
                # if stage == "fine1":
                #     deform_res += deforms[i][0](hidden_list[0])
                # if stage == "fine2" or stage == "fine3":
                #     if now_iteration % 400 >= 0 and now_iteration % 400 < 200:
                #         deform_res += deforms[i][0](hidden_list[0])
                #         deform_res += deforms[i][1](hidden_list[1].detach()).detach()
                #     else:
                #         deform_res += deforms[i][0](hidden_list[0].detach()).detach()
                #         deform_res += deforms[i][1](hidden_list[1])
                # if stage == "fine3":
                #     if now_iteration % 600 >= 0 and now_iteration % 600 < 200:
                #         deform_res += deforms[i][0](hidden_list[0])
                #         deform_res += deforms[i][1](hidden_list[1].detach()).detach()
                #         deform_res += deforms[i][2](hidden_list[2].detach()).detach()
                #     elif now_iteration % 600 >= 200 and now_iteration % 600 < 400:
                #         deform_res += deforms[i][0](hidden_list[0].detach()).detach()
                #         deform_res += deforms[i][1](hidden_list[1])
                #         deform_res += deforms[i][2](hidden_list[2].detach()).detach()
                #     else:
                #         deform_res += deforms[i][0](hidden_list[0].detach()).detach()
                #         deform_res += deforms[i][1](hidden_list[1].detach()).detach()
                #         deform_res += deforms[i][2](hidden_list[2])
                deform_res_list.append(deform_res)
        return tuple(deform_res_list)
    
    def get_dx_for_init(self, rays_pts_emb, time_emb):
        grid_feature = self.grid(rays_pts_emb[:,:3], time_emb[:,:1])
        hidden_list = []
        # hidden_list.append(torch.relu(self.feature_out_list[0](torch.cat(grid_feature, -1).detach())))
        # hidden_list.append(torch.relu(self.feature_out_list[1](torch.cat(grid_feature[1:], -1).detach())))
        # hidden_list.append(torch.relu(self.feature_out_list[2](grid_feature[-1]).detach()))
        # breakpoint()
        hidden_list.append(torch.relu(self.feature_out_list[0](torch.cat(grid_feature, -1).detach())))
        hidden_list.append(torch.relu(self.feature_out_list[1](torch.cat(grid_feature, -1).detach())))
        hidden_list.append(torch.relu(self.feature_out_list[2](torch.cat(grid_feature, -1).detach())))

        # g1_list = []
        # g2_list = []
        # for i in range(len(grid_feature)):
        #     g1_list.append(grid_feature[i][...,:16])
        #     g2_list.append(grid_feature[i][...,16:])
        
        # hidden_list.append(torch.relu(self.feature_out_list[0](torch.cat(g1_list, -1))))
        # hidden_list.append(torch.relu(self.feature_out_list[1](torch.cat(g2_list, -1))))
        # hidden_list.append(torch.relu(self.feature_out_list[1](torch.cat(g2_list, -1))))

        dx_list = []
        for i in range(self.multires_num):
            dx_list.append(self.pos_deform_list[i](hidden_list[i]))
        return dx_list

    def forward_dynamic(self,rays_pts_emb, scales_emb, rotations_emb, opacity_emb, shs_emb, time_feature, time_emb, stage="coarse", now_iteration=0):
        hidden_list = self.query_time(rays_pts_emb, scales_emb, rotations_emb, time_feature, time_emb, stage, now_iteration)
        
        pts, scales, rotations, opacity, shs = \
            self.deformation_group_process(
                [self.args.no_dx, self.args.no_ds, self.args.no_dr, self.args.no_do, self.args.no_dshs],
                [rays_pts_emb[:,:3], scales_emb[:,:3], rotations_emb[:,:4], opacity_emb[:,:1], shs_emb],
                [self.pos_deform_list, self.scales_deform_list, self.rotations_deform_list, self.opacity_deform_list, self.shs_deform_list],
                hidden_list,
                stage,
                now_iteration
            )

        return pts, scales, rotations, opacity, shs
    
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" not in name:
                parameter_list.append(param)
        return parameter_list
    def get_grid_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" in name:
                parameter_list.append(param)
        return parameter_list
    
    
class deform_network(nn.Module):
    def __init__(self, args) :
        super(deform_network, self).__init__()
        net_width = args.net_width
        timebase_pe = args.timebase_pe
        defor_depth= args.defor_depth
        posbase_pe= args.posebase_pe
        scale_rotation_pe = args.scale_rotation_pe
        opacity_pe = args.opacity_pe
        timenet_width = args.timenet_width
        timenet_output = args.timenet_output
        grid_pe = args.grid_pe
        times_ch = 2*timebase_pe+1
        self.timenet = nn.Sequential(
        nn.Linear(times_ch, timenet_width), nn.ReLU(),
        nn.Linear(timenet_width, timenet_output))
        self.deformation_net = Deformation(W=net_width, D=defor_depth, input_ch=(3)+(3*(posbase_pe))*2, grid_pe=grid_pe, input_ch_time=timenet_output, args=args)
        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2**i) for i in range(scale_rotation_pe)]))
        self.register_buffer('opacity_poc', torch.FloatTensor([(2**i) for i in range(opacity_pe)]))
        self.apply(initialize_weights)
        # print(self)

    def forward(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None, stage="coarse", now_iteration=0):
        return self.forward_dynamic(point, scales, rotations, opacity, shs, times_sel, stage, now_iteration)
    @property
    def get_aabb(self):
        
        return self.deformation_net.get_aabb
    @property
    def get_empty_ratio(self):
        return self.deformation_net.get_empty_ratio
        
    def forward_static(self, points):
        points = self.deformation_net(points)
        return points
    def forward_dynamic(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None, stage="coarse", now_iteration=0):
        # times_emb = poc_fre(times_sel, self.time_poc)
        point_emb = poc_fre(point,self.pos_poc)
        scales_emb = poc_fre(scales,self.rotation_scaling_poc)
        rotations_emb = poc_fre(rotations,self.rotation_scaling_poc)
        # time_emb = poc_fre(times_sel, self.time_poc)
        # times_feature = self.timenet(time_emb)
        means3D, scales, rotations, opacity, shs = self.deformation_net(point_emb,
                                                  scales_emb,
                                                rotations_emb,
                                                opacity,
                                                shs,
                                                None,
                                                times_sel,
                                                stage,
                                                now_iteration)
        return means3D, scales, rotations, opacity, shs
    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters() + list(self.timenet.parameters())
    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # init.constant_(m.weight, 0)
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)
            # init.constant_(m.bias, 0)
def poc_fre(input_data,poc_buf):

    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin,input_data_cos], -1)
    return input_data_emb



class DeformationModel:
    def __init__(self, hyper, opt, scene):
        self.deform = deform_network(hyper)
        self.deform.deformation_net.set_aabb(scene.xyz_max, scene.xyz_min)
        self.deform = self.deform.to("cuda")

        l = [
            {'params': list(self.deform.get_mlp_parameters()), 'lr': opt.deformation_lr_init * scene.cameras_extent, "name": "deformation"},
            {'params': list(self.deform.get_grid_parameters()), 'lr': opt.grid_lr_init * scene.cameras_extent, "name": "grid"},
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.deformation_scheduler_args = get_expon_lr_func(lr_init=opt.deformation_lr_init*scene.cameras_extent,
                                                    lr_final=opt.deformation_lr_final*scene.cameras_extent,
                                                    lr_delay_mult=opt.deformation_lr_delay_mult,
                                                    max_steps=opt.position_lr_max_steps)    
        self.grid_scheduler_args = get_expon_lr_func(lr_init=opt.grid_lr_init*scene.cameras_extent,
                                                    lr_final=opt.grid_lr_final*scene.cameras_extent,
                                                    lr_delay_mult=opt.deformation_lr_delay_mult,
                                                    max_steps=opt.position_lr_max_steps)
        
    def state_dict(self):
        return self.deform.state_dict()
    
    def load_state_dict(self, params):
        self.deform.load_state_dict(params)
    
    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if  "grid" in param_group["name"]:
                lr = self.grid_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "deformation":
                lr = self.deformation_scheduler_args(iteration)
                param_group['lr'] = lr
    

    def compute_plane_smoothness(self, t):
        batch_size, c, h, w = t.shape
        # Convolve with a second derivative filter, in the time dimension which is dimension 2
        first_difference = t[..., 1:, :] - t[..., :h-1, :]  # [batch, c, h-1, w]
        second_difference = first_difference[..., 1:, :] - first_difference[..., :h-2, :]  # [batch, c, h-2, w]
        # Take the L2 norm of the result
        return torch.square(second_difference).mean()

    def _plane_regulation(self):
        multi_res_grids = self.deform.deformation_net.grid.grids
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids =  [0,1,3]
            for grid_id in time_grids:
                total += self.compute_plane_smoothness(grids[grid_id])
        return total
    def _time_regulation(self):
        multi_res_grids = self.deform.deformation_net.grid.grids
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids =[2, 4, 5]
            for grid_id in time_grids:
                total += self.compute_plane_smoothness(grids[grid_id])
        return total
    def _l1_regulation(self):
        multi_res_grids = self.deform.deformation_net.grid.grids

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

    def compute_regulation(self, hyper):
        return hyper.plane_tv_weight*self._plane_regulation() \
               + hyper.time_smoothness_weight*self._time_regulation() \
               + hyper.l1_time_planes*self._l1_regulation()