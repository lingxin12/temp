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

import math
import os
import sys
from argparse import ArgumentParser, Namespace


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, action="store_true"
                    )
                elif t == list or t == tuple:
                    group.add_argument(
                        "--" + key,
                        ("-" + key[0:1]),
                        nargs="+",
                        type=type(value[0]),
                        default=value,
                    )
                else:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, type=t
                    )
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                elif t == list or t == tuple:
                    group.add_argument(
                        "--" + key, nargs="+", type=type(value[0]), default=value
                    )
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = True
        self.data_device = "cpu"
        self.render_spline = False
        self.use_neural_network = False
        self.eval = False
        self.num_additional_pts = 10000
        self.additional_size_multi = 1.0
        self.num_spline_frames = 480
        self.glo_latent_dim = 64
        self.max_opacity = 0.99
        self.tmin = 0.2

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = True
        self.compute_cov3D_python = False
        self.enable_GLO = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.coarse_iterations = 3000
        self.fine_iterations_1 = 10000
        self.fine_iterations_2 = 10000
        self.fine_iterations_3 = 10000

        self.betas = [0.9, 0.999]

        self.position_lr_final = 4e-7 #0.0000004
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.position_lr_init = 4e-5 #0.00004

        self.glo_lr = 0.01
        self.glo_network_lr = 0.00005

        self.feature_lr = 0.0025
        self.feature_rest_lr = 0.00025
        self.bg_lr = 0.0
        self.opacity_lr = 0.0125
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.min_opacity = 0.005
        self.min_split_opacity = 0.01
        self.percent_dense = 0.0025
        self.lambda_dssim = 0.2

        self.lambda_anisotropic = 1e-1
        self.lambda_distortion = 0
        self.sh_up_interval = 2000

        self.densification_interval = 200
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000

        # self.densify_grad_threshold: float = 2.5e-7
        self.densify_grad_threshold: float = 3e-7

        self.clone_grad_threshold: float = 1e-1

        self.center_pixel = False
        self.fallback_xy_grad = False

        self.random_background = False


        self.deformation_lr_init = 0.00016
        self.deformation_lr_final = 0.000016
        self.deformation_lr_delay_mult = 0.01
        self.grid_lr_init = 0.0016
        self.grid_lr_final = 0.00016
        self.batch_size = 1

        self.max_primitives = 800_000

        super().__init__(parser, "Optimization Parameters")


class ModelHiddenParams(ParamGroup):
    def __init__(self, parser):
        self.net_width = 64 # width of deformation MLP, larger will increase the rendering quality and decrase the training/rendering speed.
        self.timebase_pe = 4 # useless
        self.defor_depth = 1 # depth of deformation MLP, larger will increase the rendering quality and decrase the training/rendering speed.
        self.posebase_pe = 10 # useless
        self.scale_rotation_pe = 2 # useless
        self.opacity_pe = 2 # useless
        self.timenet_width = 64 # useless
        self.timenet_output = 32 # useless
        self.bounds = 1.6 
        self.plane_tv_weight = 0.0001 # TV loss of spatial grid
        self.time_smoothness_weight = 0.01 # TV loss of temporal grid
        self.l1_time_planes = 0.0001  # TV loss of temporal grid
        self.kplanes_config = {
                             'grid_dimensions': 2,
                             'input_coordinate_dim': 4,
                             'output_coordinate_dim': 32,
                             'resolution': [64, 64, 64, 25]  # [64,64,64]: resolution of spatial grid. 25: resolution of temporal grid, better to be half length of dynamic frames
                            }
        self.multires = [1, 2, 4, 8] # multi resolution of voxel grid
        self.no_dx=False # cancel the deformation of Gaussians' position
        self.no_grid=False # cancel the spatial-temporal hexplane.
        self.no_ds=True # cancel the deformation of Gaussians' scaling
        self.no_dr=True # cancel the deformation of Gaussians' rotations
        self.no_do=True # cancel the deformation of Gaussians' opacity
        self.no_dshs=True # cancel the deformation of SH colors.
        self.empty_voxel=False # useless
        self.grid_pe=0 # useless, I was trying to add positional encoding to hexplane's features
        self.static_mlp=False # useless
        self.apply_rotation=False # useless

        
        super().__init__(parser, "ModelHiddenParams")



def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
