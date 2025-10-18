

OptimizationParams = dict(

    coarse_iterations = 10000,
    deformation_lr_init = 0.00016,
    deformation_lr_final = 0.0000016,
    deformation_lr_delay_mult = 0.01,
    grid_lr_init = 0.0016,
    grid_lr_final = 0.000016,
    iterations = 30000,
    pruning_interval = 8000,
    percent_dense = 0.01,
    render_process=False,
    # no_do=False,
    # no_dshs=False
    
    # opacity_reset_interval=30000

    position_lr_final = 4e-7, #0.0000004
    position_lr_delay_mult = 0.01,
    position_lr_max_steps = 20_000,
    position_lr_init = 4e-5, #0.00004

    densify_from_iter = 500,
    densify_until_iter = 15_000,

)

ModelHiddenParams = dict(

    multires = [1, 2],
    defor_depth = 0,
    net_width = 64,
    plane_tv_weight = 0.0001,
    time_smoothness_weight = 0.01,
    l1_time_planes =  0.0001,
    weight_decay_iteration=0,
    bounds=1.6
)
