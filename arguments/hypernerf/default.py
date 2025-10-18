ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [64, 64, 64, 150]
    },
    multires = [1,2,4],
    defor_depth = 1,
    net_width = 128,
    plane_tv_weight = 0.0002,
    time_smoothness_weight = 0.001,
    l1_time_planes =  0.0001,
    render_process=True
)
OptimizationParams = dict(
    # dataloader=True,
    iterations = 30_000,
    batch_size=2,
    coarse_iterations = 3000,
    fine_iterations_1 = 7000,
    fine_iterations_2 = 13000,
    fine_iterations_3 = 10000,
    # densification_interval=200,
    densify_until_iter = 20_000,
    opacity_reset_interval = 300000,

    position_lr_max_steps = 14_000,
    max_primitives = 800_000,
    # center_pixel = True,
    # feature_rest_lr = 0.0025 / 20.0,

    # grid_lr_init = 0.0016,
    # grid_lr_final = 16,
    # opacity_threshold_coarse = 0.005,
    # opacity_threshold_fine_init = 0.005,
    # opacity_threshold_fine_after = 0.005,
    # pruning_interval = 2000
)