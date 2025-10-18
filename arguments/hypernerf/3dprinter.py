_base_="default.py"

ModelParams=dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [64, 64, 64, 100]
    },
    no_ds=True,
    no_dr=True,
    no_do=True,
    no_dshs=True,
)

OptimizationParams=dict(
    fine_iterations_1 = 3000,
    fine_iterations_2 = 4000,
    densify_until_iter = 7000,
    position_lr_max_steps = 7000,
    max_primitives = 1200_000,
)