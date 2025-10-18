_base_="default.py"

ModelParams=dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [64, 64, 64, 80]
    },
)

OptimizationParams=dict(
    fine_iterations_1 = 7000,
    fine_iterations_2 = 13000,
    densify_until_iter = 20000,
    position_lr_max_steps = 20000,
    max_primitives = 800_000,
)