_base_="default.py"
ModelParams=dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [64, 64, 64, 150]
    },
)
OptimizationParams=dict(
    fine_iterations_1 = 3000,
    fine_iterations_2 = 14000,
    max_primitives = 800_000,
)