import torch

imgs,_,_,pc,_ = torch.load("datasets/dnerf/trex/trex_preprcess.pth", weights_only=True)

imgs = imgs.permute(0,2,3,1)
pc_v = pc.reshape(-1, 3) * 2.6 - 1.3
colors = imgs.reshape(-1, 3)

mask = torch.rand((pc_v.shape[0]))
final_pc = pc_v[mask < 0.01]
final_color = colors[mask < 0.01]

torch.save((final_pc, final_color), "datasets/dnerf/trex/pointcloud.pth")