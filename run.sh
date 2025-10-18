#!/bin/bash
# for set in bicycle flowers garden stump treehill room counter kitchen bonsai; do
# # for set in bicycle flowers garden stump treehill; do
# # for set in kitchen bonsai; do
#   # PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 python train.py -s /data/nerf_synthetic/$set --densify_grad_threshold=3e-7 --convert_SHs_python
#   PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 python train.py -s /data/nerf_synthetic/$set --densify_grad_threshold=3e-7 --eval
#   # PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 python train.py -s /data/nerf_synthetic/$set --densify_grad_threshold=3e-7 --use_neural_network --data_device cpu --eval --feature_rest_lr 0.0025
# done


# python train.py -s datasets/mipnerf360/bicycle -m output/mipnerf360/bicycle --eval --images "images_4"


# python train.py -s datasets/hypernerf/vrig/vrig-chicken -m output/hypernerf/vrig/vrig-chicken --eval --configs "arguments/hypernerf/chicken.py" --start_checkpoint output/chicken_chkpnt3000.pth
# python train.py -s datasets/hypernerf/vrig/broom2 -m output/hypernerf/vrig/broom2 --eval --configs "arguments/hypernerf/broom2.py" --start_checkpoint output/broom2_chkpnt3000.pth
# python train.py -s datasets/hypernerf/vrig/vrig-peel-banana -m output/hypernerf/vrig/vrig-peel-banana --eval --configs "arguments/hypernerf/banana.py" --start_checkpoint output/banana_chkpnt3000.pth
# python train.py -s datasets/hypernerf/vrig/vrig-3dprinter -m output/hypernerf/vrig/vrig-3dprinter --eval --configs "arguments/hypernerf/3dprinter.py"

# python train.py -s datasets/dnerf/trex -m output/dnerf/trex --eval --configs "arguments/dnerf/trex.py" --start_checkpoint output/trex_chkpnt10000.pth
# python train.py -s datasets/dnerf/bouncingballs -m output/dnerf/bouncingballs --eval --configs "arguments/dnerf/bouncingballs.py"

# python train.py -s datasets/dynerf/cut_roasted_beef -m output/dynerf/cut_roasted_beef --eval --configs "arguments/dynerf/cut_roasted_beef.py"
# python train.py -s datasets/dynerf/flame_salmon_1 -m output/dynerf/flame_salmon_1 --eval --configs "arguments/dynerf/default.py"

# /root/autodl-tmp/code/dust3r/datasets/trex/trex_preprcess.pth


# python train.py -s datasets/hypernerf/vrig/broom2 -m output/hypernerf/vrig/broom2 --eval --configs "arguments/hypernerf/broom2.py" --start_checkpoint output/hypernerf/vrig/broom2/03-14---15-39-59/chkpnts/chkpnt10000.pth --start_deform_checkpoint output/hypernerf/vrig/broom2/03-14---15-39-59/chkpnts/chkpnt_deform10000.pth
# python train.py -s datasets/hypernerf/vrig/vrig-chicken-temp -m output/hypernerf/vrig/vrig-chicken --eval --configs "arguments/hypernerf/chicken.py" --start_checkpoint output/chicken_chkpnt3000.pth --skip_coarse
# python train.py -s datasets/hypernerf/vrig/vrig-chicken-temp -m output/hypernerf/vrig/vrig-chicken --eval --configs "arguments/hypernerf/chicken.py" --start_checkpoint output/hypernerf/vrig/vrig-chicken/04-12---13-48-59/chkpnts/chkpnt7000.pth --start_deform_checkpoint output/hypernerf/vrig/vrig-chicken/04-12---13-48-59/chkpnts/chkpnt_deform7000.pth --skip_coarse



# python train.py -s datasets/hypernerf/vrig/vrig-chicken-temp -m output/hypernerf/vrig/vrig-chicken --eval --configs "arguments/hypernerf/chicken.py" --start_checkpoint output/hypernerf/vrig/vrig-chicken/05-03---15-53-39/chkpnts/fine1_chkpnt10000.pth --start_deform_checkpoint output/hypernerf/vrig/vrig-chicken/05-03---15-53-39/chkpnts/fine1_chkpnt_deform10000.pth
# python train.py -s datasets/hypernerf/vrig/vrig-chicken-temp -m output/hypernerf/vrig/vrig-chicken --eval --configs "arguments/hypernerf/chicken.py" --start_checkpoint output/hypernerf/vrig/vrig-chicken/05-04---09-06-34/chkpnts/fine1_chkpnt10000.pth --start_deform_checkpoint output/hypernerf/vrig/vrig-chicken/05-04---09-06-34/chkpnts/fine1_chkpnt_deform10000.pth
# python train.py -s datasets/hypernerf/vrig/vrig-chicken-temp -m output/hypernerf/vrig/vrig-chicken --eval --configs "arguments/hypernerf/chicken.py" --start_checkpoint output/hypernerf/vrig/vrig-chicken/05-04---10-58-34/chkpnts/fine2_chkpnt20000.pth --start_deform_checkpoint output/hypernerf/vrig/vrig-chicken/05-04---10-58-34/chkpnts/fine2_chkpnt_deform20000.pth


# python train.py -s datasets/hypernerf/vrig/vrig-peel-banana -m output/hypernerf/vrig/vrig-peel-banana --eval --configs "arguments/hypernerf/banana.py"
# python train.py -s datasets/hypernerf/vrig/broom2 -m output/hypernerf/vrig/broom2 --eval --configs "arguments/hypernerf/broom2.py"
# python train.py -s datasets/hypernerf/vrig/vrig-3dprinter -m output/hypernerf/vrig/vrig-3dprinter --eval --configs "arguments/hypernerf/broom2.py"

#hynerf interp
# python train.py -s datasets/hypernerf/interp/chickchicken -m output/hypernerf/interp/chickchicken --eval --configs "arguments/hypernerf/chickchicken.py"
python train.py -s datasets/hypernerf/interp/aleks-teapot -m output/hypernerf/interp/aleks-teapot --eval --configs "arguments/hypernerf/aleks-teapot.py"
python train.py -s datasets/hypernerf/interp/cut-lemon1 -m output/hypernerf/interp/cut-lemon1 --eval --configs "arguments/hypernerf/cut-lemon1.py"
python train.py -s datasets/hypernerf/interp/hand1-dense-v2 -m output/hypernerf/interp/hand1-dense-v2 --eval --configs "arguments/hypernerf/hand1-dense-v2.py"
python train.py -s datasets/hypernerf/interp/slice-banana -m output/hypernerf/interp/slice-banana --eval --configs "arguments/hypernerf/slice-banana.py"
python train.py -s datasets/hypernerf/interp/torchocolate -m output/hypernerf/interp/torchocolate --eval --configs "arguments/hypernerf/torchocolate.py"
