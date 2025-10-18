

workdir=$1  # datasets/vrig-chicken
datatype=$2 # blender, hypernerf, llff, dynerf
export CUDA_VISIBLE_DEVICES=0
rm -rf $workdir/sparse_
rm -rf $workdir/image_colmap
python scripts/"$datatype"2colmap.py $workdir  # python scripts/hypernerf2colmap.py datasets/vrig-chicken
rm -rf $workdir/colmap
rm -rf $workdir/colmap/sparse/0

mkdir $workdir/colmap  # mkdir datasets/vrig-chicken/colmap
cp -r $workdir/image_colmap $workdir/colmap/images  # cp -r datasets/vrig-chicken/image_colmap datasets/vrig-chicken/colmap/images
cp -r $workdir/sparse_ $workdir/colmap/sparse_custom # cp -r datasets/vrig-chicken/sparse_ datasets/vrig-chicken/colmap/sparse_custom
colmap feature_extractor --database_path $workdir/colmap/database.db --image_path $workdir/colmap/images  --SiftExtraction.max_image_size 4096 --SiftExtraction.max_num_features 16384 --SiftExtraction.estimate_affine_shape 1 --SiftExtraction.domain_size_pooling 1
# colmap feature_extractor --database_path datasets/vrig-chicken/colmap/database.db --image_path datasets/vrig-chicken/colmap/images  --SiftExtraction.max_image_size 4096 --SiftExtraction.max_num_features 16384 --SiftExtraction.estimate_affine_shape 1 --SiftExtraction.domain_size_pooling 1
python database.py --database_path $workdir/colmap/database.db --txt_path $workdir/colmap/sparse_custom/cameras.txt
# python database.py --database_path datasets/vrig-chicken/colmap/database.db --txt_path datasets/vrig-chicken/colmap/sparse_custom/cameras.txt

colmap exhaustive_matcher --database_path $workdir/colmap/database.db
# colmap exhaustive_matcher --database_path datasets/vrig-chicken/colmap/database.db
mkdir -p $workdir/colmap/sparse/0  # mkdir -p datasets/vrig-chicken/colmap/sparse/0

colmap point_triangulator --database_path $workdir/colmap/database.db --image_path $workdir/colmap/images --input_path $workdir/colmap/sparse_custom --output_path $workdir/colmap/sparse/0 --clear_points 1
# colmap point_triangulator --database_path datasets/vrig-chicken/colmap/database.db --image_path datasets/vrig-chicken/colmap/images --input_path datasets/vrig-chicken/colmap/sparse_custom --output_path datasets/vrig-chicken/colmap/sparse/0 --clear_points 1

mkdir -p $workdir/colmap/dense/workspace  # mkdir -p datasets/vrig-chicken/colmap/dense/workspace
colmap image_undistorter --image_path $workdir/colmap/images --input_path $workdir/colmap/sparse/0 --output_path $workdir/colmap/dense/workspace
# colmap image_undistorter --image_path datasets/vrig-chicken/colmap/images --input_path datasets/vrig-chicken/colmap/sparse/0 --output_path datasets/vrig-chicken/colmap/dense/workspace
colmap patch_match_stereo --workspace_path $workdir/colmap/dense/workspace
# colmap patch_match_stereo --workspace_path datasets/vrig-chicken/colmap/dense/workspace
colmap stereo_fusion --workspace_path $workdir/colmap/dense/workspace --output_path $workdir/colmap/dense/workspace/fused.ply
# colmap stereo_fusion --workspace_path datasets/vrig-chicken/colmap/dense/workspace --output_path datasets/vrig-chicken/colmap/dense/workspace/fused.ply --StereoFusion.min_num_pixels 2 --StereoFusion.max_reproj_error 8.0 --StereoFusion.max_depth_error 0.1 --StereoFusion.max_normal_error 30
