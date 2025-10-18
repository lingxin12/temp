
import os
import numpy as np
import glob
import sys
import json
from PIL import Image
from tqdm import tqdm
import shutil
def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

root_dir = sys.argv[1]
colmap_dir = os.path.join(root_dir,"sparse_")
if not os.path.exists(colmap_dir):
    os.makedirs(colmap_dir)
imagecolmap_dir = os.path.join(root_dir,"image_colmap")
if not os.path.exists(imagecolmap_dir):
    os.makedirs(imagecolmap_dir)
object_images_file = open(os.path.join(colmap_dir,"images.txt"),"w")
object_cameras_file = open(os.path.join(colmap_dir,"cameras.txt"),"w")


images_dir_list = [i for i in os.listdir(os.path.join(root_dir)) if i[:3]=="cam" and len(i)==5]
images_dir_list = sorted(images_dir_list)
images_basedir_list = [f"{i:0>4}.png" for i in range(0,300,30)]
colmap_images_name = []
for i in images_dir_list:
    for j in images_basedir_list:
        colmap_images_name.append(f"{i}_{j}")

poses_arr = np.load(os.path.join(root_dir, "poses_bounds.npy"))
poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
near_fars = poses_arr[:, -2:]
H, W, focal = poses[0, :, -1]
poses_all = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)

idx=0
for cam_idx in range(len(images_dir_list)):
    pose = np.array(poses_all[cam_idx])
    R = pose[:3,:3]
    R = -R
    R[:,0] = -R[:,0]
    T = -pose[:3,3].dot(R)
    T = [str(i) for i in T]
    qevc = [str(i) for i in rotmat2qvec(R.T)]
    
    for img_idx in range(len(images_basedir_list)):
        print(idx+1," ".join(qevc)," ".join(T),cam_idx+1,colmap_images_name[idx],"\n",file=object_images_file)
        print(idx+1,"SIMPLE_PINHOLE",W/2,H/2,focal/2,W/4,H/4,file=object_cameras_file)
        shutil.copy(os.path.join(root_dir,images_dir_list[cam_idx],"images",images_basedir_list[img_idx]),\
                    os.path.join(imagecolmap_dir,colmap_images_name[idx]))
        idx+=1

object_point_file = open(os.path.join(colmap_dir,"points3D.txt"),"w")

object_cameras_file.close()
object_images_file.close()
object_point_file.close()
