import os, glob, csv, json
import torch
import argparse
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import trimesh, tqdm
import open3d as o3d
import sys
sys.path.append('..')
from utils.geometric_utils import *
from utils.dino_utils import ViTExtractor, extract_dino_features, dino_vis_pca_mask, visualize_pca_features
from utils.pytorch3d_utils import *
# os.environ['DISPLAY'] = ":99"

sys.path.append('../FeatUp/')
# from featup.util import norm, unnorm
# from featup.plotting import plot_feats



class dino_args:
    load_size = 224
    stride = 14 # stride of first convolution layer. small stride -> higher resolution.
    model_type = "dinov2_vits14"
    facet = "token"
    layer = 11 # for vits layer = 11
    bin = None
    patch_size = 14
    device = 'cuda:0'


def get_algorithm_data(data_dir):
    all_normal_data = sorted(glob.glob(os.path.join(data_dir, "normal_world", "*.npy")))
    all_batch_data = sorted(glob.glob(os.path.join(data_dir, "batch_data", "*.npy")))
    all_opacity_data = sorted(glob.glob(os.path.join(data_dir, "opacity", "*.png")))
    all_rgb_data = sorted(glob.glob(os.path.join(data_dir, "rgb_images", "*.png")))
    all_omnidata_normals = sorted(glob.glob(os.path.join(data_dir, "normal_camera_omnidata", "*_normal.png")))
    all_depth_anything = sorted(glob.glob(os.path.join(data_dir, "depth_anything", "*.npy")))
    all_omnidata_depths = sorted(glob.glob(os.path.join(data_dir, "depth_camera_omnidata", "*_depth.png")))
    sel_rgb_data = []
    print(len(all_rgb_data), data_dir, "checks...")
    for img in all_rgb_data:
        if 'rgba' in img: continue
        sel_rgb_data.append(img)
    all_rgb_data = sel_rgb_data
    return all_normal_data, all_batch_data, all_opacity_data, all_rgb_data, all_depth_anything, all_omnidata_normals


def extract_dino_data(data_dir):
    
    all_normal_data, all_batch_data, all_opacity_data, all_rgb_data, all_depth_anything, all_omnidata_normals = \
        get_algorithm_data(data_dir)
      
    dino_extractor = ViTExtractor(dino_args.model_type, dino_args.stride, device=dino_args.device)
    
    upsampler = torch.hub.load("mhamilton723/FeatUp", 'dinov2', use_norm=True).to(dino_args.device)
    upsampler.model = upsampler.model.to(dino_args.device)
    
    selected_opacity_map = []
    selected_upsampled_dino_descriptors = []
    print(len(all_rgb_data))
    for idx in tqdm.tqdm(range(len(all_rgb_data))):
        if idx%2!=0: continue
        
        # normal_world = np.load(all_normal_data[idx])
        # depth_anything = np.load(all_depth_anything[idx])
        # depth_omnidata = np.asarray(Image.open(all_omnidata_depths[idx]))
        # normal_world = cv2.resize(normal_world, (512, 512))
        # depth_anything = cv2.resize(depth_anything, (512, 512))
        
        batch_data = np.load(all_batch_data[idx], allow_pickle=True).item()
        rgb_image = np.asarray(Image.open(all_rgb_data[idx]))
        opacity_map = np.array(Image.open(all_opacity_data[idx]))[...,0] / 255.
        rgb_image = cv2.resize(rgb_image, (512, 512))
        opacity_map = cv2.resize(opacity_map, (512, 512))
        opacity_map = (opacity_map>0)*1.

       
        with torch.no_grad():
            image_batch, image_pil = dino_extractor.preprocess(all_rgb_data[idx], dino_args.load_size, dino_args.patch_size)
            image_batch = image_batch.to(dino_args.device)
            hr_descriptor_feats = upsampler(image_batch)
            hr_descriptor_feats = hr_descriptor_feats.reshape(hr_descriptor_feats.shape[0], hr_descriptor_feats.shape[1], -1)
            hr_descriptor_feats = hr_descriptor_feats.permute(0,2,1)[None]


        selected_opacity_map.append(torch.from_numpy(opacity_map).to(dino_args.device))
        selected_upsampled_dino_descriptors.append(hr_descriptor_feats[0,0])
        
    print('extracting pca....')
    pca_dino_feature_list = dino_vis_pca_mask(selected_upsampled_dino_descriptors, selected_opacity_map, normalize_pca_output=True)
    feat_idx = 0
    if not os.path.exists(os.path.join(data_dir.replace("'", "\\'"), "all_pca_dino_feats")):
        os.system("mkdir -p " + os.path.join(data_dir.replace("'", "\\'"), "all_dino_feats"))
        os.system("mkdir -p " + os.path.join(data_dir.replace("'", "\\'"), "all_pca_dino_feats"))
    
    for idx in range(len(all_rgb_data)):
        if idx%2!=0: continue
        dino_upsampled_feat = selected_upsampled_dino_descriptors[feat_idx]
        dino_upsampled_pca_feat = pca_dino_feature_list[feat_idx]
            
        print(idx, dino_upsampled_feat.shape, dino_upsampled_pca_feat.shape, "dino and dino-pca feat shape check...")
        feat_idx += 1

        np.save(os.path.join(data_dir, "all_dino_feats", "{}.npy".format(str(idx).zfill(4))), dino_upsampled_feat.cpu().numpy().reshape(256, 256, -1))
        np.save(os.path.join(data_dir, "all_pca_dino_feats", "{}.npy".format(str(idx).zfill(4))), dino_upsampled_pca_feat.reshape(256, 256, -1))
        
        dino_upsampled_pca_feat = (dino_upsampled_pca_feat - dino_upsampled_pca_feat.min(axis=-1)[...,None]) / (dino_upsampled_pca_feat.max(axis=-1)[...,None] - dino_upsampled_pca_feat.min(axis=-1)[...,None])
        Image.fromarray(np.asarray(dino_upsampled_pca_feat.reshape(256, 256, -1)[...,1:4]*255, dtype=np.uint8)).save(os.path.join(data_dir, 'all_pca_dino_feats', '{}.png'.format(str(idx).zfill(4))))

    os.system('''touch {}'''.format(os.path.join(data_dir.replace("'", "\\'"), "all_dino_feats", "dino_extracted.txt")))

def main(args):
    extract_dino_data(args.data_path)
    print("Dino feature extraction complete for data_path: {}".format(args.data_path))


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', required=True, type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)