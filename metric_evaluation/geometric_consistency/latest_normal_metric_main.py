import os, glob
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import argparse, tqdm


def get_algorithm_data(data_dir):
    all_normal_data = sorted(glob.glob(os.path.join(data_dir, "normal_world", "*.npy")))
    all_batch_data = sorted(glob.glob(os.path.join(data_dir, "batch_data", "*.npy")))
    all_opacity_data = sorted(glob.glob(os.path.join(data_dir, "opacity", "*.png")))
    all_rgb_data = sorted(glob.glob(os.path.join(data_dir, "rgb_images", "*.png")))
    all_depth_anything = sorted(glob.glob(os.path.join(data_dir, "depth_anything", "*.npy")))
    sel_rgb_data = []
    for img in all_rgb_data:
        if 'rgba' in img: continue
        sel_rgb_data.append(img)
    all_rgb_data = sel_rgb_data
    return all_normal_data, all_batch_data, all_opacity_data, all_rgb_data, all_depth_anything


def get_normal_transformed(normal, transformation):
    normal = normal.reshape(-1, 3)
    normal_transformed = transformation @ normal.transpose()
    normal_transformed = normal_transformed.swapaxes(0, 1)
    normal_transformed = normal_transformed.reshape(512, 512, 3)
    normal_transformed = normal_transformed[:,:,:3]
    return normal_transformed

def normalize_vectors(vectors, axis=-1, eps=1e-8):
    norms = np.linalg.norm(vectors, axis=axis, keepdims=True)
    return vectors / (norms + eps)

def depth_map_to_normal_map(depth_map):
    depth_map *= -1

    # # Calculate gradients using Sobel operator
    # gradient_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
    # gradient_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate gradients using Sobel operator
    gradient_y = np.gradient(depth_map, axis=0) 
    gradient_x = np.gradient(depth_map, axis=1) 


    # Calculate surface normals
    normal_x = gradient_x
    normal_y = gradient_y
    normal_z = np.ones_like(depth_map)

    # Normalize normals
    norm = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
    normal_x /= norm
    normal_y /= norm
    normal_z /= norm

    # # Map normalized values to the range [0, 1]
    # normal_x = (normal_x + 1) / 2
    # normal_y = (normal_y + 1) / 2
    # normal_z = (normal_z + 1) / 2

    # Create RGB normal map
    # normal_map = np.stack([normal_x, normal_y, normal_z], axis=-1)
    normal_map = np.stack([normal_x, normal_y, normal_z], axis=-1)
    # normal_map = normalize_vectors(normal_map)
    return normal_map


def compute_normal_metric(normal_camera, depth_anything_normal_camera, opacity_map, ignorance_threshold=0.4):
    normal_metric = np.arccos(np.sum(depth_anything_normal_camera.reshape(-1,3) * normal_camera.reshape(-1,3), axis=-1).reshape(512, 512))
    # normal_metric[normal_metric<ignorance_threshold] = 0.
    normal_metric = normal_metric * opacity_map
    # normal_metric_mask = (normal_metric>0.) * 1
    return normal_metric #, normal_metric_mask
    
def compute_normal_metric_aggregated(normal_camera, depth_anything_normal_camera, opacity_map, ignorance_threshold=0.6):
        
    def shift_normal_map(normal_map, i, j):
        h, w, _ = normal_map.shape
        
        # Calculate the padded dimensions
        pad_h_start = max(0, -i)
        pad_h_end = max(0, i)
        
        pad_w_start = max(0, -j)
        pad_w_end = max(0, j)
        
        # Create a new array with the padded dimensions
        padded_map = np.pad(normal_map, ((pad_h_start, pad_h_end), (pad_w_start, pad_w_end), (0, 0)), mode='constant')
        
        # Calculate the slice dimensions for copying
        slice_h_start = max(0, i)
        slice_h_end = min(h + max(0, i), padded_map.shape[0])
        
        slice_w_start = max(0, j)
        slice_w_end = min(w + max(0, j), padded_map.shape[1])
        
        # Slice and return the shifted map
        shifted_map = padded_map[slice_h_start:slice_h_end, slice_w_start:slice_w_end, :]
        
        return shifted_map

    def concatenate_shifted_maps(normal_map, range_i, range_j):
        # Create an empty array to store concatenated maps
        concatenated_maps = [normal_map]
        
        for i in range_i:
            for j in range_j:
                shifted_map = shift_normal_map(normal_map, i, j)
                concatenated_maps.append(shifted_map)
        
        return np.concatenate(concatenated_maps, axis=2)

    # Example usage (Assuming normal_map is your normal map of shape (h, w, 3))
    range_i = range(-5, 6)
    range_j = range(-5, 6)
    normal_camera_clone = concatenate_shifted_maps(normal_camera, range_i, range_j)
    depth_anything_normal_camera_clone = concatenate_shifted_maps(depth_anything_normal_camera, range_i, range_j)
    
    normal_metric = np.arccos(np.sum(depth_anything_normal_camera_clone.reshape(-1,1*11*11+1,3) * normal_camera_clone.reshape(-1,1*11*11+1,3), axis=-1))
    normal_metric = np.max(normal_metric, axis=-1).reshape(512, 512)
    normal_metric = normal_metric * opacity_map
    normal_metric[normal_metric<ignorance_threshold] = 0.
    normal_metric_mask = (normal_metric>0.) * 1
    return normal_metric, normal_metric_mask


def compute_normal_alignment_metric(data_path, save_normal=True, visualize_normal=True, cmap = 'gist_rainbow'):
    
    all_normal_data, all_batch_data, all_opacity_data, all_rgb_data, all_depth_anything = \
        get_algorithm_data(data_path)

    assert(len(all_normal_data)==len(all_batch_data))
    assert(len(all_normal_data)==len(all_rgb_data))
    print(len(all_normal_data), len(all_batch_data), len(all_opacity_data), len(all_rgb_data), len(all_depth_anything))
    
    for idx in tqdm.tqdm(range(len(all_normal_data))):
        if idx %2!=0: continue
        
        batch_data = np.load(all_batch_data[idx], allow_pickle=True).item()
        normal_world = np.load(all_normal_data[idx])
        rgb_image = np.array(Image.open(all_rgb_data[idx]))
        depth_anything = np.load(all_depth_anything[idx])
        opacity_map = np.array(Image.open(all_opacity_data[idx]))[...,0] / 255.
        normal_world = cv2.resize(normal_world, (512, 512))
        depth_anything = cv2.resize(depth_anything, (512, 512))
        rgb_image = cv2.resize(rgb_image, (512, 512))
        opacity_map = cv2.resize(opacity_map, (512, 512))
        opacity_map = (opacity_map>0)*1.

        depth_anything_normal_camera = depth_map_to_normal_map(depth_anything)
            
        c2w = batch_data['c2w'].cpu().numpy()
        c2w = batch_data['c2w'].cpu().numpy()[:,:3,:3]
        w2c = np.linalg.inv(c2w[0]) #.transpose()

        # threestudio normal adjustment
        normal_world = (normal_world * 2.) - 1
        normal_world = normalize_vectors(normal_world)
        normal_camera = get_normal_transformed(normal_world, w2c)
        normal_camera = normalize_vectors(normal_camera)
        normal_camera = (normal_camera + 1.0)/2.
        normal_camera = normal_camera * opacity_map[...,None]
        # from: https://github.com/deepseek-ai/DreamCraft3D/blob/b20d9386198b3965c78ba71c98156628fc41ecd3/threestudio/systems/dreamcraft3d.py#L176
        normal_camera[..., 0] = 1 - normal_camera[..., 0]
        normal_camera = (2 * normal_camera - 1)
        normal_camera = (normal_camera + 1.) /2.
        normal_camera = normalize_vectors(normal_camera)
        normal_camera = normal_camera * opacity_map[...,None]
        
        # depth anything normal adjustment
        depth_anything_normal_camera[...,-1] *= -1
        depth_anything_normal_camera = (depth_anything_normal_camera + 1.) / 2.
        depth_anything_normal_camera = (1 - 2 * depth_anything_normal_camera)  # [B, 3]
        depth_anything_normal_camera = (depth_anything_normal_camera + 1.) /2.
        depth_anything_normal_camera = normalize_vectors(depth_anything_normal_camera)
        depth_anything_normal_camera = depth_anything_normal_camera * opacity_map[...,None]

        np.save(all_batch_data[idx].replace('batch_data', 'normal_camera_3'), np.asarray(normal_camera))
        np.save(all_batch_data[idx].replace('batch_data', 'depth_anything_normal_camera_3'), np.asarray(depth_anything_normal_camera))
        
        Image.fromarray(np.asarray(normal_camera*255, dtype=np.uint8)).save(all_rgb_data[idx].replace('rgb_images', 'normal_camera_3'))
        Image.fromarray(np.asarray(depth_anything_normal_camera*255, dtype=np.uint8)).save(all_rgb_data[idx].replace('rgb_images', 'depth_anything_normal_camera_3'))


        normal_metric = compute_normal_metric(normal_camera, depth_anything_normal_camera, opacity_map)
        # normal_metric_agg, normal_metric_mask_agg = compute_normal_metric_aggregated(normal_camera, depth_anything_normal_camera, opacity_map)


        # vis_images = [rgb_image, normal_camera, depth_anything_normal_camera, rgb_image * normal_delta_filter_mask[...,None], rgb_image * normal_delta_filter_mask_agg[...,None]]
        # titles = ["RGB", "Normal Rendered", "Normal (Depth Anything)", "Error Map (0.4)", "Error Map (Conv) (0.6)"]
        # visualize_images(vis_images, titles)


        # out = np.sum(omnidata_normal_camera.reshape(-1,3) * normal_camera.reshape(-1,3), axis=-1).reshape(512, 512) 
        # out = out * opacity_map
        # plt.imshow(out)
        # # plt.scatter([200], [400], c='r')
        # plt.show()
        # print("normal alignment metric: ", out.sum()/opacity_map.sum())
        # if idx==10:
        #     break

        # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        # print(normal_metric.shape, normal_metric.min(), normal_metric.max(), all_batch_data[idx].replace('batch_data', 'normal_metric'))
        np.save(all_batch_data[idx].replace('batch_data', 'normal_metric_3'), normal_metric)
        # Image.fromarray(normal_metric / np.pi).convert("L").save(all_batch_data[idx].replace('batch_data', 'normal_metric').replace('.npy', '.png'))
    print('normal_metric done')


def main(args):
    if not os.path.exists(os.path.join(args.data_path, "normal_metric_3")):
        print('mkdir -p ' + os.path.join(args.data_path, "normal_metric_3"))
    else: os.system('rm -r -v ' + os.path.join(args.data_path, "normal_metric_3"))
    os.system('mkdir -p ' + os.path.join(args.data_path.replace("'", "\\'"), "normal_metric_3"))
    if not os.path.exists(os.path.join(args.data_path, "normal_camera_3")):
        print('mkdir -p ' + os.path.join(args.data_path, "normal_camera_3"))
    else: os.system('rm -r -v ' + os.path.join(args.data_path, "normal_camera_3"))
    os.system('mkdir -p ' + os.path.join(args.data_path.replace("'", "\\'"), "normal_camera_3"))
    if not os.path.exists(os.path.join(args.data_path, "depth_anything_normal_camera_3")):
        print('mkdir -p ' + os.path.join(args.data_path, "depth_anything_normal_camera_3"))
    else: os.system('rm -r -v ' + os.path.join(args.data_path, "depth_anything_normal_camera_3"))
    os.system('mkdir -p ' + os.path.join(args.data_path.replace("'", "\\'"), "depth_anything_normal_camera_3"))
    compute_normal_alignment_metric(data_path=args.data_path)
    

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', required=True, type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)