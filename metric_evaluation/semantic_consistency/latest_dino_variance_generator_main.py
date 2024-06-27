import os, glob, csv
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import tqdm
import trimesh
import open3d as o3d
import sys
# sys.path.append('..')
sys.path.append('/data/vision/torralba/sduggal/mnt_nfs_data/research/threed_eval/')
from utils.geometric_utils import *
from utils.dino_utils import ViTExtractor, extract_dino_features, dino_vis_pca_mask, visualize_pca_features
from utils.pytorch3d_utils import *
from pytorch3d.renderer import TexturesVertex
# os.environ['DISPLAY'] = ":99"
import argparse

sys.path.append('../FeatUp/')
# from featup.util import norm, unnorm
# from featup.plotting import plot_feats

def get_algorithm_data(data_dir):
    all_normal_data = sorted(glob.glob(os.path.join(data_dir, "normal_world", "*.npy")))
    all_batch_data = sorted(glob.glob(os.path.join(data_dir, "batch_data", "*.npy")))
    all_opacity_data = sorted(glob.glob(os.path.join(data_dir, "opacity", "*.png")))
    all_rgb_data = sorted(glob.glob(os.path.join(data_dir, "rgb_images", "*.png")))
    all_omnidata_normals = sorted(glob.glob(os.path.join(data_dir, "normal_camera_omnidata", "*_normal.png")))
    all_depth_anything = sorted(glob.glob(os.path.join(data_dir, "depth_anything", "*.npy")))
    all_omnidata_depths = sorted(glob.glob(os.path.join(data_dir, "depth_camera_omnidata", "*_depth.png")))
    all_dino_feat = sorted(glob.glob(os.path.join(data_dir, "all_dino_feats", "*.npy")))
    all_pca_dino_feat = sorted(glob.glob(os.path.join(data_dir, "all_pca_dino_feats", "*.npy")))
    return all_normal_data, all_batch_data, all_opacity_data, all_rgb_data, all_depth_anything, all_omnidata_normals, all_dino_feat, all_pca_dino_feat


def visualize_3d(meshes, vertex_colors=None, clean_via_connected_components=False, threshold=None):
    verts = meshes.verts_padded()
    faces = meshes.faces_padded()
    if vertex_colors is not None: vertex_colors = np.copy(vertex_colors.cpu())

    if clean_via_connected_components:
        o3d_mesh = make_mesh(verts[0].cpu().numpy(), faces[0].cpu().numpy(), verts[0].cpu().numpy())
        print("Cluster connected triangles")
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (
                o3d_mesh.cluster_connected_triangles())
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        cluster_area = np.asarray(cluster_area)

        import copy
        print("Show mesh with small clusters removed")
        mesh_0 = copy.deepcopy(o3d_mesh)
        triangles_to_remove = cluster_n_triangles[triangle_clusters] < 10000
        mesh_0.remove_triangles_by_mask(triangles_to_remove)
        triangles_remove = np.asarray(faces[0].cpu().numpy())[triangles_to_remove]
        
        vertex_colors[triangles_remove.reshape(-1)] = 0.
        # normalizing the visible vertex colors.
        
        # top 95%
        min_value = 0.
        # max_value = 0.3608
        # threshold_value = 0.144
        
        max_value = 0.37385
        threshold_value = 0.112
        
        threshold_value = (threshold_value - min_value) / (max_value - min_value)
        vertex_colors[:,0] = (vertex_colors[:,0] - min_value) / (max_value - min_value)
        vertex_colors[:,1] = (vertex_colors[:,1] - min_value) / (max_value - min_value)
        vertex_colors[:,2] = (vertex_colors[:,2] - min_value) / (max_value - min_value)
        
        vertex_colors[vertex_colors<threshold_value]=0.
        mesh_0 = make_mesh(np.asarray(mesh_0.vertices), np.asarray(mesh_0.triangles), vertex_colors=vertex_colors)
        # draw_plotly([mesh_0])
        o3d_mesh = mesh_0
    
    else:
        o3d_mesh = make_mesh(verts[0].cpu().numpy(), faces[0].cpu().numpy(), vertex_colors=vertex_colors)
        # draw_plotly([o3d_mesh])

    return vertex_colors, o3d_mesh
    


def extract_mesh_dino_data(data_dir, mesh_path, device):
    all_normal_data, all_batch_data, all_opacity_data, all_rgb_data, all_depth_anything, all_omnidata_normals, all_dino_features, all_pca_dino_feats = \
        get_algorithm_data(data_dir)
    
    mesh = load_mesh(mesh_path, device=device)
    # verts = mesh.verts_padded()
    # faces = mesh.faces_padded()
    
    all_rendered_dino_verts = []
    all_rendered_verts_visibility = []
    feat_idx = 0
    print('rendering initial mesh')
    for idx in tqdm.tqdm(range(len(all_batch_data))):
        if idx%2!=0: continue

        batch_data = np.load(all_batch_data[idx], allow_pickle=True).item()
        
        camera_position = batch_data['camera_positions'].cpu().numpy()
        proj_matrix = batch_data['proj_mtx'].cpu().numpy()
        renderer, rasterizer, cameras = create_renderer(camera_position, proj_matrix, device=device, elevation=batch_data['elevation'], azimuth=batch_data['azimuth'], camera_distances=batch_data['camera_distances'])
        rendered_images, rendered_verts_screen, rendered_verts_visibility = render_mesh(renderer, rasterizer, cameras, mesh)

        descriptors = np.load(all_pca_dino_feats[feat_idx])
        descriptors = descriptors.reshape(-1, 4)
        
        # descriptors = np.load(all_dino_features[feat_idx])
        # descriptors = descriptors.reshape(-1, 384)
        print(descriptors.shape, "descriptors")
        
        # descriptors = descriptors / torch.linalg.norm(descriptors, dim=-1)[...,None]
        # descriptors = (descriptors +1.) / 2
        # print(descriptors.shape, descriptors.min(), descriptors.max(), "desc check....")
        descriptors_feat_dim = descriptors.shape[1]

        if not torch.is_tensor(descriptors):
            descriptors = torch.from_numpy(descriptors)
        # print(descriptors.min(), descriptors.max(), " descriptors min max check....")

        
        rendered_verts_screen = (rendered_verts_screen - 256) / 256
        rendered_verts_dino = F.grid_sample(
            descriptors.permute(1, 0)[None].reshape(1, descriptors_feat_dim, 256, 256).to(device).float(), 
            grid=rendered_verts_screen[...,:2][:,None])
        rendered_verts_dino = rendered_verts_dino[0,:,0].permute(1,0)
        # print(rendered_verts_dino.min(), rendered_verts_dino.max(), " rendered_verts_dino min max check....")

        # print(rendered_verts_screen[...,:1].min(), rendered_verts_screen[...,:1].max(), "1 min max")
        # print(rendered_verts_screen[...,1:2].min(), rendered_verts_screen[...,1:2].max(), "2 min max")
        # print(rendered_verts_dino.shape, rendered_verts_screen.shape, rendered_verts_visibility.shape, "rendered_verts_dino.shape, rendered_verts_screen.shape, rendered_verts_visibility.shape check...")
        outside_1 = ((rendered_verts_screen[0,...,0]<-1) |  (rendered_verts_screen[0,...,0]>1))
        outside_2 = ((rendered_verts_screen[0,...,1]<-1) |  (rendered_verts_screen[0,...,1]>1))
        outside = (outside_1 | outside_2)
        rendered_verts_visibility[outside] = 0.
        # print(outside_1.shape, outside_2.shape, outside.shape, outside.min(), outside.max(), "outside shape")
        # print(rendered_verts_visibility[outside].shape)
        # print(rendered_verts_visibility[outside].min(), rendered_verts_visibility[outside].max(), "rendered_verts_visibility[outside].shape, rendered_verts_visibility[outside].min(), rendered_verts_visibility[outside].max()")
        # pritn()

        all_rendered_dino_verts.append(rendered_verts_dino)
        all_rendered_verts_visibility.append(rendered_verts_visibility)

        feat_idx += 1
    cleaned_mesh, normalized_cleaned_dino_verts_std, dino_verts_mean, dino_verts_std, dino_verts_variance = compute_dino_3d_consistency(mesh, all_rendered_dino_verts, all_rendered_verts_visibility, device)
    return mesh, cleaned_mesh, normalized_cleaned_dino_verts_std, dino_verts_mean, dino_verts_std, dino_verts_variance, torch.stack(all_rendered_verts_visibility)
        
    

def compute_dino_3d_consistency(mesh, dino_features_verts_3d, visibility_verts_3d, device):
    print('computing 3d metrics')
    dino_features_verts_3d = torch.stack(dino_features_verts_3d)
    visibility_verts_3d = torch.stack(visibility_verts_3d)
    
    dino_features_verts_3d = dino_features_verts_3d * visibility_verts_3d[...,None].to(dino_features_verts_3d.get_device())
    dino_verts_mean = torch.sum(dino_features_verts_3d, dim=0) / (torch.sum(visibility_verts_3d[...,None], dim=0) + 1e-8)
    # dino_verts_variance = torch.var(dino_features_verts_3d, dim=0) / torch.sum(visibility_verts_3d[...,None], dim=0)

    # print(dino_features_verts_3d.shape, visibility_verts_3d.shape, "dino_features_verts_3d.shape, visibility_verts_3d.shape")
    # print(torch.sum(dino_features_verts_3d, dim=0)[(torch.sum(visibility_verts_3d, dim=0)<=4)].shape, "zero visibility checks...")
    # print((torch.sum(dino_features_verts_3d, dim=0) / (torch.sum(visibility_verts_3d[...,None], dim=0) + 1e-8))[(torch.sum(visibility_verts_3d, dim=0)>4)].shape, "zero visibility checks more data...")
    

    mean_centered = dino_features_verts_3d - (dino_verts_mean[None, ...] * visibility_verts_3d[...,None].to(dino_features_verts_3d.get_device()))
    squared_mean_centered = mean_centered ** 2
    non_zero_sum_squared_diff = torch.sum(squared_mean_centered, dim=0)
    dino_verts_variance = non_zero_sum_squared_diff / (torch.sum(visibility_verts_3d[...,None], dim=0) + 1e-8)
    dino_verts_std = torch.sqrt(dino_verts_variance)

    ## Mean of variance of features
    dino_verts_variance = dino_verts_variance.mean(dim=-1)
    dino_verts_std = dino_verts_std.mean(dim=-1)
    
    print(dino_verts_mean.shape, dino_verts_mean.min(), dino_verts_mean.max(), "dino_verts_mean shape, min, max")
    print(dino_verts_variance.shape, dino_verts_variance.min(), dino_verts_variance.max(), "dino_verts_variance shape, min, max")
    print(dino_verts_std.shape, dino_verts_std.min(), dino_verts_std.max(), "dino_verts_std shape, min, max")
    
    # visualize_3d(mesh, rendered_verts_screen, rendered_verts_visibility)
    # visualize_3d(mesh, vertex_colors=dino_verts_mean[:,:3])
    # visualize_3d(mesh, vertex_colors=dino_verts_variance[:,:3])
    # visualize_3d(mesh, vertex_colors=dino_features_verts_3d[num_views-1, :, 1:4])
    visualize_3d(mesh, vertex_colors=dino_verts_mean[:, 1:4])
    visualize_3d(mesh, vertex_colors=dino_verts_variance[...,None].repeat(1,3), clean_via_connected_components=True)
    normalized_cleaned_dino_verts_std, cleaned_mesh = visualize_3d(mesh, vertex_colors=dino_verts_std[...,None].repeat(1,3), clean_via_connected_components=True)

    return cleaned_mesh, normalized_cleaned_dino_verts_std, dino_verts_mean, dino_verts_std, dino_verts_variance


def render_colored_mesh(data_dir, save_dir, mesh, vertex_data, device):
    all_normal_data, all_batch_data, all_opacity_data, all_rgb_data, all_depth_anything, all_omnidata_normals, all_dino_features, all_pca_dino_feats = \
                    get_algorithm_data(data_dir)
    
    new_texture = TexturesVertex(verts_features=torch.from_numpy(vertex_data[None]).to(device))
    mesh.textures = new_texture

    # print(torch.from_numpy(vertex_data[None]).min(), torch.from_numpy(vertex_data[None]).max(), "vertex_data min max")
    # print(np.unique(vertex_data[None]), "vertex_data min max")

    all_rendered_dino_verts = []
    all_rendered_verts_visibility = []
    feat_idx = 0
    print('rendering variance maps')
    for idx in tqdm.tqdm(range(len(all_batch_data))):
        if idx%2!=0: continue

        batch_data = np.load(all_batch_data[idx], allow_pickle=True).item()
        
        camera_position = batch_data['camera_positions'].cpu().numpy()
        proj_matrix = batch_data['proj_mtx'].cpu().numpy()
        renderer, rasterizer, cameras = create_renderer(camera_position, proj_matrix, device=device, elevation=batch_data['elevation'], azimuth=batch_data['azimuth'], camera_distances=batch_data['camera_distances'])
        rendered_images, rendered_verts_screen, rendered_verts_visibility = render_mesh(renderer, rasterizer, cameras, mesh)
        # print(rendered_images[0, ..., :3].cpu().numpy().min(), rendered_images[0, ..., :3].cpu().numpy().max(), "min max")
        # print(np.unique(rendered_images[0, ..., 0].cpu().numpy()), "unique")
        img_to_save = np.asarray(rendered_images[0, ..., :3].cpu().numpy()*255, dtype=np.uint8)
        np.save(os.path.join(save_dir, 'dino_variance_maps', str(idx).zfill(4) + '.npy'), rendered_images.cpu().numpy())
        Image.fromarray(img_to_save).save(os.path.join(save_dir, 'dino_variance_maps', str(idx).zfill(4) + '.png'))


def main(args):
    args.device = "cuda:0"
    pytorch3d_mesh, cleaned_mesh, normalized_cleaned_dino_verts_std, dino_verts_mean, dino_verts_std, dino_verts_variance, all_rendered_verts_visibility = extract_mesh_dino_data(args.data_path, args.mesh_path, args.device)
    # print(torch.from_numpy(dino_verts_mean[None]).min(), torch.from_numpy(dino_verts_mean[None]).max(), "vertex_data min max")
    # render_colored_mesh(args.data_path, args.save_dir, pytorch3d_mesh, normalized_cleaned_dino_verts_std, args.device)
    # print(normalized_cleaned_dino_verts_std.shape, dino_verts_std.shape, "normalized_cleaned_dino_verts_std.shape, dino_verts_std.shape")
    render_colored_mesh(args.data_path, args.save_dir, pytorch3d_mesh, dino_verts_std.cpu().numpy()[...,None], args.device)
    np.save(os.path.join(args.save_dir, 'all_rendered_verts_visibility.npy'), all_rendered_verts_visibility.cpu().numpy())
    np.save(os.path.join(args.save_dir, 'normalized_cleaned_dino_verts_std.npy'), normalized_cleaned_dino_verts_std)
    np.save(os.path.join(args.save_dir, 'dino_verts_mean.npy'), dino_verts_mean.cpu().numpy())
    np.save(os.path.join(args.save_dir, 'dino_verts_std.npy'), dino_verts_std.cpu().numpy())
    np.save(os.path.join(args.save_dir, 'dino_verts_variance.npy'), dino_verts_variance.cpu().numpy())
    o3d.io.write_triangle_mesh(os.path.join(args.save_dir, 'cleaned_mesh.obj'), cleaned_mesh)

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', required=True, type=str)
    parser.add_argument('--mesh_path', required=True, type=str)
    args = parser.parse_args()
    args.save_dir = os.path.join(args.data_path, "dino_variance_data_latest_v3")
    if os.path.exists(os.path.join(args.data_path, "dino_variance_data_latest_v3")):
        print('deleting dino_variance_data_latest_v3....')
        os.system('rm -r -v ' + os.path.join(args.data_path, "dino_variance_data_latest_v3"))
    if os.path.exists(os.path.join(args.data_path, "dino_variance_data_latest_v3")):
        print('deleting dino_variance_data_latest_v3....')
        os.system('rm -r -v ' + os.path.join(args.data_path, "dino_variance_data_latest_v3"))
    if not os.path.exists(os.path.join(args.save_dir, "dino_variance_maps")):
        os.system('''mkdir -p {}'''.format(os.path.join(args.save_dir.replace("'", "\\'"), "dino_variance_maps")))
    
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)