import os, glob, numpy as np, cv2, imageio, argparse
import matplotlib.pyplot as plt
from PIL import Image


def save_img_sequence(filename, imgs, save_format="mp4", fps=2):
    assert save_format in ["gif", "mp4"]
    if not filename.endswith(save_format):
        filename += f".{save_format}"
    save_path = filename

    if save_format == "gif":
        imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in imgs]
        imageio.mimsave(save_path, imgs, fps=fps, palettesize=256)
    elif save_format == "mp4":
        imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in imgs]
        imageio.mimsave(save_path, imgs, fps=fps)
    return save_path



def analyse_dino_metric(data_dir):
    rgb_images = sorted(glob.glob(os.path.join(data_dir, "rgb_images", "*.png")))
    pca_dino_feats = sorted(glob.glob(os.path.join(data_dir, "all_pca_dino_feats", "*.png")))
    dino_variance_paths = sorted(glob.glob(os.path.join(data_dir, "dino_variance_data_latest", "dino_variance_maps", "*.png")))
    sel_rgb_images = []
    for img in rgb_images:
        if 'rgba' in img: continue
        sel_rgb_images.append(img)
    rgb_images = sel_rgb_images
    print(len(rgb_images), len(pca_dino_feats), len(dino_variance_paths))
    print(rgb_images[0])
    assert((len(rgb_images) == len(pca_dino_feats)) or (len(dino_variance_paths) == len(pca_dino_feats)))

    if len(rgb_images) == len(pca_dino_feats):
        pca_dino_feats = pca_dino_feats[::2] 
    rgb_images = rgb_images[::2]
    

    imgs = []
    dino_idx=0
    for idx in range(len(rgb_images)):
        if idx%2==0:
            vis = np.zeros((512, 512*3, 3), dtype=np.uint8)
            rgb_image = np.asarray(Image.open(rgb_images[idx]))
            pca_dino = np.array(Image.open(pca_dino_feats[idx]))
            pca_dino = cv2.resize(pca_dino, (512, 512))
            rgb_image = cv2.resize(rgb_image, (512, 512))
            dino_variance = np.asarray(Image.open(dino_variance_paths[idx]))
            vis[:,:512] = rgb_image
            vis[:,512:1024] = np.asarray(pca_dino, dtype=np.uint8)
            vis[:,1024:1024+512] = dino_variance
            # print(vis.shape)
            imgs.append(vis)
            dino_idx+=1

    save_path = save_img_sequence(
        filename = os.path.join(data_dir, "dino_variance_data_latest", "variance_metric_analysis"),
        imgs = imgs,
    )
    print("video saved at: {}".format(save_path))


def main(args):
    analyse_dino_metric(args.data_path)
    print("Dino metric analysis extraction complete for data_path: {}".format(args.data_path))

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', required=True, type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)