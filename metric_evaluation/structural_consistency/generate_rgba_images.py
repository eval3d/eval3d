from PIL import Image
import matplotlib.pyplot as plt
import numpy as np, glob, tqdm, os

def get_crop_factor(rgb_image_path, opacity_mask_path):
    rgb_img = np.asarray(Image.open(rgb_image_path))
    opacity_mask_img = np.asarray(Image.open(opacity_mask_path))[...,0:1]
    
    opacity_mask_img = np.asarray((opacity_mask_img>50)*255, dtype=np.uint8)
    # print(np.unique(opacity_mask_img))
    # Find bounding box coordinates
    non_zero_indices = np.nonzero(opacity_mask_img)
    min_y, min_x = np.min(non_zero_indices[0], axis=0), np.min(non_zero_indices[1], axis=0)
    max_y, max_x = np.max(non_zero_indices[0], axis=0), np.max(non_zero_indices[1], axis=0)

    # Crop the RGB image and opacity mask using the bounding box
    cropped_rgb = rgb_img[min_y:max_y, min_x:max_x]
    cropped_opacity_mask = opacity_mask_img[min_y:max_y, min_x:max_x]

    # # Combine the cropped RGB image and opacity mask
    # cropped_rgba_image = np.dstack((cropped_rgb, cropped_opacity_mask))

    # Calculate the dimensions of the cropped image
    height, width = cropped_rgb.shape[:2]

    # Determine padding size for each side
    max_side = max(height, width)
    return max_side

def convert_to_rgba(max_side, rgb_image_path, opacity_mask_path):
    rgb_img = np.asarray(Image.open(rgb_image_path))
    opacity_mask_img = np.asarray(Image.open(opacity_mask_path))[...,0:1]
    # print(rgb_img.shape, opacity_mask_img.shape)
    
    # rgba_image = np.concatenate((rgb_img, opacity_mask_img), axis=-1)
    # Image.fromarray(rgba_image).save(rgb_image_path.replace('.png', '_rgba.png'))

    # print((opacity_mask_img>200).sum())
    opacity_mask_img = np.asarray((opacity_mask_img>50)*255, dtype=np.uint8)
    # print(np.unique(opacity_mask_img))
    # Find bounding box coordinates

    if (opacity_mask_img>200).sum() ==0:
        padded_rgb = rgb_img
        padded_opacity_mask = opacity_mask_img
    else:

        non_zero_indices = np.nonzero(opacity_mask_img)
        min_y, min_x = np.min(non_zero_indices[0], axis=0), np.min(non_zero_indices[1], axis=0)
        max_y, max_x = np.max(non_zero_indices[0], axis=0), np.max(non_zero_indices[1], axis=0)

        # Crop the RGB image and opacity mask using the bounding box
        cropped_rgb = rgb_img[min_y:max_y, min_x:max_x]
        cropped_opacity_mask = opacity_mask_img[min_y:max_y, min_x:max_x]

        # # Combine the cropped RGB image and opacity mask
        # cropped_rgba_image = np.dstack((cropped_rgb, cropped_opacity_mask))

        # Calculate the dimensions of the cropped image
        height, width = cropped_rgb.shape[:2]

        ## Determine padding size for each side
        max_side = max_side + 40
        # max_side = max(height, width) + 40
        pad_height = max_side - height
        pad_width = max_side - width

        # Calculate padding amounts for top, bottom, left, and right
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        # Pad the image with white color
        padded_rgb = np.pad(cropped_rgb, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=255)
        padded_opacity_mask = np.pad(cropped_opacity_mask[...,0], ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)[...,None]

    # Combine the padded RGB image and opacity mask
    padded_rgba_image = np.dstack((padded_rgb, padded_opacity_mask))

    # Save the resized RGBA image
    Image.fromarray(padded_rgba_image).save(rgb_image_path.replace('.png', '_cropped_rgba_video.png'))

    # # Save the cropped RGBA image
    # Image.fromarray(cropped_rgba_image).save(rgb_image_path.replace('.png', '_cropped_rgba.png'))

    
# def main():
#     base_path = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/outputs_uw_149/textmesh-if/*/save/it*-test/rgb_images/*.png"
#     base_path_rgba = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/outputs_uw_149/textmesh-if/*/save/it*-test/rgb_images/*_cropped_rgba.png"
    
#     all_image_paths, all_rgba_paths = glob.glob(base_path), glob.glob(base_path_rgba)
#     print(len(all_rgba_paths))
#     print(len(all_image_paths))
#     for path in all_rgba_paths:
#         os.system('rm -r -v ' + path)

#     all_image_paths, all_rgba_paths = glob.glob(base_path), glob.glob(base_path_rgba)
#     print(len(all_rgba_paths))
#     print(len(all_image_paths))
#     for rgb_img_path in tqdm.tqdm(sorted(all_image_paths)):
#         # print(rgb_img_path)
#         # if 'bichon' not in rgb_img_path: continue
#         if '_rgba' in rgb_img_path: continue
#         try:
#             convert_to_rgba(rgb_img_path, rgb_img_path.replace('rgb_images', 'opacity'))
#         except Exception as e: 
#             print(rgb_img_path, "rgb_img_path")
#             print('passing exception: ', e)
#             pass
#         # break

def main():
    # base_path = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/outputs_uw_149/dreamcraft3d-texture/"
    # base_path = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/objaverse_text_3D/magic3d-refine-sd/"
    # base_path = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/outputs_merged/magic123-refine-sd/"
    # base_path = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/outputs_merged/textmesh-if/"
    # base_path = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/outputs_merged/gs-sds-generation-shading/" # */save/it*-test/rgb_images/*.png"
    # base_path_rgba = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/outputs_merged/textmesh-if/*/save/it*-test/rgb_images/*_cropped_rgba.png"
    # base_path = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/outputs_real_images/mvdream/"
    base_path = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/outputs_images_large/dreamcraft3d-texture"
    # base_path = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/outputs_large/mvdream"

    for folder in tqdm.tqdm(sorted(os.listdir(base_path))):
        # if "Panther" not in folder: continue
                    
        # if '''a_bald_eagle_carved_out_of_wood@20240331-185330''' not in folder:
        #     continue

        # if '''a_beagle_in_a_detective's_outfit@20240331-185330''' not in folder:
        #     continue
        
        # if 'a_dragon-cat_hybrid@20240517-045814' not in folder:
        #     continue
        
        # missing_tags = ['A_dog_creating_sand_art_on_a_beach', 'A_sequence_of_street_lamps,_casting_pools_of_light_on_cobblestone_paths_as_twilight_descends', 'An_assortment_of_solid,_symmetrical,_smooth_marbles,_each_one_a_different_color_with_a_unique_swirl_pattern,_scattered_playfully_across_a_hardwood_floor', 'Several_large,_solid,_symmetrical_hay_bales,_with_a_rough,_golden_texture,_scattered_across_a_rural,_open_field,_with_the_setting_sun_casting_long_shadows', 'a_20-sided_die_made_out_of_glass', 'a_DSLR_photo_of_a_baby_dragon_drinking_boba', 'a_banana_peeling_itself', 'a_beautiful_rainbow_fish', 'a_bichon_frise_wearing_academic_regalia', 'a_blue_motorcycle', 'a_blue_poison-dart_frog_sitting_on_a_water_lily', 'a_brightly_colored_mushroom_growing_on_a_log', 'a_bumblebee_sitting_on_a_pink_flower', 'a_capybara_wearing_a_top_hat,_low_poly', 'a_ceramic_lion', 'a_chimpanzee_with_a_big_grin', 'a_confused_beagle_sitting_at_a_desk_working_on_homework', 'a_corgi_taking_a_selfie', 'a_crab,_low_poly', 'a_crocodile_playing_a_drum_set', 'a_cute_steampunk_elephant', 'a_delicious_hamburger', 'a_team_of_butterflies_playing_soccer_on_a_field']
        # if folder.split('@')[0] not in missing_tags: continue

        folder_path = os.path.join(base_path, folder, "save/it*-test/rgb_images/*.png")

        # folder_path_txt = glob.glob(os.path.join(base_path, folder, "save/it*-test/rgb_images/*.txt"))
        # for txt_file in folder_path_txt:
        #     print(txt_file)
        #     os.system('rm -r -v ' + txt_file)

        folder_images_path = glob.glob(folder_path)
        # flag = 0
        # for rgb_img_path in folder_images_path:
        #     if 'video' in rgb_img_path: 
        #         flag=1
        #         break
        # if flag: continue
        
        video_max_side = -1
        # print('working on: ', folder)
        for rgb_img_path in folder_images_path:
            if "_rgba" in rgb_img_path: continue
        
            try:
                max_side = get_crop_factor(rgb_img_path, rgb_img_path.replace('rgb_images', 'opacity'))
                video_max_side = max(video_max_side, max_side)
                # convert_to_rgba(max_side, rgb_img_path, rgb_img_path.replace('rgb_images', 'opacity'))
            except Exception as e: 
                print(rgb_img_path, "rgb_img_path")
                print('passing exception: ', e)
                pass


        for rgb_img_path in folder_images_path:
            if "_rgba" in rgb_img_path: continue
        
            try:
                convert_to_rgba(video_max_side, rgb_img_path, rgb_img_path.replace('rgb_images', 'opacity'))
            except Exception as e: 
                print(rgb_img_path, "rgb_img_path")
                print('passing exception: ', e)
                pass


if __name__ == "__main__":
    main()