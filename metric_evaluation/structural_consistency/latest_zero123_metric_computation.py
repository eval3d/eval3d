import os, numpy as np, torch, glob, tqdm
from PIL import Image
import matplotlib.pyplot as plt
from dreamsim import dreamsim

# algorithm_name = "mvdream-sd21-rescale0.5-shading"
algorithm_name = "mvdream"
# algorithm_name = "dreamfusion-if"
# algorithm_name = "textmesh-if"
# algorithm_name = "magic3d-refine-sd"
# algorithm_name = "prolificdreamer-texture"
# algorithm_name = "gs-sds-generation-shading"
# algorithm_name = "dreamcraft3d-texture"
# algorithm_name = "magic123-refine-sd"

# base_path = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/outputs_real_images/"
# base_path = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/outputs_merged_redo"
# base_path_2 = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/outputs_merged/"
# base_path = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/objaverse_text_3D/"
# save_path = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/metric_analysis/zero123_analysis/"
# save_path = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/metric_analysis/objaverse_zero123_analysis/"
save_path = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/metric_analysis/zero123_analysis_outputs_large/"
# base_path = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/outputs_images_large/"
base_path = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/outputs_large/"
base_path_2 = base_path
# base_path = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/outputs_large/"
num_gpus = 7

def compute_dreamsim_metric(img_0_path, img_ref_path):
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, preprocess = dreamsim(pretrained=True, device=device)

    # Load images
    algo_img = np.asarray(Image.open(img_ref_path))
    mask = (algo_img[...,3:]>0)*1
    algo_img = algo_img[...,:3] * mask + (1-mask)*255
    algo_img = Image.fromarray(np.asarray(algo_img, dtype=np.uint8))
    
    img_ref = preprocess(algo_img).to(device)
    img_0 = preprocess(Image.open(img_0_path)).to(device)

    # print(img_ref.shape, img_0.shape)
    # plt.imshow(img_ref[0].permute(1,2,0).cpu().numpy())
    # plt.show()

    # plt.imshow(img_0[0].permute(1,2,0).cpu().numpy())
    # plt.show()

    # Get distance
    d0 = model(img_ref, img_0)
    return d0.item()



def main():
    queue_images = []
    free_gpus=num_gpus

    print(algorithm_name, base_path, save_path)
    cnt=0
    zero123_algorithm_path = os.path.join(base_path, "mvimg-gen-zero123-sai", algorithm_name)
    algorithm_path = os.path.join(base_path_2, algorithm_name)
    algo_save_dir = os.path.join(save_path, algorithm_name)
    if not os.path.exists(algo_save_dir): os.system('mkdir -p '+ algo_save_dir)
    folder_idx=0
    for folder in tqdm.tqdm(sorted(os.listdir(zero123_algorithm_path))):
        # missing_tags = ['A_dog_creating_sand_art_on_a_beach', 'A_sequence_of_street_lamps,_casting_pools_of_light_on_cobblestone_paths_as_twilight_descends', 'An_assortment_of_solid,_symmetrical,_smooth_marbles,_each_one_a_different_color_with_a_unique_swirl_pattern,_scattered_playfully_across_a_hardwood_floor', 'Several_large,_solid,_symmetrical_hay_bales,_with_a_rough,_golden_texture,_scattered_across_a_rural,_open_field,_with_the_setting_sun_casting_long_shadows', 'a_20-sided_die_made_out_of_glass', 'a_DSLR_photo_of_a_baby_dragon_drinking_boba', 'a_banana_peeling_itself', 'a_beautiful_rainbow_fish', 'a_bichon_frise_wearing_academic_regalia', 'a_blue_motorcycle', 'a_blue_poison-dart_frog_sitting_on_a_water_lily', 'a_brightly_colored_mushroom_growing_on_a_log', 'a_bumblebee_sitting_on_a_pink_flower', 'a_capybara_wearing_a_top_hat,_low_poly', 'a_ceramic_lion', 'a_chimpanzee_with_a_big_grin', 'a_confused_beagle_sitting_at_a_desk_working_on_homework', 'a_corgi_taking_a_selfie', 'a_crab,_low_poly', 'a_crocodile_playing_a_drum_set', 'a_cute_steampunk_elephant', 'a_delicious_hamburger', 'a_team_of_butterflies_playing_soccer_on_a_field']
        # if folder.split('@')[0] not in missing_tags: continue

        # if 'Panther' not in folder: continue
        # if folder_idx>40:
        #     folder_idx+=1 
        #     continue
        
        if folder_idx<320 or folder_idx>360: 
            folder_idx+=1
            continue
        
        
        # if folder_idx<150 and folder_idx>300: 
        #     folder_idx+=1
        #     continue
        folder_idx+=1
        # try:
        zero123_data_path = os.path.join(zero123_algorithm_path, folder.split("@")[0]+"*", "0000_cropped_rgba_video.png*/save/it0-test/rgb_images/")
        zero123_data_path = sorted(glob.glob(zero123_data_path))[-1]
        folder_selected = zero123_data_path.split('/')[-6]
        if os.path.exists(os.path.join(algo_save_dir, folder_selected + '.npy')): 
            cnt+=1
            continue
        print(folder, "folder")
        zero123_data_path = os.path.join(zero123_data_path, "*.png")
        all_zero123_images = sorted(glob.glob(zero123_data_path))
        # print(all_zero123_images)

        zero123_data_path = os.path.join(zero123_algorithm_path, folder.split("@")[0]+"*", "0030_cropped_rgba_video.png*/save/it0-test/rgb_images/")
        zero123_data_path = sorted(glob.glob(zero123_data_path))[-1]

        zero123_data_path = os.path.join(zero123_data_path, "*.png")
        all_zero123_images_90shift = sorted(glob.glob(zero123_data_path))
        # print(all_zero123_images_90shift)

        all_zero123_images_90shift = all_zero123_images_90shift[-2:-1] + all_zero123_images_90shift[0:-2] + all_zero123_images_90shift[-1:]
        if len(all_zero123_images)<5: continue
        if len(all_zero123_images_90shift)<5: continue
        
        algorithm_data_path = os.path.join(algorithm_path, folder.split("@")[0]+"*", "save/it*-test/rgb_images/")
        print(algorithm_data_path, "algorithm_data_path")
        algorithm_data_path = sorted(glob.glob(algorithm_data_path))[-1]
        algorithm_data_path = os.path.join(algorithm_data_path, "*_cropped_rgba_video.png")
        all_algorithm_images = sorted(glob.glob(algorithm_data_path))
        if len(all_zero123_images)<5: continue
        if len(all_algorithm_images)<5: continue
        # print(all_algorithm_images)

        
        vis_image_paths_zero123 = [all_zero123_images[0], all_zero123_images[1], all_zero123_images[2], all_zero123_images[3]] #, all_zero123_images[4]]
        vis_image_paths_zero123_90_shift = [all_zero123_images_90shift[0], all_zero123_images_90shift[1], all_zero123_images_90shift[2], all_zero123_images_90shift[3]] #, all_zero123_images_90shift[4]]
        vis_image_paths_algorithm = [all_algorithm_images[0], all_algorithm_images[30], all_algorithm_images[60], all_algorithm_images[90]] # , all_algorithm_images[119]    

        img_h = 256
        dreamsim_scores = []
        zero123_img_canvas = np.zeros((img_h*3, img_h*len(vis_image_paths_zero123), 3), dtype=np.uint8)
        for img_idx, (zero123_img_file, zero123_90shift_img_file, algo_img_file) in enumerate(zip(vis_image_paths_zero123, vis_image_paths_zero123_90_shift, vis_image_paths_algorithm)):
            
            algo_img = np.asarray(Image.open(algo_img_file))
            mask = (algo_img[...,3:]>0)*1
            algo_img = algo_img[...,:3] * mask + (1-mask)*255
            algo_img = Image.fromarray(np.asarray(algo_img, dtype=np.uint8)).resize((256,256))

            zero123_img = np.asarray(Image.open(zero123_img_file))
            zero123_img_90shift = np.asarray(Image.open(zero123_90shift_img_file))

            zero123_img_canvas[0*img_h:1*img_h, img_h*img_idx:img_h*(img_idx+1)] = zero123_img
            zero123_img_canvas[1*img_h:2*img_h, img_h*img_idx:img_h*(img_idx+1)] = zero123_img_90shift
            zero123_img_canvas[2*img_h:3*img_h, img_h*img_idx:img_h*(img_idx+1)] = algo_img

            score_1 = compute_dreamsim_metric(img_ref_path=algo_img_file, img_0_path=zero123_img_file)
            score_2 = compute_dreamsim_metric(img_ref_path=algo_img_file, img_0_path=zero123_90shift_img_file)
            dreamsim_scores.append((score_1, score_2))

        
        data_dict = {
            'vis_image_paths_zero123': vis_image_paths_zero123,
            'vis_image_paths_zero123_90_shift': vis_image_paths_zero123_90_shift,
            'vis_image_paths_algorithm': vis_image_paths_algorithm,
            'dreamsim_scores': dreamsim_scores
        }

        save_dir = os.path.join(algo_save_dir, folder_selected + '.npy')
        print(folder, dreamsim_scores, save_dir, "folder: dreamsim_scores: save_dir")
        # if os.path.exists(save_dir):
        #     assert(False)
        # np.save(save_dir, dreamsim_scores)
        np.save(save_dir, data_dict)
        plt.imsave(save_dir.replace('.npy', '.png'), zero123_img_canvas)
        cnt+=1
        # except:
        #     print('{} failed'.format(folder))

    print(cnt)
    print('done......')


if __name__ == "__main__":
    main()