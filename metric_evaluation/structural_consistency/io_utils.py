import os, glob, csv, subprocess


class args:
    # base_dir = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/outputs_uw_149"
    # base_dir = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/outputs_merged"
    # base_dir = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/text_3D_missing_data"
    # base_dir = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/objaverse_text_3D/"
    # base_dir = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/outputs_real_images/"
    # base_dir = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/outputs_images_large/"
    base_dir = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/outputs_large/"

    # base_dir = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/"
    # log_dir = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/logs/" 

    # algorithm_name = "dreamfusion-if"
    # # display_algorithm_name = "mvimg-gen-zero123-sai--dreamfusion-if"
    # config_name = "dreamfusion-if"

    # algorithm_name = "prolificdreamer-texture"
    # # display_algorithm_name = "mvimg-gen-zero123-sai--prolificdreamer-texture"
    # config_name = "prolificdreamer-texture"

    # algorithm_name = "magic3d-refine-sd"
    # # display_algorithm_name = "mvimg-gen-zero123-sai--magic3d-refine-sd"
    # config_name = "magic3d-refine-sd"

    # algorithm_name = "textmesh-if"
    # # display_algorithm_name = "mvimg-gen-zero123-sai--textmesh-if"
    # config_name = "textmesh-if"

    # config_name = "custom/threestudio-mvdream/configs/mvdream-sd21-shading.yaml"
    # # display_algorithm_name = "mvimg-gen-zero123-sai--mvdream-sd21-rescale0.5-shading"
    # # algorithm_name = "mvdream-sd21-rescale0.5-shading"
    algorithm_name = "mvdream"

    # algorithm_name = "gs-sds-generation-shading"

    # algorithm_name = "magic123-refine-sd"
    # algorithm_name = "dreamcraft3d-texture"

    
    num_gpus = 8
    start_idx = 0
    end_idx = 1000
    # all_gpu_ids = [0,1,2,3,4,5,6]
    all_gpu_ids = [0,1,2,3,4,5,6,7]
    # all_gpu_ids = [0,1,2,3,4,5,6,7]
    

def read_zero123_image_generator_data(args):
    log_file = os.path.join(args.log_dir, "threestudio_logs_{}.csv".format(args.algorithm_name))
    with open(log_file, 'r') as file:
        csv_reader = csv.reader(file)
        data = {}
        for idx, row in enumerate(csv_reader):
            if idx==0: continue
            if args.algorithm_name=="magic3d-refine-sd" or args.algorithm_name=="textmesh-if":
                folder_content = sorted(glob.glob(os.path.join(args.base_dir, row[2].replace("outputs", "outputs_cleaned").split('@')[0] + "@*", 'save/*/rgb_images/*_cropped_rgba.png')))
            else:
                folder_content = sorted(glob.glob(os.path.join(args.base_dir, row[2], 'save/*/rgb_images/*_cropped_rgba.png')))
            if len(folder_content)>0:
                folder_content_id=0
                desired_ids = [0, 30]
                # print(row[0], folder_content[0], folder_content[0].split('/')[-5])
                for d_id in desired_ids:
                    data[folder_content[folder_content_id].split('@')[0] + '-{}'.format(str(d_id))] = (row[0], folder_content[folder_content_id].split('/')[-5], folder_content[folder_content_id].replace('0000_cropped_rgba.png', '{}_cropped_rgba.png'.format(str(d_id).zfill(4))))
    return data


# def read_zero123_image_generator_data_inverse_round(args):
#     data = {}
#     base_path = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/outputs/mvimg-gen-zero123-sai/{}/*/*/save/it0-test/rgb_images/0001.png".format(args.algorithm_name)
#     all_images = sorted(glob.glob(base_path))

#     for idx, img in enumerate(all_images):
#         object_name = "--".join([img.split('/')[-6], img.split('/')[-5]])
#         print(idx, object_name, img)
#         data[idx] = (idx, object_name, img)

#     return data



def read_zero123_image_generator_big(args):

    data = {}
    
    prompt_idx = 0
    for idx, object_id in enumerate(os.listdir(os.path.join(args.base_dir, args.algorithm_name))):
        data_flag=0
        old_flag=0
        
        data_folder_content = sorted(glob.glob(os.path.join(args.base_dir, args.algorithm_name, object_id, 'save/*/rgb_images/*_cropped_rgba_video.png')))
        if len(data_folder_content)>0:data_flag=1
            
            
        if data_flag:
            if data_folder_content[0].split('@')[0] in data.keys(): 
                curr_prompt_idx=prompt_idx
                old_flag=1
                prompt_idx = data[data_folder_content[0].split('@')[0]][0]
                assert(False)
            
            # data[data_folder_content[0].split('@')[0]] = (prompt_idx, "/" + "/".join(data_folder_content[0].split('/')[1:-2]))
            folder_content_id=0
            desired_ids = [0, 30]
            # print(row[0], folder_content[0], folder_content[0].split('/')[-5])
            folder_content = data_folder_content
            for d_id in desired_ids:
                # if 'Panther' not in folder_content[folder_content_id]: continue
                            
                # if '''a_bald_eagle_carved_out_of_wood@20240331-185330''' not in data_folder_content[0]:
                #     continue

                # if '''a_beagle_in_a_detective's_outfit@20240331-185330''' not in data_folder_content[0]:
                #     continue
                
                # if 'a_dragon-cat_hybrid@20240517-045814' not in data_folder_content[0]:
                #     continue
                
                # missing_tags = ['A_dog_creating_sand_art_on_a_beach', 'A_sequence_of_street_lamps,_casting_pools_of_light_on_cobblestone_paths_as_twilight_descends', 'An_assortment_of_solid,_symmetrical,_smooth_marbles,_each_one_a_different_color_with_a_unique_swirl_pattern,_scattered_playfully_across_a_hardwood_floor', 'Several_large,_solid,_symmetrical_hay_bales,_with_a_rough,_golden_texture,_scattered_across_a_rural,_open_field,_with_the_setting_sun_casting_long_shadows', 'a_20-sided_die_made_out_of_glass', 'a_DSLR_photo_of_a_baby_dragon_drinking_boba', 'a_banana_peeling_itself', 'a_beautiful_rainbow_fish', 'a_bichon_frise_wearing_academic_regalia', 'a_blue_motorcycle', 'a_blue_poison-dart_frog_sitting_on_a_water_lily', 'a_brightly_colored_mushroom_growing_on_a_log', 'a_bumblebee_sitting_on_a_pink_flower', 'a_capybara_wearing_a_top_hat,_low_poly', 'a_ceramic_lion', 'a_chimpanzee_with_a_big_grin', 'a_confused_beagle_sitting_at_a_desk_working_on_homework', 'a_corgi_taking_a_selfie', 'a_crab,_low_poly', 'a_crocodile_playing_a_drum_set', 'a_cute_steampunk_elephant', 'a_delicious_hamburger', 'a_team_of_butterflies_playing_soccer_on_a_field']
                # # print(folder_content[folder_content_id].split('@')[0], "check")
                # if folder_content[folder_content_id].split('@')[0].split('/')[-1] not in missing_tags: continue

                data[folder_content[folder_content_id].split('@')[0] + '-{}'.format(str(d_id))] = (prompt_idx, folder_content[folder_content_id].split('/')[-5], folder_content[folder_content_id].replace('0000_cropped_rgba_video.png', '{}_cropped_rgba_video.png'.format(str(d_id).zfill(4))))
                print((prompt_idx, folder_content[folder_content_id].split('/')[-5], folder_content[folder_content_id].replace('0000_cropped_rgba_video.png', '{}_cropped_rgba_video.png'.format(str(d_id).zfill(4)))))

            # print(prompt_idx, "/" + "/".join(data_folder_content[0].split('/')[1:-2]), mesh_folder_content[0])

            if old_flag: prompt_idx = curr_prompt_idx
            else: prompt_idx+=1

    return data