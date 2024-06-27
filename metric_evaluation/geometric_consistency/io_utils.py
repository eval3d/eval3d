import os, glob, csv, subprocess


class args:
    # base_dir = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/outputs_uw_149/"
    # base_dir = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/outputs_merged/"
    # base_dir = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/text_3D_missing_data"
    # base_dir = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/objaverse_text_3D/"
    # base_dir = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/outputs_real_images/"
    # base_dir = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/outputs_large/"
    base_dir = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/outputs_images_large/"

    # base_dir = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/"
    # log_dir = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/logs/" 

    # algorithm_name = "dreamfusion-if"
    # config_name = "dreamfusion-if"

    # algorithm_name = "prolificdreamer-texture"
    # config_name = "prolificdreamer-texture"

    # algorithm_name = "magic3d-refine-sd"
    # config_name = "magic3d-refine-sd"

    # algorithm_name = "textmesh-if"
    # config_name = "textmesh-if"
    
    # algorithm_name = "mvdream"
    # config_name = "mvdream"
    # algorithm_name = "mvdream-sd21-rescale0.5-shading"

    # algorithm_name = "gs-sds-generation-shading"
    # algorithm_name = "gs-sds-generation-shading-no-init"

    # algorithm_name = "magic123-refine-sd"
    algorithm_name = "dreamcraft3d-texture"
    
    num_gpus = 7
    start_idx = 0
    end_idx = 8*7+8*7+8*7+8*7+8*7+8*7+8*7+8*7+8*7+8*7

def read_depth_anything_data(args):

    data = {}
    
    prompt_idx = 0
    for idx, object_id in enumerate(os.listdir(os.path.join(args.base_dir, args.algorithm_name))):
        data_flag=0
        old_flag=0
        
        data_folder_content = sorted(glob.glob(os.path.join(args.base_dir, args.algorithm_name, object_id, 'save/*/rgb_images/*.png')))
        if len(data_folder_content)>0:data_flag=1

        depth_anything_folder = sorted(glob.glob(os.path.join(args.base_dir, args.algorithm_name, object_id, 'save/*/depth_anything/*.png')))
        if len(depth_anything_folder)==120:
            prompt_idx+=1
            continue
                
        if data_flag:
            if data_folder_content[0].split('@')[0] in data.keys(): 
                curr_prompt_idx=prompt_idx
                old_flag=1
                prompt_idx = data[data_folder_content[0].split('@')[0]][0]
            
            # print(data_folder_content[0])
            # if 'a_zoomed_out_DSLR_photo_of_A_punk_rock_squirrel_in_a_studded_leather_jacket_shouting_into_a_microphone_while_standing_on_a_stump_and_holding_a_beer@20240401-021027' not in data_folder_content[0]:
            #     continue
            # if '''a_bald_eagle_carved_out_of_wood@20240331-185330''' not in data_folder_content[0]:
            #     continue
            # if 'a_dragon-cat_hybrid@20240517-045814' not in data_folder_content[0]:
            #     continue
            data[data_folder_content[0].split('@')[0]] = (prompt_idx, "/" + "/".join(data_folder_content[0].split('/')[1:-1]))
            
            print(len(depth_anything_folder), prompt_idx, "/" + "/".join(data_folder_content[0].split('/')[1:-1]))

            if old_flag: prompt_idx = curr_prompt_idx
            else: prompt_idx+=1

    return data


def read_normal_data(args):

    data = {}
    
    prompt_idx = 0
    for idx, object_id in enumerate(os.listdir(os.path.join(args.base_dir, args.algorithm_name))):
        data_flag=0
        old_flag=0
        
        data_folder_content = sorted(glob.glob(os.path.join(args.base_dir, args.algorithm_name, object_id, 'save/*/rgb_images/*.png')))
        if len(data_folder_content)>0:data_flag=1
                
        depth_anything_folder = sorted(glob.glob(os.path.join(args.base_dir, args.algorithm_name, object_id, 'save/*/depth_anything_normal_camera_3/*.png')))
        if len(depth_anything_folder)==60:
            prompt_idx+=1
            continue


        if data_flag:
            if data_folder_content[0].split('@')[0] in data.keys(): 
                curr_prompt_idx=prompt_idx
                old_flag=1
                prompt_idx = data[data_folder_content[0].split('@')[0]][0]
            
            # if '''a_bald_eagle_carved_out_of_wood@20240331-185330''' not in data_folder_content[0]:
            #     continue

            # if '''a_beagle_in_a_detective's_outfit@20240331-185330''' not in data_folder_content[0]:
            #     continue
            
            # if 'a_dragon-cat_hybrid@20240517-045814' not in data_folder_content[0]:
            #     continue
            # if "delicious_hamburger@20240510"  not in data_folder_content[0]: continue
            # if "a_beagle_in_a_detective's_outfit@20240331-184810" not in data_folder_content[0]: continue

            data[data_folder_content[0].split('@')[0]] = (prompt_idx, "/" + "/".join(data_folder_content[0].split('/')[1:-2]))
            
            print(len(depth_anything_folder), prompt_idx, "/" + "/".join(data_folder_content[0].split('/')[1:-2]))

            if old_flag: prompt_idx = curr_prompt_idx
            else: prompt_idx+=1

    return data