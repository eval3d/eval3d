import os, glob, csv, subprocess


class args:
    # base_dir = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/outputs_uw_149/"
    # base_dir = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/outputs_merged/"
    # base_dir = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/objaverse_text_3D/"
    base_dir = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/outputs_real_images/"
    # base_dir = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/outputs_images_large/"
    # base_dir = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/outputs_large/"
    
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

    # config_name = "custom/threestudio-mvdream/configs/mvdream-sd21-shading.yaml"
    # # algorithm_name = "mvdream-sd21-rescale0.5-shading"
    # algorithm_name = "mvdream"

    # algorithm_name = "gs-sds-generation-shading-no-init"
    # config_name = "gs-sds-generation-shading-no-init"

    # algorithm_name = "gs-sds-generation-shading"
    # algorithm_name = "magic123-refine-sd"
    algorithm_name = "dreamcraft3d-texture"

    num_gpus = 8
    start_idx = 0 #8*7+8*7
    end_idx = 8*7+8*7 #+8*7+8*7+8*7 #+8*7+8*7+8*7+8*7
    


# def read_data(args):
#     log_file = os.path.join(args.log_dir, "threestudio_logs_{}.csv".format(args.algorithm_name))
#     with open(log_file, 'r') as file:
#         csv_reader = csv.reader(file)
#         data = {}
#         mesh_data = {}
#         for idx, row in enumerate(csv_reader):
#             if idx==0: continue
#             if args.algorithm_name=="magic3d-refine-sd" or args.algorithm_name=="textmesh-if":
#                 folder_content = sorted(glob.glob(os.path.join(args.base_dir, row[2].replace("outputs", "outputs_cleaned").split('@')[0] + "@*", 'save/*/batch_data/*.npy')))
#             else:
#                 folder_content = sorted(glob.glob(os.path.join(args.base_dir, row[2], 'save/*/batch_data/*.npy')))
#             if len(folder_content)>0:
#                 # print(row[0], folder_content[0])
#                 data[folder_content[0].split('@')[0]] = (row[0], "/" + "/".join(folder_content[0].split('/')[1:-2]))

#             folder_content = sorted(glob.glob(os.path.join(args.base_dir, row[2].replace("outputs", "outputs_cleaned").split('@')[0] + "@*", 'save/*/model.obj')))
#             if len(folder_content)>0:
#                 # print(row[0], folder_content[0])
#                 if args.algorithm_name=="magic3d-refine-sd" or args.algorithm_name=="textmesh-if":
#                     mesh_data[folder_content[0].split('@')[0]] = (row[0], folder_content[0])
#                 else:
#                     mesh_data[folder_content[0].split('@')[0].replace("outputs_cleaned", "outputs")] = (row[0], folder_content[0])

#     return data, mesh_data


def read_dino_variance_extraction_data(args):
    data = {}
    mesh_data = {}

    prompt_idx = 0
    for idx, object_id in enumerate(os.listdir(os.path.join(args.base_dir, args.algorithm_name))):
        data_flag=0
        mesh_flag=0
        old_flag=0
        
        data_folder_content = sorted(glob.glob(os.path.join(args.base_dir, args.algorithm_name, object_id.split('@')[0] + '*', 'save/*/all_dino_feats/*.npy')))
        if len(data_folder_content)>0:data_flag=1

        dino_variance_content = sorted(glob.glob(os.path.join(args.base_dir, args.algorithm_name, object_id.split('@')[0] + '*', 'save/*/dino_variance_data_latest_v3/dino_variance_maps/*.npy'))) 
        if len(dino_variance_content)==60:
            prompt_idx+=1
            continue
            
        mesh_folder_content = sorted(glob.glob(os.path.join(args.base_dir, args.algorithm_name, object_id.split('@')[0] + '*', 'save/*/model.obj')))
        if len(mesh_folder_content)>0: mesh_flag=1

        # print(object_id, len(data_folder_content), len(mesh_folder_content))
        
        
        if data_flag and mesh_flag:
            if data_folder_content[0].split('@')[0] in data.keys(): 
                curr_prompt_idx=prompt_idx
                old_flag=1
                prompt_idx = data[data_folder_content[0].split('@')[0]][0]

            # if "squirrel" in data_folder_content[0]:
            #     print(data_folder_content[0])

            # if 'An_ancient,_weathered_statue,_now_covered_in_a_blanket_of_moss@20240413-103353' not in data_folder_content[0]: continue
            # if "A_pen_leaking_blue_ink@20240413-024104" not in data_folder_content[0]: continue
            # if "a_zoomed_out_DSLR_photo_of_a_baby_bunny_sitting_on_top_of_a_stack_of_pancakes@20240414-022055" not in data_folder_content[0]: continue

            # if 'a_zoomed_out_DSLR_photo_of_A_punk_rock_squirrel_in_a_studded_leather_jacket_shouting_into_a_microphone_while_standing_on_a_stump_and_holding_a_beer@20240401-021027' not in data_folder_content[0]: continue
            # if 'a_DSLR_photo_of_a_plate_of_fried_chicken_and_waffles_with_maple_syrup_on_them@' not in data_folder_content[0]: continue
            # if 'amigurumi_motorcycle' not in data_folder_content[0]: continue
            # if 'a_DSLR_photo_of_a_plate_of_fried_chicken_and_waffles_with_maple_syrup_on_them@20240413-133247' not in data_folder_content[0] and 'a_zoomed_out_DSLR_photo_of_a_model_of_a_house_in_Tudor_style@20240413-212408' not in data_folder_content[0] and 'An_assortment_of_vintage,_fragrant_perfumes_on_display@20240413-015907' not in data_folder_content[0]:
            #     continue
    
            # if 'beagle' in data_folder_content[0]:
            #     print(data_folder_content[0].split('/')[-5])
            # missing_folders = ["a_beagle_in_a_detective's_outfit@20240510-004320", 'a_nest_with_a_few_white_eggs_and_one_golden_egg@20240510-005156', 'an_English_castle,_aerial_view@20240510-005156']
            # if data_folder_content[0].split('/')[-5] not in missing_folders: continue

            data[data_folder_content[0].split('@')[0]] = (prompt_idx, "/" + "/".join(data_folder_content[0].split('/')[1:-2]))
            mesh_data[mesh_folder_content[0].split('@')[0].replace("outputs_cleaned", "outputs")] = (prompt_idx, mesh_folder_content[0])
            
            print(len(dino_variance_content), prompt_idx, "/" + "/".join(data_folder_content[0].split('/')[1:-2]), mesh_folder_content[0])

            if old_flag: prompt_idx = curr_prompt_idx
            else: prompt_idx+=1

    return data, mesh_data




def read_dino_metric_analysis_data(args):
    data = {}
    mesh_data = {}

    prompt_idx = 0
    for idx, object_id in enumerate(os.listdir(os.path.join(args.base_dir, args.algorithm_name))):
        data_flag=0
        mesh_flag=0
        old_flag=0
        
        data_folder_content = sorted(glob.glob(os.path.join(args.base_dir, args.algorithm_name, object_id, 'save/*/all_dino_feats/*.npy')))
        if len(data_folder_content)>0:data_flag=1
            
        # mesh_folder_content = sorted(glob.glob(os.path.join(args.base_dir, args.algorithm_name, object_id, 'save/*/model.obj')))
        # if len(mesh_folder_content)>0: mesh_flag=1
        
        
        if data_flag:
            # if 'a_DSLR_photo_of_a_plate_of_fried_chicken_and_waffles_with_maple_syrup_on_them@20240413-133247' not in data_folder_content[0] and 'a_zoomed_out_DSLR_photo_of_a_model_of_a_house_in_Tudor_style@20240413-212408' not in data_folder_content[0] and 'An_assortment_of_vintage,_fragrant_perfumes_on_display@20240413-015907' not in data_folder_content[0]:
            #     continue

            # if 'a_DSLR_photo_of_a_plate_of_fried_chicken_and_waffles_with_maple_syrup_on_them@' not in data_folder_content[0]: continue
            # if 'a_zoomed_out_DSLR_photo_of_A_punk_rock_squirrel_in_a_studded_leather_jacket_shouting_into_a_microphone_while_standing_on_a_stump_and_holding_a_beer@20240401-021027' not in data_folder_content[0]: continue
            

            if data_folder_content[0].split('@')[0] in data.keys(): 
                curr_prompt_idx=prompt_idx
                old_flag=1
                prompt_idx = data[data_folder_content[0].split('@')[0]][0]
            
            data[data_folder_content[0].split('@')[0]] = (prompt_idx, "/" + "/".join(data_folder_content[0].split('/')[1:-2]))
            # mesh_data[mesh_folder_content[0].split('@')[0].replace("outputs_cleaned", "outputs")] = (prompt_idx, mesh_folder_content[0])
            
            print(prompt_idx, "/" + "/".join(data_folder_content[0].split('/')[1:-2]))

            if old_flag: prompt_idx = curr_prompt_idx
            else: prompt_idx+=1

    return data

