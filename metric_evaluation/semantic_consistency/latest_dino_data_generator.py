import os, glob, csv, subprocess


class args:
    base_dir = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/"
    log_dir = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/logs/"
    base_dir = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/outputs_merged/" 
    base_dir = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/outputs_images_large/"

    # algorithm_name = "dreamfusion-if"
    # config_name = "dreamfusion-if"

    # algorithm_name = "prolificdreamer-texture"
    # config_name = "prolificdreamer-texture"

    # algorithm_name = "magic3d-refine-sd"
    # config_name = "magic3d-refine-sd"

    # algorithm_name = "textmesh-if"
    # config_name = "textmesh-if"

    # config_name = "latentnerf-refine"
    # algorithm_name = "latentnerf-refine-old-with-ckpts"

    # config_name = "custom/threestudio-mvdream/configs/mvdream-sd21-shading.yaml"
    # algorithm_name = "mvdream-sd21-rescale0.5-shading"

    # algorithm_name = "gs-sds-generation-shading"
    # config_name = "gs-sds-generation-shading"

    algorithm_name = "dreamcraft3d-texture"
    
    # num_gpus = 8
    start_idx = 0
    end_idx = 200
    # all_gpu_ids = [0,1,2,3,4,5,6]
    # all_gpu_ids = [0,1,2,3,5,6]
    # all_gpu_ids = [0,1,2,3,4,5,6]


def read_data(args):
    log_file = os.path.join(args.log_dir, "threestudio_logs_{}.csv".format(args.algorithm_name))
    with open(log_file, 'r') as file:
        csv_reader = csv.reader(file)
        data = {}
        for idx, row in enumerate(csv_reader):
            if idx==0: continue
            if args.algorithm_name=="magic3d-refine-sd" or args.algorithm_name=="textmesh-if":
                folder_content = sorted(glob.glob(os.path.join(args.base_dir, row[2].replace("outputs", "outputs_cleaned").split('@')[0] + "@*", 'save/*/batch_data/*.npy')))
            else:
                folder_content = sorted(glob.glob(os.path.join(args.base_dir, row[2], 'save/*/batch_data/*.npy')))
            if len(folder_content)>0:
                print(row[0], "/" + "/".join(folder_content[0].split('/')[1:-2]))
                data[folder_content[0].split('@')[0]] = (row[0], "/" + "/".join(folder_content[0].split('/')[1:-2]))
    return data


def read_data_big(args):
    algorithm_dir = os.path.join(args.base_dir, args.algorithm_name)
    data = {}
    folder_idx=0
    for folder in sorted(os.listdir(algorithm_dir)):
        folder_content = sorted(glob.glob(os.path.join(algorithm_dir, folder, 'save/*/batch_data/*.npy')))
        if len(folder_content)>0:
            print(folder_idx, "/" + "/".join(folder_content[0].split('/')[1:-2]))
            data[folder_content[0].split('@')[0]] = (folder_idx, "/" + "/".join(folder_content[0].split('/')[1:-2]))
            folder_idx+=1
    return data



algorithm_data = read_data_big(args)
# algorithm_data = read_data(args)
# algorithm_data = {
#     '/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/outputs_merged/magic3d-refine-sd/a_zoomed_out_DSLR_photo_of_a_model_of_a_house_in_Tudor_style@20240413-212408': ('0', "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/outputs_merged/magic3d-refine-sd/a_zoomed_out_DSLR_photo_of_a_model_of_a_house_in_Tudor_style@20240413-212408/save/it5000-test"),
#     '/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/outputs_merged/magic3d-refine-sd/An_assortment_of_vintage,_fragrant_perfumes_on_display@20240413-015907': ('1', '/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/outputs_merged/magic3d-refine-sd/An_assortment_of_vintage,_fragrant_perfumes_on_display@20240413-015907/save/it5000-test')
# }
# algorithm_data = {
#     '/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/outputs_merged/gs-sds-generation-shading/a_nest_with_a_few_white_eggs_and_one_golden_egg@20240510-005156': ('0', "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/outputs_merged/gs-sds-generation-shading/a_nest_with_a_few_white_eggs_and_one_golden_egg@20240510-005156/save/it5000-test"),
#     "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/outputs_merged/gs-sds-generation-shading/a_beagle_in_a_detective's_outfit@20240510-004320": ('1', "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/outputs_merged/gs-sds-generation-shading/a_beagle_in_a_detective's_outfit@20240510-004320/save/it5000-test"),
#     '/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/outputs_merged/gs-sds-generation-shading/an_English_castle,_aerial_view@20240510-005156': ('2', "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/outputs_merged/gs-sds-generation-shading/an_English_castle,_aerial_view@20240510-005156/save/it5000-test")
# }

gpu_id = 0
tmux_id=1
queue_images = []
data_idx = 0
num_gpus=8
free_gpus=num_gpus
# avail_gpus = [1,2,3,5]
avail_gpus = [0,1,2,3,4,5,6,7]
# avail_gpus = [0,1,2,3,4,5,6]
# avail_gpus = [5,7]
# avail_gpus = [7]
while data_idx < len(algorithm_data.keys()):
    if free_gpus<num_gpus:
        removal_idx = []
        for img_to_check_idx, (img_to_check_tmux, img_to_check) in enumerate(queue_images):
            if os.path.exists(img_to_check):
                print(img_to_check, " exists at tmux: ", img_to_check_tmux)
                os.system('tmux kill-session -t ' + str(img_to_check_tmux))
                removal_idx.append(img_to_check_idx)

        for img_to_check_idx in removal_idx:
            _ = queue_images.pop(img_to_check_idx)
            break

        if len(queue_images)==0: 
            free_gpus=num_gpus
            print('running more.....')
        else: 
            continue

    data_key = list(algorithm_data.keys())[data_idx]
    datum = algorithm_data[data_key]
    
    prompt_idx = int(datum[0])    
    data_path = datum[1].replace("'", "\\'")
    dino_extracted_file = os.path.join(datum[1], "all_dino_feats", "dino_extracted.txt")

    if prompt_idx<args.start_idx or prompt_idx>=args.end_idx: 
        data_idx+=1
        # gpu_id+=1
        tmux_id+=1
        continue
    if os.path.exists(dino_extracted_file):
        print(data_idx, dino_extracted_file, " exists.")
        data_idx+=1
        # gpu_id+=1
        tmux_id+=1
        continue


    print('DINO | GPU: {} | PROMPT ID: {} | DATA_PATH: {}'.format(avail_gpus[gpu_id%num_gpus], prompt_idx, data_path))
    command = '''CUDA_VISIBLE_DEVICES={} python dino_metric/latest_dino_data_generator_main.py --data_path {}'''.format(avail_gpus[gpu_id%num_gpus], data_path)
    command = command + "; sleep infinity"
    subprocess.Popen(["tmux", "new-session", "-d", "-s", str(tmux_id), command])

    queue_images.append((tmux_id, dino_extracted_file))
    print(prompt_idx, len(queue_images))
    data_idx+=1
    gpu_id+=1
    tmux_id+=1
    if len(queue_images)==num_gpus:
        free_gpus=0

    if len(queue_images)==20: 
        pritn()
        break