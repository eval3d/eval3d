import os, glob, csv, subprocess
from io_utils import read_zero123_image_generator_data, args, read_zero123_image_generator_big



# algorithm_data = read_zero123_image_generator_data(args)

# # idx=0
# tmux_id=0
# curr_gpu_id=0
# for idx, (_, datum) in enumerate(algorithm_data.items()):
#     object_name = datum[1]
#     image_path = datum[2]
#     prompt_idx = int(datum[0])
#     if not (prompt_idx >= args.start_idx and prompt_idx < args.end_idx): continue
    
#     # if not 'motorcycle' in image_path: continue
#     print(idx, image_path)

#     gpu_id = curr_gpu_id % args.num_gpus
#     gpu_id = args.all_gpu_ids[gpu_id]
#     print('ZERO123 | GPU: {} | PROMPT ID: {} | OBJ-NAME: {} | IMG-PATH: {}'.format(gpu_id, prompt_idx, object_name, image_path))

#     # command = '''CUDA_VISIBLE_DEVICES={} python latest_dino_data_generator_main.py --data_path {}'''.format(gpu_id, data_path)
#     # gpu_id=7
#     command = '''cd threestudio; CUDA_VISIBLE_DEVICES={} python launch_original.py --config custom/threestudio-mvimg-gen/configs/stable-zero123.yaml --train --gpu 0 data.image_path={} object_name={} algorithm_name={}'''.format(gpu_id, image_path, object_name, args.algorithm_name)
#     command = command + "; sleep infinity"
#     # prompt_idx = prompt_idx + gpu_id
#     subprocess.Popen(["tmux", "new-session", "-d", "-s", str(tmux_id), command])
#     tmux_id+=1
#     curr_gpu_id+=1
#     # break



algorithm_data = read_zero123_image_generator_big(args)

# idx=0
tmux_id=1
curr_gpu_id=0
# for idx, (_, datum) in enumerate(algorithm_data.items()):
idx = 0
queue_images = []
free_gpus=args.num_gpus
while idx < len(list(algorithm_data.keys())):
    if free_gpus<args.num_gpus:
        removal_idx = []
        for img_to_check_idx, img_to_check in enumerate(queue_images):
            if os.path.exists(img_to_check.replace('.png', '.txt')):
                print(img_to_check.replace('.png', '.txt'), " exists")
                # os.system('rm -r -v ' + img_to_check.replace('.png', '.txt'))
                removal_idx.append(img_to_check_idx)

        for img_to_check_idx in removal_idx:
            _ = queue_images.pop(img_to_check_idx)
            break

        if len(queue_images)==0: 
            free_gpus=args.num_gpus
            print('running more.....')
        else: 
            continue
    
    try:
        print(idx, list(algorithm_data.keys())[idx])
        datum_key = list(algorithm_data.keys())[idx]
        datum = algorithm_data[datum_key]
        object_name = datum[1]
        image_path = datum[2]
        if os.path.exists(image_path.replace('.png', '.txt')): 
            os.system('rm -r -v ' + image_path.replace('.png', '.txt'))
            # print('exists')
            # idx+=1
            # continue
        prompt_idx = int(datum[0])
        # if not (prompt_idx >= args.start_idx and prompt_idx < args.end_idx): continue
        
        # if not 'beautiful_dress' in image_path: continue
        # print(idx, image_path)

        gpu_id = curr_gpu_id % args.num_gpus
        gpu_id = args.all_gpu_ids[gpu_id]
        print('ZERO123 | GPU: {} | PROMPT ID: {} | OBJ-NAME: {} | IMG-PATH: {}'.format(gpu_id, prompt_idx, object_name, image_path))

        # command = '''CUDA_VISIBLE_DEVICES={} python latest_dino_data_generator_main.py --data_path {}'''.format(gpu_id, data_path)
        # gpu_id=7
        command = '''cd threestudio; CUDA_VISIBLE_DEVICES={} python launch_original.py --config custom/threestudio-mvimg-gen/configs/stable-zero123.yaml --train --gpu 0 data.image_path="{}" object_name="{}" algorithm_name={}'''.format(gpu_id, image_path, object_name, args.algorithm_name)
        command = command + "; sleep infinity"
        # prompt_idx = prompt_idx + gpu_id
        subprocess.Popen(["tmux", "new-session", "-d", "-s", str(tmux_id), command])
        tmux_id+=1
        curr_gpu_id+=1
        # break
        # if os.path.exists(image_path.replace('.png', '.txt')): os.system('rm -v ' + image_path.replace('.png', '.txt'))
        queue_images.append(image_path)
        print(idx, len(queue_images))
        idx += 1
        if len(queue_images)==args.num_gpus:
            free_gpus=0
        
    except Exception as e:
        print('exception: ', e)
        pass

print("done....")