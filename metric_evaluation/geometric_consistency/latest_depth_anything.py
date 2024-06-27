from io_utils import args, read_depth_anything_data
import os, glob, csv, subprocess

algorithm_data = read_depth_anything_data(args)

available_gpus = [1,2,3,5,6,4,7]
for data_key, datum in algorithm_data.items():
    prompt_idx = int(datum[0])
    if not (prompt_idx >= args.start_idx and prompt_idx < args.end_idx): continue
    data_path = datum[1].replace("'", "\\'")
    # if prompt_idx!=3: continue
    
    gpu_id = available_gpus[prompt_idx % args.num_gpus]
    print('DEPTH-ANYTHING | GPU: {} | PROMPT ID: {} | DATA_PATH: {}'.format(gpu_id, prompt_idx, data_path))

    command = '''cd Depth-Anything/; CUDA_VISIBLE_DEVICES={} python run.py --encoder vitl --img-path {}  --outdir {}'''.format(gpu_id, data_path, data_path.replace('rgb_images', 'depth_anything'))
    command = command + "; sleep infinity"
    subprocess.Popen(["tmux", "new-session", "-d", "-s", str(3+prompt_idx), command])