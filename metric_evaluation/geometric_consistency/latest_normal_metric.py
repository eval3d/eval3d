from io_utils import args, read_normal_data
import os, glob, csv, subprocess

algorithm_data = read_normal_data(args)
available_gpus = [1,2,3,4,5,6,7]
for data_key, datum in algorithm_data.items():
    prompt_idx = int(datum[0])
    if not (prompt_idx >= args.start_idx and prompt_idx < args.end_idx): continue
    data_path = datum[1].replace("'", "\\'")
    # if prompt_idx!=3: continue
    
    gpu_id = prompt_idx % args.num_gpus
    gpu_id = available_gpus[gpu_id]
    print('NORMAL METRIC | GPU: {} | PROMPT ID: {} | DATA_PATH: {}'.format(gpu_id, prompt_idx, data_path))

    command = '''CUDA_VISIBLE_DEVICES={} python run_scripts/normal_metric/latest_normal_metric_main.py --data_path {}'''.format(gpu_id, data_path)
    command = command + "; sleep infinity"
    subprocess.Popen(["tmux", "new-session", "-d", "-s", str(prompt_idx), command])