from io_utils import args, read_dino_variance_extraction_data
import subprocess, time

job_queue = []
def is_gpu_available(gpu_id):
    return gpu_id not in [job['gpu_id'] for job in job_queue]


meta_data, mesh_data = read_dino_variance_extraction_data(args)

available_gpus = [1,2,3,4,5,6,0,7]
for data_idx, key in enumerate(meta_data.keys()):
    if key not in mesh_data: continue
    data_path, mesh_path = meta_data[key], mesh_data[key]
    # print(data_path, "data_path")
    assert(data_path[0]==mesh_path[0])
    prompt_idx = int(data_path[0])
    if not (prompt_idx >= args.start_idx and prompt_idx < args.end_idx): continue
    
    data_path_1 = data_path[1].replace("'", "\\'")
    mesh_path_1 = mesh_path[1].replace("'", "\\'")
    
    gpu_id = available_gpus[prompt_idx % args.num_gpus]
    print('DINO VARIANCE | GPU: {} | PROMPT ID: {} | DATA_PATH: {} | MESH_PATH: {}'.format(gpu_id, prompt_idx, data_path_1, mesh_path_1))
    command = '''CUDA_VISIBLE_DEVICES={} python dino_metric/latest_dino_variance_generator_main.py --data_path {} --mesh_path {}'''.format(gpu_id, data_path_1, mesh_path_1)
    print(command)
    command = command + "; sleep infinity"
    subprocess.Popen(["tmux", "new-session", "-d", "-s", str(prompt_idx), command])
    