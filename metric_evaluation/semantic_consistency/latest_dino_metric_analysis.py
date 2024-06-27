import os, glob, csv, subprocess
import numpy as np
from io_utils import args, read_dino_metric_analysis_data
    

# def read_data(args):
#     log_file = os.path.join(args.log_dir, "threestudio_logs_{}.csv".format(args.algorithm_name))
#     with open(log_file, 'r') as file:
#         csv_reader = csv.reader(file)
#         data = {}
#         for idx, row in enumerate(csv_reader):
#             if idx==0: continue
#             if args.algorithm_name=="magic3d-refine-sd" or args.algorithm_name=="textmesh-if":
#                 folder_content = sorted(glob.glob(os.path.join(args.base_dir, row[2].replace("outputs", "outputs_cleaned").split('@')[0] + "@*", 'save/*/batch_data/*.npy')))
#             else:
#                 folder_content = sorted(glob.glob(os.path.join(args.base_dir, row[2], 'save/*/batch_data/*.npy')))
#             if len(folder_content)>0:
#                 data[folder_content[0].split('@')[0]] = (row[0], "/" + "/".join(folder_content[0].split('/')[1:-2]))

#     return data


def log_dino_metric_data(csv_file_path, algorithm_name, folder_name, dino_verts_var, dino_verts_std, dino_verts_normalized_std):

    # Compute statistics for dino_verts_var
    dino_verts_var_mean = np.mean(dino_verts_var)
    dino_verts_var_min = np.min(dino_verts_var)
    dino_verts_var_max = np.max(dino_verts_var)
    dino_verts_var_median = np.median(dino_verts_var)
    # dino_verts_var_mode = float(np.argmax(np.bincount(dino_verts_var)))

    # Compute statistics for dino_verts_std
    dino_verts_std_mean = np.mean(dino_verts_std)
    dino_verts_std_min = np.min(dino_verts_std)
    dino_verts_std_max = np.max(dino_verts_std)
    dino_verts_std_median = np.median(dino_verts_std)
    threshold_value=0.144
    mask = (dino_verts_std>threshold_value)*1.0
    dino_verts_std_mean_threshold_0_144 = np.sum(dino_verts_std * mask) / (np.sum(mask)+1.e-8)
    dino_verts_std_invalid_verts_0_144 = np.sum(mask) / (dino_verts_std.shape[0]+1.e-8)
    # dino_verts_std_mode = float(np.argmax(np.bincount(dino_verts_std)))

    # Compute statistics for dino_verts_normalized_std
    dino_verts_normalized_std_mean = np.mean(dino_verts_normalized_std)
    dino_verts_normalized_std_min = np.min(dino_verts_normalized_std)
    dino_verts_normalized_std_max = np.max(dino_verts_normalized_std)
    dino_verts_normalized_std_median = np.median(dino_verts_normalized_std)
    # dino_verts_normalized_std_mode = float(np.argmax(np.bincount(dino_verts_normalized_std)))

    # Prepare data for CSV
    csv_data = []
    if not os.path.exists(csv_file_path):
        csv_data = [
            [
                "Algorithm Name", "Folder Name", 
                "dino_verts_var_mean", "dino_verts_var_min", "dino_verts_var_max", "dino_verts_var_median",
                "dino_verts_std_mean", "dino_verts_std_min", "dino_verts_std_max", "dino_verts_std_median", "dino_verts_std_mean_threshold_0.144", "dino_verts_std_invalid_verts_0.144",
                "dino_verts_normalized_std_mean", "dino_verts_normalized_std_min", "dino_verts_normalized_std_max", "dino_verts_normalized_std_median",
            ]
        ]

    # Append data to CSV data
    csv_data.append([
        algorithm_name, folder_name.split('/')[-1], 
        dino_verts_var_mean, dino_verts_var_min, dino_verts_var_max, dino_verts_var_median,
        dino_verts_std_mean, dino_verts_std_min, dino_verts_std_max, dino_verts_std_median, dino_verts_std_mean_threshold_0_144, dino_verts_std_invalid_verts_0_144,
        dino_verts_normalized_std_mean, dino_verts_normalized_std_min, dino_verts_normalized_std_max, dino_verts_normalized_std_median
    ])


    # Write or append data to CSV
    with open(csv_file_path, mode='a' if os.path.exists(csv_file_path) else 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(csv_data)


meta_data = read_dino_metric_analysis_data(args)

for data_idx, key in enumerate(meta_data.keys()):
    try:
        data_path = meta_data[key]
        prompt_idx = int(data_path[0])
        if not (prompt_idx >= args.start_idx and prompt_idx < args.end_idx): continue
        
        data_path_1 = data_path[1].replace("'", "\\'")
        
        gpu_id = prompt_idx % args.num_gpus
        print('DINO ANALYSIS | GPU: {} | PROMPT ID: {} | DATA_PATH: {}'.format(gpu_id, prompt_idx, key.split('/')[-1]))
        command = '''CUDA_VISIBLE_DEVICES={} python dino_metric/latest_dino_metric_analysis_main.py --data_path {}'''.format(gpu_id, data_path_1)
        # print(command)
        command = command + "; sleep infinity"
        subprocess.Popen(["tmux", "new-session", "-d", "-s", str(prompt_idx), command])

        
        data_dir = data_path_1
        dino_verts_mean = os.path.join(data_dir, "dino_variance_data_latest", "dino_verts_mean.npy")
        dino_verts_var = os.path.join(data_dir, "dino_variance_data_latest", "dino_verts_variance.npy")
        dino_verts_std = os.path.join(data_dir, "dino_variance_data_latest", "dino_verts_std.npy")
        dino_verts_normalized_std = os.path.join(data_dir, "dino_variance_data_latest", "normalized_cleaned_dino_verts_std.npy")
        
        dino_verts_mean, dino_verts_var, dino_verts_std, dino_verts_normalized_std = \
            np.load(dino_verts_mean), np.load(dino_verts_var), np.load(dino_verts_std), np.load(dino_verts_normalized_std)

        csv_file_path = os.path.join(args.base_dir, "metric_analysis", "dino_metric_{}.csv".format(args.algorithm_name))
        if not os.path.exists(os.path.join(args.base_dir, "metric_analysis")):
            os.system('mkdir -p ' + os.path.join(args.base_dir, "metric_analysis"))
        log_dino_metric_data(csv_file_path, args.algorithm_name, key, dino_verts_var, dino_verts_std, dino_verts_normalized_std)
    except:
        print('not executing: ', key.split('/')[-1])