import os
import numpy as np
import h5py
from tqdm import tqdm
import shutil

task_names = [
    # 'Rest',
    'WM',
    'LANGUAGE',
    # 'MOTOR',
    'SOCIAL',
    'GAMBLING',
    'RELATIONAL',
    'EMOTION',
]
mat_path = 'Timeseries_NonlinearNDD_HCP-Task_atlas-schaefer-100_lowpass-standard.mat'
save_root = 'Retest_114x1'

if os.path.exists(save_root):
    shutil.rmtree(save_root)
os.makedirs(save_root)
# os.makedirs(save_root, exist_ok=True)
def extract_array(ds):
    if isinstance(ds, h5py.Dataset):
        return ds[:]
    elif isinstance(ds, h5py.Group):
        keys = list(ds.keys())
        if len(keys) == 1:
            return ds[keys[0]][:]
        else:
            raise ValueError(f"Group has multiple keys: {keys}")
    else:
        raise ValueError(f"Unexpected type: {type(ds)}")

with h5py.File(mat_path, 'r') as f:
    for task in task_names:
        print(f"🔄 Processing task: {task}")
        task_refs = f['HCP_fMRI_TS'][task]

        # 创建保存目录
        save_nat_dir = os.path.join(save_root, task, 'natural')
        save_dec_dir = os.path.join(save_root, task, 'deconv')
        os.makedirs(save_nat_dir, exist_ok=True)
        os.makedirs(save_dec_dir, exist_ok=True)

        for subj_idx in tqdm(range(task_refs.shape[0])):
            ref = task_refs[subj_idx][0]
            group = f[ref]

            try:
                natural = extract_array(group['natural'])  # shape: (114, T)
                deconv = extract_array(group['deconv'])    # shape: (114, T)

                T = natural.shape[1]
                for t in range(T):
                    nat_vec = natural[:, t]
                    dec_vec = deconv[:, t]

                    filename = f"subj{subj_idx:03d}_t{t:03d}.npy"
                    np.save(os.path.join(save_nat_dir, filename), nat_vec)
                    np.save(os.path.join(save_dec_dir, filename), dec_vec)

            except Exception as e:
                continue  # Skip invalid subject
