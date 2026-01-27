import os
import numpy as np
from collections import defaultdict
import random
import pandas as pd
def compute_roi_mean_std(subject_set, eps: float = 1e-6):
    """
    subject_set: [(label_8, label_770, full_series[R,L]), ...]
    只用训练集计算逐 ROI 的 mean/std，避免数据泄漏。
    返回:
      mean: [R, 1]
      std:  [R, 1]
    """
    # 初始化
    _, _, first = subject_set[0]
    R = first.shape[0]
    sum_ = np.zeros((R, 1), dtype=np.float64)
    sumsq = np.zeros((R, 1), dtype=np.float64)
    count = 0  # 累计时间点数（总列数）

    # 逐样本累加（内存友好，不拼接）
    for _, _, full in subject_set:
        # full: [R, L_i]
        sum_  += full.sum(axis=1, keepdims=True)
        sumsq += (full ** 2).sum(axis=1, keepdims=True)
        count += full.shape[1]

    mean = sum_ / max(count, 1)
    var  = sumsq / max(count, 1) - mean ** 2
    var  = np.maximum(var, eps)           # 防止负零/数值问题
    std  = np.sqrt(var)

    return mean.astype(np.float32), std.astype(np.float32)

class ROIStandardScaler:
    """
    Per-ROI standardization for arrays shaped [R, L].
    mean/std 可为 shape [R] 或 [R,1]；内部会 reshape 成 [R,1] 便于广播。
    """
    def __init__(self, mean: np.ndarray, std: np.ndarray, eps: float = 1e-6):
        mean = np.asarray(mean, dtype=np.float32)
        std  = np.asarray(std,  dtype=np.float32)
        if mean.ndim == 1:
            mean = mean[:, None]
        if std.ndim == 1:
            std = std[:, None]
        self.mean = mean
        self.std  = std
        self.eps  = eps

    def transform(self, data: np.ndarray) -> np.ndarray:
        # data: [R, L]
        data = np.asarray(data, dtype=np.float32)
        assert data.shape[0] == self.mean.shape[0], \
            f"R mismatch: data {data.shape[0]} vs mean {self.mean.shape[0]}"
        return (data - self.mean) / (self.std + self.eps)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=np.float32)
        return data * (self.std + self.eps) + self.mean

def load_and_split_subjects_train_cls(data_dir='processed_timeseries_114x1',
                            map_sub_label = './dataset/subid_mapping.csv',
                            condition='natural',
                            train_ratio=0.7,
                            seed=42):
    df = pd.read_csv(map_sub_label)
    idx_to_label = dict(zip(df['sub_idx'], df['Label']))
    task_to_index = {
        'Rest': 0,
        'WM': 1,
        'LANGUAGE': 2,
        'MOTOR': 3,
        'SOCIAL': 4,
        'GAMBLING': 5,
        'RELATIONAL': 6,
        'EMOTION': 7,
    }
    train_set = []
    test_set = []  # [(task_label, subj_id, full_series)]
    Retest_set = []
    for task in sorted(os.listdir(data_dir)):
        if task.startswith('.'):
            continue

        task_label = task_to_index[task]
        task_dir = os.path.join(data_dir, task, condition)
        if not os.path.isdir(task_dir):
            continue

        subject_dict = defaultdict(list)
        for fname in os.listdir(task_dir):
            if fname.endswith('.npy'):
                subj_id = fname.split('_t')[0]
                subject_dict[subj_id].append(fname)

        for subj_id, file_list in subject_dict.items():
            file_list.sort()
            time_series = [np.load(os.path.join(task_dir, f)).reshape(114, 1) for f in file_list]
            if len(time_series) == 0:
                continue
            full_series = np.concatenate(time_series, axis=1)

            if task == 'Rest':
                retest = full_series[:, 2400:]
                full_series = full_series[:, :2400]

            idx = int(subj_id.replace('subj', ''))
            if idx not in idx_to_label:
                raise KeyError(f"❌ Subject index {idx} not found in mapping CSV!")
            subj_label = idx_to_label[idx]
            if task == 'Rest':
                Retest_set.append((task_label, subj_label, retest))

            if subj_label == -1:
                continue
            T = full_series.shape[1]
            split_idx = int(train_ratio * T)
            train_series = full_series[:, :split_idx]
            test_series = full_series[:, split_idx:]
            if train_series.shape[1] > 0:
                train_set.append((task_label, subj_label, train_series))
            if test_series.shape[1] > 0:
                test_set.append((task_label, subj_label, test_series))

    return train_set, test_set, Retest_set


def load_and_split_subjects_train_cls_random(data_dir='processed_timeseries_114x1',
                                      map_sub_label='./dataset/subid_mapping.csv',
                                      condition='natural',
                                      train_ratio=0.7,
                                      seed=42):
    df = pd.read_csv(map_sub_label)
    idx_to_label = dict(zip(df['sub_idx'], df['Label']))
    task_to_index = {
        'Rest': 0,
        'WM': 1,
        'LANGUAGE': 2,
        'MOTOR': 3,
        'SOCIAL': 4,
        'GAMBLING': 5,
        'RELATIONAL': 6,
        'EMOTION': 7,
    }
    train_set = []
    test_set = []  # [(task_label, subj_id, full_series)]
    Retest_set = []
    for task in sorted(os.listdir(data_dir)):
        if task.startswith('.'):
            continue

        task_label = task_to_index[task]
        task_dir = os.path.join(data_dir, task, condition)
        if not os.path.isdir(task_dir):
            continue

        subject_dict = defaultdict(list)
        for fname in os.listdir(task_dir):
            if fname.endswith('.npy'):
                subj_id = fname.split('_t')[0]
                subject_dict[subj_id].append(fname)

        for subj_id, file_list in subject_dict.items():
            file_list.sort()
            time_series = [np.load(os.path.join(task_dir, f)).reshape(114, 1) for f in file_list]
            if len(time_series) == 0:
                continue
            full_series = np.concatenate(time_series, axis=1)
            # 备份（可选但推荐）
            full_series_orig = full_series.copy()

            # 每个样本（每列）独立打乱时间顺序，并覆盖回 full_series
            N_time = full_series.shape[1]
            rng = np.random.default_rng(0)
            perm_t = rng.permutation(N_time)  # 对时间点做同一个置换
            full_series = full_series[:, perm_t]

            if task == 'Rest':
                retest = full_series[:, 2400:]
                full_series = full_series[:, :2400]

            idx = int(subj_id.replace('subj', ''))
            if idx not in idx_to_label:
                raise KeyError(f"❌ Subject index {idx} not found in mapping CSV!")
            subj_label = idx_to_label[idx]
            if task == 'Rest':
                Retest_set.append((task_label, subj_label, retest))

            if subj_label == -1:
                continue
            T = full_series.shape[1]
            split_idx = int(train_ratio * T)
            train_series = full_series[:, :split_idx]
            test_series = full_series[:, split_idx:]
            if train_series.shape[1] > 0:
                train_set.append((task_label, subj_label, train_series))
            if test_series.shape[1] > 0:
                test_set.append((task_label, subj_label, test_series))

    return train_set, test_set, Retest_set
def load_and_split_subjects_train_reg(data_dir='processed_timeseries_114x1',
                            map_sub_label = './dataset/subid_mapping.csv',
                            condition='natural',
                            train_ratio=0.7,
                            seed=42):
    df = pd.read_csv(map_sub_label)
    idx_to_label_STORY = dict(zip(df['sub_idx'], df['STORY_RT']))
    idx_to_label_MATH = dict(zip(df['sub_idx'], df['MATH_RT']))
    idx_to_label_Rela = dict(zip(df['sub_idx'], df['Relational_RT']))
    idx_to_label_TOM = dict(zip(df['sub_idx'], df['TOM_RT']))
    idx_to_label_Random = dict(zip(df['sub_idx'], df['Random_RT']))
    idx_to_label_twoback = dict(zip(df['sub_idx'], df['TwoBackRT']))

    story_mask = np.loadtxt('./dataset/story_mask.csv', delimiter=',')
    math_mask = np.loadtxt('./dataset/math_mask.csv', delimiter=',')
    TOM_mask = np.loadtxt('./dataset/TOM_mask.csv', delimiter=',')
    Random_mask = np.loadtxt('./dataset/RC_mask.csv', delimiter=',')
    twoback_mask = np.loadtxt('./dataset/twoback_mask.csv', delimiter=',')

    task_to_index = {
        'STORY': 0,
        'MATH': 1,
        'TOM': 2,
        'Random': 3,
        'WM': 4,
        'Rela': 5,
    }
    dataset = []
    for task in sorted(os.listdir(data_dir)):
        if task.startswith('.'):
            continue
        if task == 'SOCIAL':
            task_dir = os.path.join(data_dir, task, condition)
            if not os.path.isdir(task_dir):
                continue

            subject_dict = defaultdict(list)
            for fname in os.listdir(task_dir):
                if fname.endswith('.npy'):
                    subj_id = fname.split('_t')[0]
                    subject_dict[subj_id].append(fname)

            for subj_id, file_list in subject_dict.items():
                file_list.sort()
                time_series = [np.load(os.path.join(task_dir, f)).reshape(114, 1) for f in file_list]
                if len(time_series) == 0:
                    continue
                full_series = np.concatenate(time_series, axis=1)
                L_TOM = min(full_series.shape[1], len(TOM_mask))
                TOM_series = full_series[:, TOM_mask[:L_TOM] == 1]
                idx = int(subj_id.replace('subj', ''))
                if idx not in idx_to_label_TOM:
                    raise KeyError(f"❌ Subject index {idx} not found in mapping CSV!")
                subj_label = idx_to_label_TOM[idx]
                if pd.isna(subj_label):
                    continue
                if subj_label == -1:
                    continue
                task_label = task_to_index["TOM"]
                dataset.append((task_label, subj_label, TOM_series))

                L_Random = min(full_series.shape[1], len(Random_mask))
                Random_series = full_series[:, Random_mask[: L_Random] == 1]
                idx = int(subj_id.replace('subj', ''))
                if idx not in idx_to_label_Random:
                    raise KeyError(f"❌ Subject index {idx} not found in mapping CSV!")
                subj_label = idx_to_label_Random[idx]
                if pd.isna(subj_label):
                    continue
                if subj_label == -1:
                    continue
                task_label = task_to_index["Random"]
                dataset.append((task_label, subj_label, Random_series))

        if task == 'LANGUAGE':
            task_dir = os.path.join(data_dir, task, condition)
            if not os.path.isdir(task_dir):
                continue

            subject_dict = defaultdict(list)
            for fname in os.listdir(task_dir):
                if fname.endswith('.npy'):
                    subj_id = fname.split('_t')[0]
                    subject_dict[subj_id].append(fname)

            for subj_id, file_list in subject_dict.items():
                file_list.sort()
                time_series = [np.load(os.path.join(task_dir, f)).reshape(114, 1) for f in file_list]
                if len(time_series) == 0:
                    continue
                full_series = np.concatenate(time_series, axis=1)
                L_story = min(full_series.shape[1], len(story_mask))
                story_series = full_series[:, story_mask[:L_story] == 1]
                idx = int(subj_id.replace('subj', ''))
                if idx not in idx_to_label_STORY:
                    raise KeyError(f"❌ Subject index {idx} not found in mapping CSV!")
                subj_label = idx_to_label_STORY[idx]
                if pd.isna(subj_label):
                    continue
                if subj_label == -1:
                    continue
                task_label = task_to_index["STORY"]
                dataset.append((task_label, subj_label, story_series))

                L_math = min(full_series.shape[1], len(math_mask))
                math_series = full_series[:, math_mask[: L_math] == 1]
                idx = int(subj_id.replace('subj', ''))
                if idx not in idx_to_label_MATH:
                    raise KeyError(f"❌ Subject index {idx} not found in mapping CSV!")
                subj_label = idx_to_label_MATH[idx]
                if pd.isna(subj_label):
                    continue
                if subj_label == -1:
                    continue
                task_label = task_to_index["MATH"]
                dataset.append((task_label, subj_label, math_series))
        if task == 'RELATIONAL':
            task_dir = os.path.join(data_dir, task, condition)
            if not os.path.isdir(task_dir):
                continue

            subject_dict = defaultdict(list)
            for fname in os.listdir(task_dir):
                if fname.endswith('.npy'):
                    subj_id = fname.split('_t')[0]
                    subject_dict[subj_id].append(fname)

            for subj_id, file_list in subject_dict.items():
                file_list.sort()
                time_series = [np.load(os.path.join(task_dir, f)).reshape(114, 1) for f in file_list]
                if len(time_series) == 0:
                    continue
                full_series = np.concatenate(time_series, axis=1)
                Rela_series = full_series
                idx = int(subj_id.replace('subj', ''))
                if idx not in idx_to_label_Rela:
                    raise KeyError(f"❌ Subject index {idx} not found in mapping CSV!")
                subj_label = idx_to_label_Rela[idx]
                if pd.isna(subj_label):
                    continue
                if subj_label == -1:
                    continue
                task_label = task_to_index["Rela"]
                dataset.append((task_label, subj_label, Rela_series))

        if task == 'WM':
            task_dir = os.path.join(data_dir, task, condition)
            if not os.path.isdir(task_dir):
                continue

            subject_dict = defaultdict(list)
            for fname in os.listdir(task_dir):
                if fname.endswith('.npy'):
                    subj_id = fname.split('_t')[0]
                    subject_dict[subj_id].append(fname)

            for subj_id, file_list in subject_dict.items():
                file_list.sort()
                time_series = [np.load(os.path.join(task_dir, f)).reshape(114, 1) for f in file_list]
                if len(time_series) == 0:
                    continue
                full_series = np.concatenate(time_series, axis=1)
                L_twoback = min(full_series.shape[1], len(twoback_mask))
                twoback_series = full_series[:, twoback_mask[:L_twoback] == 1]
                idx = int(subj_id.replace('subj', ''))
                if idx not in idx_to_label_twoback:
                    raise KeyError(f"❌ Subject index {idx} not found in mapping CSV!")
                subj_label = idx_to_label_twoback[idx]
                if pd.isna(subj_label):
                    continue
                if subj_label == -1:
                    continue
                task_label = task_to_index["WM"]
                dataset.append((task_label, subj_label, twoback_series))

    random.seed(seed)
    # 打乱原始数据集
    random.shuffle(dataset)
    split_idx = int(len(dataset) * train_ratio)
    train_set = dataset[:split_idx]
    test_set = dataset[split_idx:]

    return train_set, test_set

def load_and_split_subjects_train_reg_random(data_dir='processed_timeseries_114x1',
                            map_sub_label = './dataset/subid_mapping.csv',
                            condition='natural',
                            train_ratio=0.7,
                            seed=42):
    df = pd.read_csv(map_sub_label)
    idx_to_label_STORY = dict(zip(df['sub_idx'], df['STORY_RT']))
    idx_to_label_MATH = dict(zip(df['sub_idx'], df['MATH_RT']))
    idx_to_label_Rela = dict(zip(df['sub_idx'], df['Relational_RT']))
    idx_to_label_TOM = dict(zip(df['sub_idx'], df['TOM_RT']))
    idx_to_label_Random = dict(zip(df['sub_idx'], df['Random_RT']))
    idx_to_label_twoback = dict(zip(df['sub_idx'], df['TwoBackRT']))

    story_mask = np.loadtxt('./dataset/story_mask.csv', delimiter=',')
    math_mask = np.loadtxt('./dataset/math_mask.csv', delimiter=',')
    TOM_mask = np.loadtxt('./dataset/TOM_mask.csv', delimiter=',')
    Random_mask = np.loadtxt('./dataset/RC_mask.csv', delimiter=',')
    twoback_mask = np.loadtxt('./dataset/twoback_mask.csv', delimiter=',')

    task_to_index = {
        'STORY': 0,
        'MATH': 1,
        'TOM': 2,
        'Random': 3,
        'WM': 4,
        'Rela': 5,
    }
    dataset = []
    for task in sorted(os.listdir(data_dir)):
        if task.startswith('.'):
            continue
        if task == 'SOCIAL':
            task_dir = os.path.join(data_dir, task, condition)
            if not os.path.isdir(task_dir):
                continue

            subject_dict = defaultdict(list)
            for fname in os.listdir(task_dir):
                if fname.endswith('.npy'):
                    subj_id = fname.split('_t')[0]
                    subject_dict[subj_id].append(fname)

            for subj_id, file_list in subject_dict.items():
                file_list.sort()
                time_series = [np.load(os.path.join(task_dir, f)).reshape(114, 1) for f in file_list]
                if len(time_series) == 0:
                    continue
                full_series = np.concatenate(time_series, axis=1)
                full_series_orig = full_series.copy()
                N_time = full_series.shape[1]
                rng = np.random.default_rng(0)
                perm_t = rng.permutation(N_time)  # 对时间点做同一个置换
                full_series = full_series[:, perm_t]

                L_TOM = min(full_series.shape[1], len(TOM_mask))
                TOM_series = full_series[:, TOM_mask[:L_TOM] == 1]
                idx = int(subj_id.replace('subj', ''))
                if idx not in idx_to_label_TOM:
                    raise KeyError(f"❌ Subject index {idx} not found in mapping CSV!")
                subj_label = idx_to_label_TOM[idx]
                if pd.isna(subj_label):
                    continue
                if subj_label == -1:
                    continue
                task_label = task_to_index["TOM"]
                dataset.append((task_label, subj_label, TOM_series))

                L_Random = min(full_series.shape[1], len(Random_mask))
                Random_series = full_series[:, Random_mask[: L_Random] == 1]
                idx = int(subj_id.replace('subj', ''))
                if idx not in idx_to_label_Random:
                    raise KeyError(f"❌ Subject index {idx} not found in mapping CSV!")
                subj_label = idx_to_label_Random[idx]
                if pd.isna(subj_label):
                    continue
                if subj_label == -1:
                    continue
                task_label = task_to_index["Random"]
                dataset.append((task_label, subj_label, Random_series))

        if task == 'LANGUAGE':
            task_dir = os.path.join(data_dir, task, condition)
            if not os.path.isdir(task_dir):
                continue

            subject_dict = defaultdict(list)
            for fname in os.listdir(task_dir):
                if fname.endswith('.npy'):
                    subj_id = fname.split('_t')[0]
                    subject_dict[subj_id].append(fname)

            for subj_id, file_list in subject_dict.items():
                file_list.sort()
                time_series = [np.load(os.path.join(task_dir, f)).reshape(114, 1) for f in file_list]
                if len(time_series) == 0:
                    continue
                full_series = np.concatenate(time_series, axis=1)
                full_series_orig = full_series.copy()
                N_time = full_series.shape[1]
                rng = np.random.default_rng(0)
                perm_t = rng.permutation(N_time)  # 对时间点做同一个置换
                full_series = full_series[:, perm_t]
                L_story = min(full_series.shape[1], len(story_mask))
                story_series = full_series[:, story_mask[:L_story] == 1]
                idx = int(subj_id.replace('subj', ''))
                if idx not in idx_to_label_STORY:
                    raise KeyError(f"❌ Subject index {idx} not found in mapping CSV!")
                subj_label = idx_to_label_STORY[idx]
                if pd.isna(subj_label):
                    continue
                if subj_label == -1:
                    continue
                task_label = task_to_index["STORY"]
                dataset.append((task_label, subj_label, story_series))

                L_math = min(full_series.shape[1], len(math_mask))
                math_series = full_series[:, math_mask[: L_math] == 1]
                idx = int(subj_id.replace('subj', ''))
                if idx not in idx_to_label_MATH:
                    raise KeyError(f"❌ Subject index {idx} not found in mapping CSV!")
                subj_label = idx_to_label_MATH[idx]
                if pd.isna(subj_label):
                    continue
                if subj_label == -1:
                    continue
                task_label = task_to_index["MATH"]
                dataset.append((task_label, subj_label, math_series))
        if task == 'RELATIONAL':
            task_dir = os.path.join(data_dir, task, condition)
            if not os.path.isdir(task_dir):
                continue

            subject_dict = defaultdict(list)
            for fname in os.listdir(task_dir):
                if fname.endswith('.npy'):
                    subj_id = fname.split('_t')[0]
                    subject_dict[subj_id].append(fname)

            for subj_id, file_list in subject_dict.items():
                file_list.sort()
                time_series = [np.load(os.path.join(task_dir, f)).reshape(114, 1) for f in file_list]
                if len(time_series) == 0:
                    continue
                full_series = np.concatenate(time_series, axis=1)
                full_series_orig = full_series.copy()
                N_time = full_series.shape[1]
                rng = np.random.default_rng(0)
                perm_t = rng.permutation(N_time)  # 对时间点做同一个置换
                full_series = full_series[:, perm_t]
                Rela_series = full_series
                idx = int(subj_id.replace('subj', ''))
                if idx not in idx_to_label_Rela:
                    raise KeyError(f"❌ Subject index {idx} not found in mapping CSV!")
                subj_label = idx_to_label_Rela[idx]
                if pd.isna(subj_label):
                    continue
                if subj_label == -1:
                    continue
                task_label = task_to_index["Rela"]
                dataset.append((task_label, subj_label, Rela_series))

        if task == 'WM':
            task_dir = os.path.join(data_dir, task, condition)
            if not os.path.isdir(task_dir):
                continue

            subject_dict = defaultdict(list)
            for fname in os.listdir(task_dir):
                if fname.endswith('.npy'):
                    subj_id = fname.split('_t')[0]
                    subject_dict[subj_id].append(fname)

            for subj_id, file_list in subject_dict.items():
                file_list.sort()
                time_series = [np.load(os.path.join(task_dir, f)).reshape(114, 1) for f in file_list]
                if len(time_series) == 0:
                    continue
                full_series = np.concatenate(time_series, axis=1)
                full_series_orig = full_series.copy()
                N_time = full_series.shape[1]
                rng = np.random.default_rng(0)
                perm_t = rng.permutation(N_time)  # 对时间点做同一个置换
                full_series = full_series[:, perm_t]
                L_twoback = min(full_series.shape[1], len(twoback_mask))
                twoback_series = full_series[:, twoback_mask[:L_twoback] == 1]
                idx = int(subj_id.replace('subj', ''))
                if idx not in idx_to_label_twoback:
                    raise KeyError(f"❌ Subject index {idx} not found in mapping CSV!")
                subj_label = idx_to_label_twoback[idx]
                if pd.isna(subj_label):
                    continue
                if subj_label == -1:
                    continue
                task_label = task_to_index["WM"]
                dataset.append((task_label, subj_label, twoback_series))



    random.seed(seed)
    # 打乱原始数据集
    random.shuffle(dataset)
    split_idx = int(len(dataset) * train_ratio)
    train_set = dataset[:split_idx]
    test_set = dataset[split_idx:]

    return train_set, test_set

# def load_and_split_subjects_train_SZ(data_dir='processed_timeseries_114x1',
#                             map_sub_label = './dataset/subid_mapping.csv',
#                             condition='natural',
#                             train_ratio=0.7,
#                             seed=42):
#     df = pd.read_csv(map_sub_label)
#     idx_to_label = dict(zip(df['subID'], df['diagnosis']))
#     idx_to_conv = dict(zip(df['subID'], df['converter']))
#     task_to_index = {
#         'diagnosis': 0,
#     }
#     diagnosis_to_index = {
#         'Healthy': 0,
#         'HighRisk': 0,
#         'SZ': 1,
#     }
#     train_set = []
#     test_set = []  # [(task_label, subj_id, full_series)]
#     # full_set = []
#     for task in sorted(os.listdir(data_dir)):
#         if task.startswith('.'):
#             continue
#
#         task_label = task_to_index[task]
#         task_dir = os.path.join(data_dir, task, condition)
#         if not os.path.isdir(task_dir):
#             continue
#
#         subject_dict = defaultdict(list)
#         for fname in os.listdir(task_dir):
#             if fname.endswith('.npy'):
#                 subj_id = fname.split('_t')[0]
#                 subject_dict[subj_id].append(fname)
#
#         for subj_id, file_list in subject_dict.items():
#             file_list.sort()
#             time_series = [np.load(os.path.join(task_dir, f)).reshape(86, 1) for f in file_list]
#             if len(time_series) == 0:
#                 continue
#             full_series = np.concatenate(time_series, axis=1)
#             # if task == 'Rest':
#             #     full_series = full_series[:, :2400]
#             # idx = int(subj_id.replace('subj', ''))
#             if subj_id not in idx_to_label:
#                 raise KeyError(f"❌ Subject index {subj_id} not found in mapping CSV!")
#             subj_label = idx_to_label[subj_id]
#             conv_label = idx_to_conv[subj_id]
#             if conv_label == 1:
#                 continue
#             subj_label = diagnosis_to_index[subj_label]
#
#             # full_set.append((task_label, subj_label, full_series))
#             T = full_series.shape[1]
#             split_idx = int(train_ratio * T)
#             train_series = full_series[:, :split_idx]
#             test_series = full_series[:, split_idx:]
#             if train_series.shape[1] > 0:
#                 train_set.append((task_label, subj_label, train_series))
#             if test_series.shape[1] > 0:
#                 test_set.append((task_label, subj_label, test_series))
#     # random.shuffle(full_set)
#     # train_size = int(len(full_set) * train_ratio)
#     # train_set = full_set[:train_size]
#     # test_set = full_set[train_size:]
#     return train_set, test_set
def generate_sliding_windows(subject_set, sequence_length, window_size):
    samples, tasks, sub_idx = [], [], []
    target_len = 464
    for task_label, subj_label, full_series in subject_set:
        D, T = full_series.shape  # D=114, T=原始长度
        pad_width = target_len - T
        padded = np.pad(full_series, ((0, 0), (0, pad_width)), mode='constant')
        tasks.append(task_label)
        sub_idx.append(subj_label)
        samples.append(padded)
    return samples, tasks, sub_idx

def load_and_split_subjects(data_dir_Rest='processed_timeseries_114x1',
                            data_dir_activity='data-Retest',
                            map_sub_label = './dataset/subid_mapping.csv',
                            condition='natural',
                            train_ratio=0.7,
                            train_series_length=None,
                            seed=42):
    df = pd.read_csv(map_sub_label)
    idx_to_label = dict(zip(df['sub_idx'], df['Label']))
    sub_id_to_label = dict(zip(df['sub_id'], df['Label']))
    task_to_index = {
        'Rest': 0,
        'WM': 1,
        'LANGUAGE': 2,
        'MOTOR': 3,
        'SOCIAL': 4,
        'GAMBLING': 5,
        'RELATIONAL': 6,
        'EMOTION': 7,
    }
    train_set = []
    test_set = []  # [(task_label, subj_id, full_series)]
    # for data_dir in [data_dir_Rest, data_dir_activity]:
    for data_dir in [data_dir_activity]:
        for task in sorted(os.listdir(data_dir)):
            if task.startswith('.'):
                continue
            if data_dir == data_dir_Rest and task != 'Rest':
                continue
            task_label = task_to_index[task]
            task_dir = os.path.join(data_dir, task, condition)
            if not os.path.isdir(task_dir):
                continue

            subject_dict = defaultdict(list)
            for fname in os.listdir(task_dir):
                if fname.endswith('.npy'):
                    subj_id = fname.split('_t')[0]
                    subject_dict[subj_id].append(fname)

            for subj_id, file_list in subject_dict.items():
                file_list.sort()
                time_series = [np.load(os.path.join(task_dir, f)).reshape(114, 1) for f in file_list]
                if len(time_series) == 0:
                    continue
                full_series = np.concatenate(time_series, axis=1)
                if data_dir == data_dir_Rest and task == 'Rest':
                    full_series = full_series[:, -2400:]
                    idx = int(subj_id.replace('subj', ''))
                    if idx not in idx_to_label:
                        raise KeyError(f"❌ Subject index {idx} not found in mapping CSV!")
                    subj_label = idx_to_label[idx]
                else:
                    subj_label = sub_id_to_label[subj_id]

                if subj_label == -1:
                    continue
                T = full_series.shape[1]
                split_idx = int(train_ratio * T)
                train_series = full_series[:, :split_idx]
                test_series = full_series[:, split_idx:]
                if train_series.shape[1] > 0:
                    train_set.append((task_label, subj_label, train_series))
                if test_series.shape[1] > 0:
                    test_set.append((task_label, subj_label, test_series))

    return train_set, test_set

def load_and_split_subjects_random(data_dir_Rest='processed_timeseries_114x1',
                            data_dir_activity='data-Retest',
                            map_sub_label = './dataset/subid_mapping.csv',
                            condition='natural',
                            train_ratio=0.7,
                            train_series_length=None,
                            seed=42):
    df = pd.read_csv(map_sub_label)
    idx_to_label = dict(zip(df['sub_idx'], df['Label']))
    sub_id_to_label = dict(zip(df['sub_id'], df['Label']))
    task_to_index = {
        'Rest': 0,
        'WM': 1,
        'LANGUAGE': 2,
        'MOTOR': 3,
        'SOCIAL': 4,
        'GAMBLING': 5,
        'RELATIONAL': 6,
        'EMOTION': 7,
    }
    train_set = []
    test_set = []  # [(task_label, subj_id, full_series)]
    # for data_dir in [data_dir_Rest, data_dir_activity]:
    for data_dir in [data_dir_activity]:
        for task in sorted(os.listdir(data_dir)):
            if task.startswith('.'):
                continue
            if data_dir == data_dir_Rest and task != 'Rest':
                continue
            task_label = task_to_index[task]
            task_dir = os.path.join(data_dir, task, condition)
            if not os.path.isdir(task_dir):
                continue

            subject_dict = defaultdict(list)
            for fname in os.listdir(task_dir):
                if fname.endswith('.npy'):
                    subj_id = fname.split('_t')[0]
                    subject_dict[subj_id].append(fname)

            for subj_id, file_list in subject_dict.items():
                file_list.sort()
                time_series = [np.load(os.path.join(task_dir, f)).reshape(114, 1) for f in file_list]
                if len(time_series) == 0:
                    continue
                full_series = np.concatenate(time_series, axis=1)
                full_series_orig = full_series.copy()
                N_time = full_series.shape[1]
                rng = np.random.default_rng(0)
                perm_t = rng.permutation(N_time)  # 对时间点做同一个置换
                full_series = full_series[:, perm_t]
                if data_dir == data_dir_Rest and task == 'Rest':
                    full_series = full_series[:, -2400:]
                    idx = int(subj_id.replace('subj', ''))
                    if idx not in idx_to_label:
                        raise KeyError(f"❌ Subject index {idx} not found in mapping CSV!")
                    subj_label = idx_to_label[idx]
                else:
                    subj_label = sub_id_to_label[subj_id]

                if subj_label == -1:
                    continue
                T = full_series.shape[1]
                split_idx = int(train_ratio * T)
                train_series = full_series[:, :split_idx]
                test_series = full_series[:, split_idx:]
                if train_series.shape[1] > 0:
                    train_set.append((task_label, subj_label, train_series))
                if test_series.shape[1] > 0:
                    test_set.append((task_label, subj_label, test_series))

    return train_set, test_set

def load_and_split_subjects_train_sz(data_dir='processed_timeseries_114x1',
                            map_sub_label = './dataset/Schizophrenia_mapping.csv',
                            condition='natural',
                            train_ratio=0.7,
                            seed=42):
    df = pd.read_csv(map_sub_label)
    idx_to_label = dict(zip(df['subID'], df['diagnosis']))
    idx_to_conv = dict(zip(df['subID'], df['converter']))
    task_to_index = {
        'diagnosis': 0,
    }
    diagnosis_to_index = {
        'Healthy': 0,
        'HighRisk': 0,
        'SZ': 1,
    }
    train_set = []
    test_set = []  # [(task_label, subj_id, full_series)]
    # full_set = []
    for task in sorted(os.listdir(data_dir)):
        if task.startswith('.'):
            continue

        task_label = task_to_index[task]
        task_dir = os.path.join(data_dir, task, condition)
        if not os.path.isdir(task_dir):
            continue

        subject_dict = defaultdict(list)
        for fname in os.listdir(task_dir):
            if fname.endswith('.npy'):
                subj_id = fname.split('_t')[0]
                subject_dict[subj_id].append(fname)

        for subj_id, file_list in subject_dict.items():
            file_list.sort()
            time_series = [np.load(os.path.join(task_dir, f)).reshape(86, 1) for f in file_list]
            if len(time_series) == 0:
                continue
            full_series = np.concatenate(time_series, axis=1)
            # if task == 'Rest':
            #     full_series = full_series[:, :2400]
            # idx = int(subj_id.replace('subj', ''))
            if subj_id not in idx_to_label:
                raise KeyError(f"❌ Subject index {subj_id} not found in mapping CSV!")
            subj_label = idx_to_label[subj_id]
            conv_label = idx_to_conv[subj_id]
            if conv_label == 1:
                continue
            subj_label = diagnosis_to_index[subj_label]

            # full_set.append((task_label, subj_label, full_series))
            T = full_series.shape[1]
            split_idx = int(train_ratio * T)
            train_series = full_series[:, :split_idx]
            test_series = full_series[:, split_idx:]
            if train_series.shape[1] > 0:
                train_set.append((task_label, subj_label, train_series))
            if test_series.shape[1] > 0:
                test_set.append((task_label, subj_label, test_series))
    # random.shuffle(full_set)
    # train_size = int(len(full_set) * train_ratio)
    # train_set = full_set[:train_size]
    # test_set = full_set[train_size:]
    return train_set, test_set

def load_and_split_subjects_test_sz(data_dir='processed_timeseries_114x1',
                            map_sub_label = './dataset/subid_mapping.csv',
                            condition='natural',
                            train_ratio=0.0,
                            seed=42):
    df = pd.read_csv(map_sub_label)
    idx_to_label = dict(zip(df['subID'], df['diagnosis']))
    idx_to_conv = dict(zip(df['subID'], df['converter']))
    task_to_index = {
        'diagnosis': 0,
    }
    diagnosis_to_index = {
        'Healthy': 0,
        'HighRisk': 1,
        'SZ': 0,
    }
    train_set = []
    test_set = []  # [(task_label, subj_id, full_series)]
    # full_set = []
    for task in sorted(os.listdir(data_dir)):
        if task.startswith('.'):
            continue

        # task_label = task_to_index[task]
        task_dir = os.path.join(data_dir, task, condition)
        if not os.path.isdir(task_dir):
            continue

        subject_dict = defaultdict(list)
        for fname in os.listdir(task_dir):
            if fname.endswith('.npy'):
                subj_id = fname.split('_t')[0]
                subject_dict[subj_id].append(fname)

        for subj_id, file_list in subject_dict.items():
            task_label = subj_id
            file_list.sort()
            time_series = [np.load(os.path.join(task_dir, f)).reshape(86, 1) for f in file_list]
            if len(time_series) == 0:
                continue
            full_series = np.concatenate(time_series, axis=1)
            # if task == 'Rest':
            #     full_series = full_series[:, :2400]
            # idx = int(subj_id.replace('subj', ''))
            if subj_id not in idx_to_label:
                raise KeyError(f"❌ Subject index {subj_id} not found in mapping CSV!")
            subj_label = idx_to_label[subj_id]
            conv_label = idx_to_conv[subj_id]
            if conv_label != 1:
                continue
            subj_label = diagnosis_to_index[subj_label]
            if subj_label != 1:
                continue
            # full_set.append((task_label, subj_label, full_series))

            test_series = full_series
            if test_series.shape[1] > 0:
                test_set.append((task_label, subj_label, test_series))
    # random.shuffle(full_set)
    # train_size = int(len(full_set) * train_ratio)
    # train_set = full_set[:train_size]
    # test_set = full_set[train_size:]
    return test_set