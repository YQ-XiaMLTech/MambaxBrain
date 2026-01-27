import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler

class ClsDatasetStaticFC(Dataset):
    def __init__(self, subject_set, sequence_length, window_size, scaler=None):
        """
        参数:
            subject_set: [(label_8, label_770, full_series)]，full_series 形状 [R, L]
            sequence_length: 滑窗长度
            window_size: 滑窗步长
            scaler: 可选，按 ROI 标准化器（输入期望 [R, L]）
        """
        self.sequence_length = sequence_length
        self.window_size = window_size
        self.scaler = scaler

        self.sequences = []     # 每个元素: ndarray [R, sequence_length]
        self.labels_8 = []      # int
        self.labels_770 = []    # int

        # 生成定长片段
        for label_8, label_770, full_series in subject_set:
            R, T = full_series.shape
            for start in range(0, T - sequence_length + 1, window_size):
                seq = full_series[:, start:start + sequence_length]  # [R, Lw]
                if seq.shape[1] == sequence_length:
                    self.sequences.append(seq)
                    self.labels_8.append(label_8)
                    self.labels_770.append(label_770)

        assert len(self.sequences) > 0, "Empty dataset!"
        # 记录 R 并预生成上三角索引（不含对角）
        self.R = self.sequences[0].shape[0]
        tri = torch.triu_indices(self.R, self.R, offset=1)
        self.triu_i = tri[0]
        self.triu_j = tri[1]
        # 也可检查所有样本 R 是否一致
        for s in self.sequences:
            assert s.shape[0] == self.R, "All samples must have the same ROI count (R)."

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # x_raw: [R, L]
        x_raw = self.sequences[idx].astype(np.float32)

        # 可选标准化：按 ROI（对 [R, L] 进行 transform）
        if self.scaler is not None:
            x_raw = self.scaler.transform(x_raw)

        # 模型时序输入：转为 [L, R]
        x = torch.from_numpy(x_raw.T.copy())  # [L, R], float32

        # 计算静态 FC（在 [R, L] 上）
        # 用 torch 避免 numpy 往返
        with torch.no_grad():
            ts = torch.from_numpy(x_raw)          # [R, L]
            fc = torch.corrcoef(ts)               # [R, R]
            fc = torch.nan_to_num(fc)             # 数值稳健
            upper = fc[self.triu_i, self.triu_j]  # [E]

        sample = {
            "x": x,                                        # [L, R]
            "fc": upper.to(torch.float32),                 # [E]
            "label_8": torch.tensor(self.labels_8[idx], dtype=torch.long),
            "label_770": torch.tensor(self.labels_770[idx], dtype=torch.long),
            "task_kind": "CLS",
        }
        return sample

class RegDatasetStaticFC(Dataset):
    def __init__(self, subject_set, x_scaler=None, y_scaler=None):
        """
        参数:
            subject_set: [(label_8, label_770, full_series)]，full_series 形状 [R, L]
            sequence_length: 滑窗长度
            window_size: 滑窗步长
            scaler: 可选，按 ROI 标准化器（输入期望 [R, L]）
        """
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler

        self.seq_pad = []     # 每个元素: ndarray [R, sequence_length]
        self.labels_task = []      # int
        self.labels_reg = []    # int
        self.sequences = []
        target_len = 464
        # 生成定长片段
        for label_task, label_reg, full_series in subject_set:
            self.sequences.append(full_series)
            R, T = full_series.shape
            pad_width = target_len - T
            padded = np.pad(full_series, ((0, 0), (0, pad_width)), mode='constant')
            self.labels_task.append(label_task)
            self.labels_reg.append(label_reg)
            self.seq_pad.append(padded)

        assert len(self.sequences) > 0, "Empty dataset!"
        # 记录 R 并预生成上三角索引（不含对角）
        self.R = self.sequences[0].shape[0]
        tri = torch.triu_indices(self.R, self.R, offset=1)
        self.triu_i = tri[0]
        self.triu_j = tri[1]
        # 也可检查所有样本 R 是否一致
        for s in self.sequences:
            assert s.shape[0] == self.R, "All samples must have the same ROI count (R)."

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x_raw = self.sequences[idx].astype(np.float32)
        if self.x_scaler is not None:
            x_raw = self.x_scaler.transform(x_raw)
        with torch.no_grad():
            ts = torch.from_numpy(x_raw)          # [R, L]
            fc = torch.corrcoef(ts)               # [R, R]
            fc = torch.nan_to_num(fc)             # 数值稳健
            upper = fc[self.triu_i, self.triu_j]  # [E]
        x_pad = self.seq_pad[idx].astype(np.float32)
        if self.x_scaler is not None:
            x_pad = self.x_scaler.transform(x_pad)
        x = torch.from_numpy(x_pad.T.copy())
        # 回归标签标准化（若提供 y_scaler）
        y_reg = np.array([[self.labels_reg[idx]]], dtype=np.float32)
        if self.y_scaler is not None:
            y_reg = self.y_scaler.transform(y_reg)  # [[z]]
        y_reg = float(y_reg[0, 0])

        sample = {
            "x": x,                                        # [L, R]
            "fc": upper.to(torch.float32),                 # [E]
            "label_task": torch.tensor(self.labels_task[idx], dtype=torch.long),
            "label_reg": torch.tensor(self.labels_reg[idx], dtype=torch.float32),
            "label_reg_scale": torch.tensor(y_reg, dtype=torch.float32),
            "task_kind": "REG",
        }
        return sample

class SZDatasetStaticFC(Dataset):
    def __init__(self, subject_set, sequence_length, window_size, scaler=None):
        """
        参数:
            subject_set: [(label_8, label_770, full_series)]，full_series 形状 [R, L]
            sequence_length: 滑窗长度
            window_size: 滑窗步长
            scaler: 可选，按 ROI 标准化器（输入期望 [R, L]）
        """
        self.sequence_length = sequence_length
        self.window_size = window_size
        self.scaler = scaler

        self.sequences = []     # 每个元素: ndarray [R, sequence_length]
        self.labels = []

        for _, subj_label, full_series in subject_set:
            R, T = full_series.shape
            for start in range(0, T - sequence_length + 1, window_size):
                seq = full_series[:, start:start + sequence_length]
                if seq.shape[1] == sequence_length:
                    self.sequences.append(seq)
                    self.labels.append(subj_label)

        assert len(self.sequences) > 0, "Empty dataset!"
        # 记录 R 并预生成上三角索引（不含对角）
        # self.R = self.sequences[0].shape[0]
        self.R = 114
        tri = torch.triu_indices(self.R, self.R, offset=1)
        self.triu_i = tri[0]
        self.triu_j = tri[1]
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # x_raw: [R, L]
        x_raw = self.sequences[idx].astype(np.float32)

        # 可选标准化：按 ROI（对 [R, L] 进行 transform）
        if self.scaler is not None:
            x_raw = self.scaler.transform(x_raw)

        # ---- padding x 到 [L, 114] ----
        target_R = 114
        if x_raw.shape[0] < target_R:  # x_raw 形状是 [R, L]
            pad_R = target_R - x_raw.shape[0]
            x_raw = np.pad(x_raw, ((0, pad_R), (0, 0)), mode="constant")  # [114, L]

        # 模型时序输入：转为 [L, R]
        x = torch.from_numpy(x_raw.T.copy())  # [L, 114]
        # 用 torch 避免 numpy 往返
        with torch.no_grad():
            ts = torch.from_numpy(x_raw)          # [R, L]
            fc = torch.corrcoef(ts)               # [R, R]
            fc = torch.nan_to_num(fc)             # 数值稳健
            upper = fc[self.triu_i, self.triu_j]  # [E]

        sample = {
            "x": x,                                        # [L, R]
            "fc": upper.to(torch.float32),                 # [E]
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "task_kind": "SZ",
        }
        return sample


import numpy as np
from sklearn.utils import shuffle
class StandardScaler_1:
    """
    Standard the input
    """

    def __init__(self, mean: np.array, std: np.array):
        self.mean = mean
        self.std = std

    def transform(self, data: np.array):
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.array):
        return (data * self.std) + self.mean

def split_datasets(X, y, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"

    # 打乱数据
    X, y = shuffle(X, y, random_state=random_seed)
    total_samples = len(X)

    # 计算每个子集的样本数
    train_end = int(train_ratio * total_samples)
    val_end = train_end + int(val_ratio * total_samples)

    # 划分数据
    train_X, train_y = X[:train_end], y[:train_end]
    val_X, val_y = X[train_end:val_end], y[train_end:val_end]
    test_X, test_y = X[val_end:], y[val_end:]

    return train_X, train_y, val_X, val_y, test_X, test_y

import torch
from torch.utils.data import Dataset
import numpy as np

class SlidingWindowDataset(Dataset):
    def __init__(self, subject_set, sequence_length, window_size, scaler=None):
        """
        参数:
            subject_set: [(task_label, subj_label, full_series)]
            sequence_length: 窗口长度
            window_size: 滑动步长
            scaler: 可选，标准化器（如 StandardScaler）
        """
        self.sequence_length = sequence_length
        self.window_size = window_size
        self.scaler = scaler

        self.sequences = []
        self.labels = []
        self.subject_ids = []

        for task_label, subj_label, full_series in subject_set:
            T = full_series.shape[1]
            for start in range(0, T - sequence_length + 1, window_size):
                seq = full_series[:, start:start + sequence_length]
                if seq.shape[1] == sequence_length:
                    self.sequences.append(seq)
                    self.labels.append(task_label)
                    self.subject_ids.append(subj_label)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        # （可选）标准化：如果你的 scaler 期望按 ROI 做标准化，通常输入 [R, L]
        if self.scaler:
            seq = self.scaler.transform(seq)

        # 供时序模型使用的输入：转为 [L, R]
        x = torch.tensor(seq.T, dtype=torch.float32)

        task = self.labels[idx]
        y = self.subject_ids[idx]
        return (
            x,  # [L, R]
            torch.tensor(y, dtype=torch.long),  # subject id
            task
        )



class BoldTimeSeriesDataset(Dataset):
    def __init__(self, X, y, task):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.task = torch.tensor(task, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.task[idx]


from torch.utils.data import DataLoader

def create_dataloaders(train_X, train_y, train_task, val_X, val_y, val_task, batch_size=32):
    train_dataset = BoldTimeSeriesDataset(train_X.transpose(0, 2, 1), train_y, train_task)
    val_dataset = BoldTimeSeriesDataset(val_X.transpose(0, 2, 1), val_y, val_task)
    # test_dataset = BoldTimeSeriesDataset(test_X.transpose(0, 2, 1), test_y)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def create_dataloaders_test(val_X, val_y, val_task, batch_size=32):
    # train_dataset = BoldTimeSeriesDataset(train_X.transpose(0, 2, 1), train_y, train_task)
    val_dataset = BoldTimeSeriesDataset(val_X.transpose(0, 2, 1), val_y, val_task)
    # test_dataset = BoldTimeSeriesDataset(test_X.transpose(0, 2, 1), test_y)

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return val_loader