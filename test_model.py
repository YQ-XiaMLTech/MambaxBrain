import time
import torch
import torch.nn as nn
import os
import os.path as osp
from copy import deepcopy
from torch.optim import lr_scheduler
# import cv2
from torchvision import models
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import numpy as np
from thop import profile
import torch.optim as optim
import utils
from model.mamba_simple import *
import shutil
# from torch_geometric.loader import DataLoader
# from thop import profile
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# from process_dataset import split_data
from datetime import datetime
from config import config
from dataset.load_time_series_dataset import *
from dataset.pre_data import *
from sklearn.preprocessing import StandardScaler
from figure import *
# from torchvision import transforms
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, recall_score, accuracy_score
import seaborn as sns
from torch.utils.data import DataLoader
from utils import *
import torch
import numpy as np
from collections import defaultdict
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
import json
from pathlib import Path
from test_task import *
def fmt4(x):
    return "None" if x is None else f"{x:.4f}"
if __name__ == "__main__":
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    parser = argparse.ArgumentParser()
    # ---- 基本训练参数 ----
    parser.add_argument(
        "--run_dir",
        type=str,
        default="result_model/3task",
        help="existing save folder name under saves/, e.g. 20250707_...-lr0.0005"
    )
    parser.add_argument("--modes",type=json.loads, default=config.task_modes)
    parser.add_argument("--epoch", type=int, default=config.epoch)
    parser.add_argument("--fewshot_epoch", type=int, default=config.fewshot_epoch)
    parser.add_argument("--batch_size", type=int, default=config.batch_size)
    parser.add_argument("--lr", type=float, default=config.lr)

    # ---- 数据/窗口参数（以 args 为准）----
    parser.add_argument("--window_size", type=int, default=config.window_size)
    parser.add_argument("--sequence_length", type=int, default=config.sequence_length)
    # parser.add_argument("--SZ_seqlen", type=int, default=config.SZ_sequence_length)
    parser.add_argument("--dataset", type=str, default=config.dataset)
    # parser.add_argument("--SZ_dataset", type=str, default=config.SZ_dataset)
    parser.add_argument("--main_path", type=str, default=config.main_path)

    # 比例建议用 float（0~1）
    parser.add_argument("--train_ratio", type=float, default=config.train_ratio)
    # parser.add_argument("--SZ_train_ratio", type=float, default=config.SZ_train_ratio)
    parser.add_argument("--fewshot_train_ratio", type=float, default=config.fewshot_train_ratio)
    # parser.add_argument("--SZ_fewshot_train_ratio", type=float, default=config.SZ_fewshot_train_ratio)
    # ---- 设备/随机性 ----
    parser.add_argument("--gpu", action="store_true", default=True, help="use GPU")
    parser.add_argument("--no-gpu", dest="gpu", action="store_false", help="force CPU")
    parser.add_argument("--seed", type=int, default=config.seed)

    # ---- 多任务相关超参（可按需调整）----
    parser.add_argument("--d_state", type=int, default=150)
    parser.add_argument("--d_conv", type=int, default=16)
    parser.add_argument("--expand", type=int, default=16)
    parser.add_argument("--lambda_770", type=float, default=config.lambda_770, help="loss weight for 770-class task")
    parser.add_argument("--lambda_8", type=float, default=config.lambda_8, help="loss weight for 8-class task")
    parser.add_argument("--lambda_reg", type=float, default=config.lambda_reg, help="loss weight for regression task")
    # parser.add_argument("--lambda_sz", type=float, default=config.lambda_sz, help="loss weight for sz task")
    # parser.add_argument("--mt_cls_steps", type=int, default=1, help="how many cls steps before 1 reg step")
    # parser.add_argument("--mt_reg_every", type=int, default=1, help="do 1 reg step every N cls steps (alias)")

    # 其他标志
    parser.add_argument("--condition", type=str, default=config.condition, help="experiment tag/notes")

    args = parser.parse_args()

    # ---- 设备选择：只保留一个 device ----
    device = torch.device("cuda" if (torch.cuda.is_available() and args.gpu) else "cpu")
    args.device = device  # 以后以 args.device 为准
    config.device = device
    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    set_seed(args.seed)

    timestring = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    args.timestring = timestring

    run_name = "-".join([
        timestring,
        f"window{args.window_size}",
        f"seq{args.sequence_length}",
        f"bs{args.batch_size}",
        f"lr{args.lr}"
    ])

    save_root = Path(args.main_path) / "saves" / run_name
    save_root.mkdir(parents=True, exist_ok=True)

    # 在 save_root 下创建 train_task 子文件夹
    train_task_dir = save_root / "train_task"
    train_task_dir.mkdir(parents=True, exist_ok=True)  # 如果不存在就新建

    # 然后所有路径都改到 train_task_dir 下
    model_save_path = train_task_dir / "model_full.pth"
    model_state_dict = train_task_dir / "model_path.pth"
    fig_loss = train_task_dir / "loss.png"
    Per_Task_Accuracy_HCP_task = train_task_dir / "Per_Task_Accuracy_HCP_task.png"
    figure_save_path_confusion_matrix = train_task_dir / "confusion_matrix_cognitive.png"
    tsne_save_path = train_task_dir / "tsne_cls8.png"
    # ROC_figure = train_task_dir / "ROC_sz.png"
    Scatterplot_train = train_task_dir / "Scatterplot_train.png"
    Per_Task_Accuracy_Retest = train_task_dir / "Per_Task_Accuracy_Retest.png"
    Scatterplot_val = train_task_dir / "Scatterplot_val.png"
    # csv_summary = train_task_dir / "sequence_length_metrics_summary.csv"
    # reg_true_npy = train_task_dir / "test_true.npy"
    # reg_pred_npy = train_task_dir / "test_pred.npy"

    # 打印与落盘配置
    print("[Step 1] Configurations")
    print(f"device: {device}")
    # 展示 args（更真实地反映当前运行配置）
    for k, v in sorted(vars(args).items()):
        print(f"{k}: {v}")

    # 同时把 args 与（可选）config 的关键项落盘，方便复现实验
    with open(save_root / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)
    try:
        cfg_dump = {k: v for k, v in config.__dict__.items() if not k.startswith("_")}
        with open(save_root / "config_snapshot.json", "w") as f:
            json.dump(cfg_dump, f, indent=2, default=str)
    except Exception as e:
        print(f"[warn] failed to dump config snapshot: {e}")

    test_finger = (args.modes.get("fingerprint", "test") == "test")
    test_cogn = (args.modes.get("cognitive", "test") == "test")
    test_reg = (args.modes.get("rt", "test") == "test")
    # train_sz.sh = (args.modes["sz"] == "train")
    print("[Step 3] Initializing model/paths")
    print(f"save_root: {save_root}")
    print(f"model_save_path: {model_save_path}")
    print(f"loss_and_acc_fig: {fig_loss}")
    # print(f"regression_result_test_true: {reg_true_npy}")
    # print(f"regression_result_test_pred: {reg_pred_npy}")

    print("[Step 2] Preparing dataset...")
    if test_finger or test_cogn:
        train_set_cls, val_set_cls, Retest_set = load_and_split_subjects_train_cls(
            data_dir=args.dataset,
            map_sub_label='./dataset/subid_mapping.csv',
            condition=args.condition,
            train_ratio=args.train_ratio,
            seed=args.seed
        )
        train_mean_cls, train_std_cls = compute_roi_mean_std(
            train_set_cls)  # train_set_cls: [(label8, label770, full_series[R,L]), ...]
        train_scaler_cls = ROIStandardScaler(mean=train_mean_cls, std=train_std_cls)

        train_dataset_cls = ClsDatasetStaticFC(train_set_cls, sequence_length=args.sequence_length,
                                             window_size=args.window_size, scaler=train_scaler_cls)
        val_dataset_cls = ClsDatasetStaticFC(val_set_cls, sequence_length=args.sequence_length,
                                           window_size=args.window_size,
                                           scaler=train_scaler_cls)
        print(f"Train_cls dataset size: {len(train_dataset_cls)}")
        print(f"Val_cls   dataset size: {len(val_dataset_cls)}")
        train_loader_cls = DataLoader(train_dataset_cls, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                  pin_memory=True)
        val_loader_cls = DataLoader(val_dataset_cls, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                pin_memory=True)
        print("cls dataset over")
    else:
        train_loader_cls = None
        val_loader_cls = None
    if test_reg:
        train_set_reg, val_set_reg = load_and_split_subjects_train_reg(
            data_dir=args.dataset,
            map_sub_label='./dataset/subid_mapping.csv',
            condition=args.condition,
            train_ratio=args.train_ratio,
            seed=args.seed
        )

        train_mean_reg, train_std_reg = compute_roi_mean_std(
            train_set_reg)  # train_set_cls: [(label8, label770, full_series[R,L]), ...]
        train_scaler_reg = ROIStandardScaler(mean=train_mean_reg, std=train_std_reg)
        train_y_arr = np.array([lbl_reg for _, lbl_reg, _ in train_set_reg], dtype=np.float32).reshape(-1, 1)

        scaler_y = StandardScaler(with_mean=True, with_std=True)
        scaler_y.fit(train_y_arr)
        train_dataset_reg = RegDatasetStaticFC(train_set_reg, x_scaler=train_scaler_reg, y_scaler=scaler_y)
        val_dataset_reg = RegDatasetStaticFC(val_set_reg, x_scaler=train_scaler_reg, y_scaler=scaler_y)
        print(f"Train_reg dataset size: {len(train_dataset_reg)}")
        print(f"Val_reg   dataset size: {len(val_dataset_reg)}")
        train_loader_reg = DataLoader(train_dataset_reg, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                      pin_memory=True)
        val_loader_reg = DataLoader(val_dataset_reg, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                    pin_memory=True)
        print("reg dataset over")
    else:
        train_loader_reg = None
        val_loader_reg = None

    print("[Step 3] Using existing run_dir for testing only")
    save_root = Path(args.main_path) / "saves" / args.run_dir
    train_task_dir = save_root / "train_task"  # 你训练时就是存这里
    model_full_path = Path(args.run_dir) / "model_full.pth"  # best_state dict
    model_state_path = Path(args.run_dir) / "model_path.pth"  # 纯 state_dict（如果你存了）

    print(f"save_root: {save_root}")
    print(f"model_full_path: {model_full_path}")
    print(f"model_state_path: {model_state_path}")

    # ========= 2) 初始化模型 + 加载权重 =========
    print("[Step 5] Initializing model & loading checkpoint")
    model = MambaClassifierMultiHead(
        input_dim=114,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand
    ).to(device)

    if model_state_path.exists():
        state_dict = torch.load(model_state_path, map_location=device)
        model.load_state_dict(state_dict)
        print("[load] loaded state_dict from model_path.pth")
    else:
        # 否则从 model_full.pth 里取出 best_state["model_state"]
        ckpt = torch.load(model_full_path, map_location=device)
        if isinstance(ckpt, dict) and ("model_state" in ckpt):
            model.load_state_dict(ckpt["model_state"])
            print("[load] loaded model_state from model_full.pth (best_state dict)")
        else:
            raise ValueError(
                "Unrecognized checkpoint format. Expect model_state_path or a dict with key 'model_state'.")

    model.eval()
    print("[Step 6] Testing only...")

    val_metrics = val_epoch(
        model, device,
        cls_val_loader=val_loader_cls,  # 或 None
        reg_val_loader=val_loader_reg,  # 或 None
        # sz_val_loader=val_loader_SZ,  # SZ 通常单独评估
        train_finger=test_finger,
        train_cogn=test_cogn,
        lambda_770=args.lambda_770, lambda_8=args.lambda_8, lambda_reg=args.lambda_reg,
        y_scaler=scaler_y if test_reg else None,
    )

    val_acc_770 = val_metrics["acc_770"]  # None 或 数值
    val_acc_8 = val_metrics["acc_8"]  # None 或 数值
    val_r2 = val_metrics["r2"]  # None 或 数值

    print(
        f"fingerprint acc: {val_acc_770} | "
        f"cognitive acc: {val_acc_8} | "
        f"RT R2: {val_r2}"
    )

    print("test over")


