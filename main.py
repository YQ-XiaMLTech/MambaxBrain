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
if __name__ == "__main__":
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    parser = argparse.ArgumentParser()
    # ---- 基本训练参数 ----
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

    train_finger = (args.modes["fingerprint"] == "train")
    train_cogn = (args.modes["cognitive"] == "train")
    train_reg = (args.modes["rt"] == "train")
    # train_sz.sh = (args.modes["sz"] == "train")
    print("[Step 3] Initializing model/paths")
    print(f"save_root: {save_root}")
    print(f"model_save_path: {model_save_path}")
    print(f"loss_and_acc_fig: {fig_loss}")
    # print(f"regression_result_test_true: {reg_true_npy}")
    # print(f"regression_result_test_pred: {reg_pred_npy}")

    print("[Step 2] Preparing dataset...")
    if train_finger or train_cogn:
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
    if train_reg:
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

    # args.dataset_Rest = args.main_path + 'dataset/processed_timeseries_114x1'
    # args.dataset_activity = args.main_path + 'dataset/Retest_114x1'
    # _, test_set = load_and_split_subjects(
    #     data_dir_Rest=args.dataset_Rest,
    #     data_dir_activity=args.dataset_activity,
    #     map_sub_label='./dataset/subid_mapping.csv',
    #     condition=args.condition,
    #     train_ratio=0,
    #     seed=args.seed
    # )
    # # test_set.extend(Retest_set)
    # test_dataset_cls = ClsDatasetStaticFC(test_set, sequence_length=args.sequence_length,
    #                                       window_size=args.window_size,
    #                                       scaler=train_scaler_cls)
    # test_loader_cls = DataLoader(test_dataset_cls, batch_size=args.batch_size, shuffle=False, num_workers=4,
    #                              pin_memory=True)

    # if train_sz.sh:
    #     train_set_SZ, val_set_SZ = load_and_split_subjects_train_SZ(
    #         data_dir=args.SZ_dataset,
    #         map_sub_label='./dataset/Schizophrenia_mapping.csv',
    #         condition=args.condition,
    #         train_ratio=args.SZ_train_ratio,
    #         seed=args.seed
    #     )
    #     train_mean_SZ, train_std_SZ = compute_roi_mean_std(
    #         train_set_SZ)  # train_set_cls: [(label8, label770, full_series[R,L]), ...]
    #     train_scaler_SZ = ROIStandardScaler(mean=train_mean_SZ, std=train_std_SZ)
    #
    #     train_dataset_SZ = SZDatasetStaticFC(train_set_SZ, sequence_length=args.SZ_seqlen,
    #                                          window_size=args.window_size, scaler=train_scaler_SZ)
    #     val_dataset_SZ = SZDatasetStaticFC(val_set_SZ, sequence_length=args.SZ_seqlen,
    #                                        window_size=args.window_size,
    #                                        scaler=train_scaler_SZ)
    #     print(f"Train_SZ dataset size: {len(train_dataset_SZ)}")
    #     print(f"Val_SZ   dataset size: {len(val_dataset_SZ)}")
    #
    #     train_loader_SZ = DataLoader(train_dataset_SZ, batch_size=args.batch_size, shuffle=True, num_workers=4,
    #                               pin_memory=True)
    #     val_loader_SZ = DataLoader(val_dataset_SZ, batch_size=args.batch_size, shuffle=False, num_workers=4,
    #                             pin_memory=True)
    #     print("SZ dataset over")
    # else:
    #     train_loader_SZ = None
    #     val_loader_SZ = None

    print("[Step 5] Initializing model")
    # checkpoint = torch.load(model_ckpt_path, map_location=device)
    model = MambaClassifierMultiHead(
        input_dim=114,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=3, verbose=True)

    # model.load_state_dict(checkpoint["model_state"])
    # optimizer.load_state_dict(checkpoint["optimizer_state"])
    # scheduler.load_state_dict(checkpoint["scheduler_state"])
    # best_score = checkpoint["best_score"]

    print(model)

    print("[Step 6] Training...")

    epoch_train_loss_cls_list,epoch_train_loss_finger_list,epoch_train_loss_cogntive_list, epoch_train_loss_reg_list, epoch_train_loss_sz_list = [], [], [], [], []
    epoch_val_loss_cls_list, epoch_val_loss_finger_list, epoch_val_loss_cogntive_list, epoch_val_loss_reg_list, epoch_val_loss_sz_list = [], [], [], [], []
    epoch_val_acc_770_list, epoch_val_acc_8_list, epoch_val_acc_sz_list = [], [], []
    epoch_val_r2_list = []

    norm_baseline_cls = None if (train_finger or train_cogn) else 0.0
    norm_baseline_reg = None if train_reg else 0.0
    # norm_baseline_sz = None if train_sz.sh else 0.0

    task_accuracy_series = {i: [] for i in range(8)}
    early_stopping_patience = 8
    best_score = float('inf')
    best_state = None
    no_improve_epochs = 0

    for epoch in range(args.epoch):
        epoch_start_time = time.time()

        # train_metrics = utils.train_epoch(model, train_loader_cls, train_loader_reg, optimizer, device, y_scaler=scaler_y)
        train_metrics = train_epoch(
            model,
            cls_loader=train_loader_cls,  # 或 None
            rt_loader=train_loader_reg,  # 或 None
            # sz_loader=train_loader_SZ,  # 通常 SZ 独立跑
            optimizer=optimizer,
            device=device,
            train_finger=train_finger,
            train_cogn=train_cogn,
            lambda_770=args.lambda_770, lambda_8=args.lambda_8, lambda_reg=args.lambda_reg,
            rt_every=2,
            sz_every=4,
            scheduler=scheduler, scheduler_step_every_batch=False,
            y_scaler=scaler_y if train_reg else None,
        )
        # val_metrics = utils.val_epoch(model, device, val_loader_cls, val_loader_reg,  y_scaler=scaler_y)
        val_metrics = val_epoch(
            model, device,
            cls_val_loader=val_loader_cls,  # 或 None
            reg_val_loader=val_loader_reg,  # 或 None
            # sz_val_loader=val_loader_SZ,  # SZ 通常单独评估
            train_finger=train_finger,
            train_cogn=train_cogn,
            lambda_770=args.lambda_770, lambda_8=args.lambda_8, lambda_reg=args.lambda_reg,
            y_scaler=scaler_y if train_reg else None,
        )
        # 取出训练指标（可能为 None）
        train_loss_cls = train_metrics["loss_cls"]
        train_loss_finger = train_metrics["loss_finger"]
        train_loss_cogn = train_metrics["loss_cogn"]
        train_loss_reg = train_metrics["loss_reg"]
        # train_loss_sz = train_metrics["loss_sz"]

        # 取出验证指标（可能为 None）
        val_loss_cls = val_metrics["loss_cls"]
        val_loss_finger = val_metrics["loss_finger"]
        val_loss_cogn = val_metrics["loss_cogn"]
        val_loss_reg = val_metrics["loss_reg"]
        # val_loss_sz = val_metrics["loss_sz"]

        val_acc_770 = val_metrics["acc_770"]  # None 或 数值
        val_acc_8 = val_metrics["acc_8"]  # None 或 数值
        # val_acc_sz = val_metrics["acc_sz"]
        val_r2 = val_metrics["r2"]  # None 或 数值

        # emb = val_metrics["embeddings_8"].numpy()
        # y = val_metrics["labels_8"].numpy()

        epoch_train_loss_cls_list.append(train_loss_cls)
        epoch_train_loss_finger_list.append(train_loss_finger)
        epoch_train_loss_cogntive_list.append(train_loss_cogn)
        epoch_train_loss_reg_list.append(train_loss_reg)
        # epoch_train_loss_sz_list.append(train_loss_sz)

        epoch_val_loss_cls_list.append(val_loss_cls)
        epoch_val_loss_finger_list.append(val_loss_finger)
        epoch_val_loss_cogntive_list.append(val_loss_cogn)
        epoch_val_loss_reg_list.append(val_loss_reg)
        # epoch_val_loss_sz_list.append(val_loss_sz)

        epoch_val_acc_770_list.append(val_acc_770)
        epoch_val_acc_8_list.append(val_acc_8)
        # epoch_val_acc_sz_list.append(val_acc_sz)
        epoch_val_r2_list.append(val_r2)

        # ------ 动态设置/更新归一化基线（仅对参与任务） ------
        if (train_finger or train_cogn) and (val_loss_cls is not None):
            if norm_baseline_cls is None:
                norm_baseline_cls = max(val_loss_cls, 1e-8)
        if train_reg and (val_loss_reg is not None):
            if norm_baseline_reg is None:
                norm_baseline_reg = max(val_loss_reg, 1e-8)
        # if train_sz.sh and (val_loss_sz is not None):
        #     if norm_baseline_sz is None:
        #         norm_baseline_sz = max(val_loss_sz, 1e-8)

        # ------ 组合验证损失（仅对存在的项做加权）------
        parts = []
        if (train_finger or train_cogn) and (val_loss_cls is not None) and (
                norm_baseline_cls and norm_baseline_cls > 0):
            parts.append(val_loss_cls / norm_baseline_cls)
        if train_reg and (val_loss_reg is not None) and (norm_baseline_reg and norm_baseline_reg > 0):
            parts.append(val_loss_reg / norm_baseline_reg)
        # if train_sz.sh and (val_loss_sz is not None) and (norm_baseline_sz and norm_baseline_sz > 0):
        #     parts.append(val_loss_sz / norm_baseline_sz)

        # 如果一个任务都没参与，防御性处理
        val_loss_total_norm = sum(parts) if len(parts) > 0 else 0.0

        scheduler.step(val_loss_total_norm)
        # ------ Save best & Early stopping ------
        if val_loss_total_norm < best_score:
            best_score = val_loss_total_norm
            best_state = {
                "epoch": epoch,
                "model_state": deepcopy(model.state_dict()),
                "optimizer_state": deepcopy(optimizer.state_dict()),
                "scheduler_state": deepcopy(scheduler.state_dict()),
                "best_score": best_score,
                "norm_baseline_cls": norm_baseline_cls,
                "norm_baseline_reg": norm_baseline_reg,
                # "norm_baseline_sz": norm_baseline_sz,
            }
            torch.save(best_state, model_save_path)
            torch.save(model.state_dict(), model_state_dict)
            no_improve_epochs = 0
            improved = True
            best_metrics = val_metrics
            best_train_metrics = train_metrics
        else:
            no_improve_epochs += 1
            improved = False

        # ------ 打印（只打印已开启/已计算的指标）------
        elapsed = time.time() - epoch_start_time

        msg = [f"[Epoch {epoch + 1:03d}/{args.epoch}] time={elapsed:.2f}s"]

        if train_loss_cls is not None:
            msg.append(f"train_cls={train_loss_cls:.4f}")
        if train_loss_reg is not None:
            msg.append(f"train_reg={train_loss_reg:.4f}")
        # if train_loss_sz is not None:
        #     msg.append(f"train_sz.sh={train_loss_sz:.4f}")

        if val_loss_cls is not None:
            msg.append(f"val_cls={val_loss_cls:.4f}")
        if val_loss_reg is not None:
            msg.append(f"val_reg={val_loss_reg:.4f}")
        # if val_loss_sz is not None:
        #     msg.append(f"val_sz={val_loss_sz:.4f}")

        if val_acc_770 is not None:
            msg.append(f"acc770={val_acc_770:.3f}")
        if val_acc_8 is not None:
            msg.append(f"acc8={val_acc_8:.3f}")
        # if val_acc_sz is not None:
        #     msg.append(f"accSZ={val_acc_sz:.3f}")
        if val_r2 is not None:
            msg.append(f"r2={val_r2:.3f}")

        msg.append(f"val_total={val_loss_total_norm:.4f}")
        if improved:
            msg.append("↑best")

        print(" | ".join(msg))

        # 早停（基于组合损失）
        if no_improve_epochs >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch + 1}. Best score={best_score:.4f}")
            break

    # 训练完成后，加载最优权重
    if best_state is not None:
        model.load_state_dict(best_state["model_state"])
        print(f"Loaded best model from epoch {best_state['epoch'] + 1}, best_score={best_state['best_score']:.4f}")

    if train_finger:
        epochs = range(1, len(epoch_train_loss_finger_list) + 1)
        plot_loss_curves(
            y_lists=[epoch_train_loss_finger_list, epoch_val_loss_finger_list],
            x_list=epochs,
            color_list=["tab:blue", "tab:red"],
            line_style_list=["solid", "dashed"],
            legend_list=["Fingerprinting Train loss", "Fingerprinting val loss"],
            fig_title="Fingerprinting - Training/Validation Loss (classification)",
            fig_x_label="Epoch",
            fig_y_label="Loss",
            fig_size=(8, 6),
            fig_grid=False,
            show_flag=False,
            save_flag=True,
            save_path=str(fig_loss).replace(".png", "_finger_cls.png"),
            save_dpi=300
        )
        # 2) Per Task Accuracy（使用 best_metrics["taskwise_accuracy"]）
        print("Per Task Accuracy (HCP-task):")
        task_names = {
            0: "Rest", 1: "WM", 2: "LANGUAGE", 3: "MOTOR",
            4: "SOCIAL", 5: "GAMBLING", 6: "RELATIONAL", 7: "EMOTION"
        }
        task_labels, task_values = [], []
        for tid in range(8):
            name = task_names.get(tid, f"Task{tid}")
            acc = best_metrics["taskwise_accuracy"].get(tid, 0.0)
            task_accuracy_series[tid].append(acc)
            print(f"{name} (Task {tid}): {acc:.4f}")
            task_labels.append(name)
            task_values.append(acc)

        plt.figure(figsize=(8, 6))
        bars = plt.bar(task_labels, task_values)
        for bar, value in zip(bars, task_values):
            plt.text(bar.get_x() + bar.get_width() / 2, value + 0.01,
                     f"{value:.4f}", ha='center', va='bottom', fontsize=10)
        plt.ylim(0.3, 1.05)
        plt.title(
            f"Per Task Accuracy - HCP-task")
        plt.ylabel("Accuracy")
        plt.xlabel("Task")
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(Per_Task_Accuracy_HCP_task)
        plt.close()

    if train_cogn:
        epochs = range(1, len(epoch_train_loss_cogntive_list) + 1)
        # 1) 分类损失（用 cls 的曲线）
        plot_loss_curves(
            y_lists=[epoch_train_loss_cogntive_list, epoch_val_loss_cogntive_list],
            x_list=epochs,
            color_list=["tab:blue", "tab:red"],
            line_style_list=["solid", "dashed"],
            legend_list=["Cogntive tasks Train loss", "Cogntive tasks val loss"],
            fig_title="Cogntive tasks - Training/Validation Loss (classification)",
            fig_x_label="Epoch",
            fig_y_label="Loss",
            fig_size=(8, 6),
            fig_grid=False,
            show_flag=False,
            save_flag=True,
            save_path=str(fig_loss).replace(".png", "_cogn_cls.png"),
            save_dpi=300
        )
        y_true8 = np.array(best_metrics["val_true_8"])
        y_pred8 = np.array(best_metrics["val_pred_8"])
        class_names_8 = ["Rest", "WM", "LANG", "MOTOR", "SOCIAL", "GAMBL", "RELAT", "EMOT"]
        plot_confusion_matrix(
            y_true=y_true8,
            y_pred=y_pred8,
            class_names=class_names_8,
            title="Cognitive TD - Confusion Matrix (8 classes)",
            save_path=figure_save_path_confusion_matrix,
        )

        # === t-SNE Visualization of Cognitive Task Embeddings ===
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        import seaborn as sns

        # 提取 embedding 与标签
        emb = best_metrics["embeddings_8"]
        labels = best_metrics["labels_8"]
        if emb is not None and labels is not None:
            emb_np = emb.numpy()
            labels_np = labels.numpy()

            # 执行 t-SNE 降维
            print("Running t-SNE on cognitive embeddings...")
            tsne = TSNE(
                n_components=2,
                perplexity=30,
                learning_rate=200,
                n_iter=1000,
                random_state=42,
                init="pca",
            )
            emb_2d = tsne.fit_transform(emb_np)

            # 类别名称（与混淆矩阵一致）
            class_names_8 = ["Rest", "WM", "LANG", "MOTOR", "SOCIAL", "GAMBL", "RELAT", "EMOT"]
            num_classes = len(class_names_8)

            # 可视化
            plt.figure(figsize=(8, 7))
            palette = sns.color_palette("tab10", num_classes)
            for i in range(num_classes):
                idx = labels_np == i
                plt.scatter(
                    emb_2d[idx, 0],
                    emb_2d[idx, 1],
                    s=10,
                    color=palette[i],
                    alpha=0.7,
                    label=class_names_8[i],
                )

            # plt.title("t-SNE visualization of cognitive task embeddings", fontsize=14)
            plt.xlabel("t-SNE Dimension 1", fontsize=12)
            plt.ylabel("t-SNE Dimension 2", fontsize=12)
            plt.legend(markerscale=2, fontsize=9, frameon=False, bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()

            plt.savefig(tsne_save_path, dpi=300, bbox_inches="tight")
            plt.close()

    if train_reg:
        epochs = range(1, len(epoch_train_loss_reg_list) + 1)
        # 1) 回归损失
        plot_loss_curves(
            y_lists=[epoch_train_loss_reg_list, epoch_val_loss_reg_list],
            x_list=epochs,
            color_list=["tab:blue", "tab:red"],
            line_style_list=["solid", "dashed"],
            legend_list=["RT Train loss", "RT val loss"],
            fig_title="RT - Training/Validation Loss (classification)",
            fig_x_label="Epoch",
            fig_y_label="Loss",
            fig_size=(8, 6),
            fig_grid=False,
            show_flag=False,
            save_flag=True,
            save_path=str(fig_loss).replace(".png", "_rt_reg.png"),
            save_dpi=300
        )
        # 2) 任务分组散点（训练/验证）
        task_to_index = {
            'STORY': 0, 'MATH': 1, 'TOM': 2, 'Random': 3, 'WM': 4, 'Rela': 5,
        }

        plot_multi_task_regression(
            save=Scatterplot_train,
            y_true=best_train_metrics["y_true_reg"],
            y_pred=best_train_metrics["y_pred_reg"],
            task_ids=best_train_metrics["task_ids"],
            task_to_index=task_to_index,
            title="Task-wise Regression Scatterplot (train)"
        )

        plot_multi_task_regression(
            save=Scatterplot_val,
            y_true=best_metrics["y_true_reg"],
            y_pred=best_metrics["y_pred_reg"],
            task_ids=best_metrics["task_ids"],
            task_to_index=task_to_index,
            title="Task-wise Regression Scatterplot (val)"
        )

    # if train_sz.sh:
    #     epochs = range(1, len(epoch_train_loss_sz_list) + 1)
    #     # 1) SZ 损失
    #     plot_loss_curves(
    #         y_lists=[epoch_train_loss_sz_list, epoch_train_loss_sz_list],
    #         x_list=epochs,
    #         color_list=["tab:blue", "tab:red"],
    #         line_style_list=["solid", "dashed"],
    #         legend_list=["SZ Train loss", "SZ val loss"],
    #         fig_title="SZ - Training/Validation Loss (classification)",
    #         fig_x_label="Epoch",
    #         fig_y_label="Loss",
    #         fig_size=(8, 6),
    #         fig_grid=False,
    #         show_flag=False,
    #         save_flag=True,
    #         save_path=str(fig_loss).replace(".png", "_sz.png"),
    #         save_dpi=300
    #     )
    #     # 2) 混淆矩阵
    #     y_true_sz = np.array(best_metrics["val_true_sz"])
    #     y_pred_sz = np.array(best_metrics["val_pred_sz"])
    #     class_names = ["Control", "SZ"]
    #     plot_confusion_matrix(
    #         y_true=y_true_sz,
    #         y_pred=y_pred_sz,
    #         class_names=class_names,
    #         normalize=False,
    #         title="SZ Classification Confusion Matrix",
    #         save_path=str(figure_save_path_confusion_matrix).replace(".png", "_sz.png"),
    #     )
    #     # 3) ROC
    #     y_true_sz = np.array(best_metrics["val_true_sz"])
    #     y_prob_sz = np.array(best_metrics["val_prob_sz"])  # 正类概率
    #
    #     plot_roc_curve(
    #         y_true=y_true_sz,
    #         y_prob=y_prob_sz,
    #         title="SZ Classification ROC Curve",
    #         save_path=ROC_figure
    #     )
    print("train over")

    model, scaler_cls = test_task(args, model, optimizer, scheduler,save_root, device, train_finger=(not train_finger), train_cogn=(not train_cogn), train_reg=(not train_reg), train_sz=False)
    if "train_scaler_cls" in locals() and train_scaler_cls is not None:
        final_scaler = train_scaler_cls
    elif scaler_cls is not None:
        final_scaler = scaler_cls
    else:
        final_scaler = None

    # test HCP-Retest
    args.dataset_Rest = args.main_path + 'dataset/processed_timeseries_114x1'
    args.dataset_activity = args.main_path + 'dataset/Retest_114x1'
    _, test_set = load_and_split_subjects(
        data_dir_Rest=args.dataset_Rest,
        data_dir_activity=args.dataset_activity,
        map_sub_label='./dataset/subid_mapping.csv',
        condition=args.condition,
        train_ratio=0,
        seed=args.seed
    )
    test_set.extend(Retest_set)
    test_dataset_cls = ClsDatasetStaticFC(test_set, sequence_length=args.sequence_length,
                                          window_size=args.window_size,
                                          scaler=final_scaler)
    test_loader_cls = DataLoader(test_dataset_cls, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                 pin_memory=True)
    # test_metrics = utils.val_epoch(model, device, test_loader_cls, y_scaler=scaler_y)
    test_metrics = val_epoch(
        model, device,
        cls_val_loader=test_loader_cls,  # 或 None
        reg_val_loader=None,  # 或 None
        # sz_val_loader=None,  # SZ 通常单独评估
        train_finger=True,
        train_cogn=True,
        lambda_770=args.lambda_770, lambda_8=args.lambda_8, lambda_reg=args.lambda_reg,
        y_scaler=scaler_y if train_reg else None,
    )
    test_loss_cls = test_metrics["loss_cls"]
    test_acc_770 = test_metrics["acc_770"]
    test_acc_8 = test_metrics["acc_8"]
    print(f"Retest Loss: {test_loss_cls:.4f}", f"｜Retest fingerprinting acc: {test_acc_770:.4f}",
          f"｜Retest cognitive acc: {test_acc_8:.4f}")
    print("Per Task Accuracy Retest:")
    task_names = {
        0: "Rest",
        1: "WM",
        2: "LANGUAGE",
        3: "MOTOR",
        4: "SOCIAL",
        5: "GAMBLING",
        6: "RELATIONAL",
        7: "EMOTION"
    }
    task_labels = []
    task_values = []
    # task_accuracy_series = {i: [] for i in range(8)}
    for task_id in range(8):
        task_name = task_names[task_id]
        acc = test_metrics["taskwise_accuracy"].get(task_id, 0.0)
        task_accuracy_series[task_id].append(acc)
        print(f"{task_name} (Task {task_id}): {acc:.4f}")
        task_labels.append(task_name)
        task_values.append(acc)

    plt.figure(figsize=(8, 6))
    bars = plt.bar(task_labels, task_values)
    # 在柱状图上方标出数值
    for bar, value in zip(bars, task_values):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 value + 0.01,  # 数值显示在柱子上方
                 f"{value:.4f}",
                 ha='center', va='bottom', fontsize=10)

    plt.ylim(0.3, 1.05)
    plt.title(f"Per Task Accuracy_HCP-task")
    plt.ylabel("Accuracy")
    plt.xlabel("Task")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plot_filename = f"Per_Task_Accuracy_HCP_retask_win.png"
    plt.savefig(Per_Task_Accuracy_Retest)


