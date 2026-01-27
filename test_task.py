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
def freeze_backbone_except_heads(model, train_finger=False, train_cogn=False, train_reg=False, train_sz=False):
    # 全部冻结
    for p in model.parameters():
        p.requires_grad = False

    # 只打开需要微调的 head
    if train_finger:  # 表示fingerprint是测试任务 → 我们要微调它的head
        for p in model.head_770.parameters():
            p.requires_grad = True
    if train_cogn:
        for p in model.head_8.parameters():
            p.requires_grad = True
    if train_reg:
        for p in model.head_reg.parameters():
            p.requires_grad = True
    if train_sz:
        for p in model.head_sz.parameters():
            p.requires_grad = True

def test_task(args, model, optimizer, scheduler, save_root, device, train_finger=True,train_cogn=True, train_reg=True, train_sz=True):
    # 在 save_root 下创建 train_task 子文件夹
    test_task_dir = save_root / "test_task"
    test_task_dir.mkdir(parents=True, exist_ok=True)  # 如果不存在就新建

    # 然后所有路径都改到 train_task_dir 下
    model_save_path = test_task_dir / "model_full.pth"
    model_state_dict = test_task_dir / "model_path.pth"
    fig_loss = test_task_dir / "loss.png"
    Per_Task_Accuracy_HCP_task = test_task_dir / "Per_Task_Accuracy_HCP_task.png"
    figure_save_path_confusion_matrix = test_task_dir / "confusion_matrix_cognitive.png"
    ROC_figure = test_task_dir / "ROC_sz.png"
    Scatterplot_train = test_task_dir / "Scatterplot_train.png"
    Scatterplot_val = test_task_dir / "Scatterplot_val.png"

    if train_finger or train_cogn:
        train_set_cls, val_set_cls, Retest_set = load_and_split_subjects_train_cls(
            data_dir=args.dataset,
            map_sub_label='./dataset/subid_mapping.csv',
            condition=args.condition,
            train_ratio=args.fewshot_train_ratio,
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
            train_ratio=args.fewshot_train_ratio,
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
    if train_sz:
        train_set_SZ, val_set_SZ = load_and_split_subjects_train_SZ(
            data_dir=args.SZ_dataset,
            map_sub_label='./dataset/Schizophrenia_mapping.csv',
            condition=args.condition,
            train_ratio=args.SZ_fewshot_train_ratio,
            seed=args.seed
        )
        train_mean_SZ, train_std_SZ = compute_roi_mean_std(
            train_set_SZ)  # train_set_cls: [(label8, label770, full_series[R,L]), ...]
        train_scaler_SZ = ROIStandardScaler(mean=train_mean_SZ, std=train_std_SZ)

        train_dataset_SZ = SZDatasetStaticFC(train_set_SZ, sequence_length=args.SZ_seqlen,
                                             window_size=args.window_size, scaler=train_scaler_SZ)
        val_dataset_SZ = SZDatasetStaticFC(val_set_SZ, sequence_length=args.SZ_seqlen,
                                           window_size=args.window_size,
                                           scaler=train_scaler_SZ)
        print(f"Train_SZ dataset size: {len(train_dataset_SZ)}")
        print(f"Val_SZ   dataset size: {len(val_dataset_SZ)}")

        train_loader_SZ = DataLoader(train_dataset_SZ, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                     pin_memory=True)
        val_loader_SZ = DataLoader(val_dataset_SZ, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                   pin_memory=True)
        print("SZ dataset over")
    else:
        train_loader_SZ = None
        val_loader_SZ = None

    # 训练前调用：
    freeze_backbone_except_heads(model, train_finger, train_cogn, train_reg, train_sz)

    epoch_train_loss_cls_list, epoch_train_loss_finger_list, epoch_train_loss_cogntive_list, epoch_train_loss_reg_list, epoch_train_loss_sz_list = [], [], [], [], []
    epoch_val_loss_cls_list, epoch_val_loss_finger_list, epoch_val_loss_cogntive_list, epoch_val_loss_reg_list, epoch_val_loss_sz_list = [], [], [], [], []
    epoch_val_acc_770_list, epoch_val_acc_8_list, epoch_val_acc_sz_list = [], [], []
    epoch_val_r2_list = []

    norm_baseline_cls = None if (train_finger or train_cogn) else 0.0
    norm_baseline_reg = None if train_reg else 0.0
    norm_baseline_sz = None if train_sz else 0.0
    lambda_val_cls = getattr(args, "lambda_val_cls", 1.0) if (train_finger or train_cogn) else 0.0
    lambda_val_reg = getattr(args, "lambda_val_reg", 1.0) if train_reg else 0.0
    # lambda_val_sz = getattr(args, "lambda_val_sz", 1.0) if train_sz.sh else 0.0

    task_accuracy_series = {i: [] for i in range(8)}
    early_stopping_patience = 3
    best_score = float('inf')
    best_state = None
    no_improve_epochs = 0

    for epoch in range(args.fewshot_epoch):
        epoch_start_time = time.time()
        train_metrics = train_epoch(
            model,
            cls_loader=train_loader_cls,   # 指纹+任务解码的loader
            rt_loader=train_loader_reg,    # RT 的 loader
            # sz_loader=train_loader_SZ,                # SZ 不参与联合就传 None
            optimizer=optimizer,
            device=device,
            train_finger=train_finger, train_cogn=train_cogn,
            lambda_770=args.lambda_770, lambda_8=args.lambda_8, lambda_reg=args.lambda_reg,
            rt_every=2, sz_every=4,
            scheduler=scheduler, scheduler_step_every_batch=False,
            y_scaler=scaler_y if train_reg else None,
        )

        val_metrics = val_epoch(
            model, device,
            cls_val_loader=val_loader_cls,
            reg_val_loader=val_loader_reg,
            # sz_val_loader=val_loader_SZ,
            train_finger=train_finger, train_cogn=train_cogn,
            lambda_770=args.lambda_770, lambda_8=args.lambda_8, lambda_reg=args.lambda_reg,
            y_scaler=scaler_y if train_reg else None,
        )
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
            parts.append(lambda_val_cls * (val_loss_cls / norm_baseline_cls))
        if train_reg and (val_loss_reg is not None) and (norm_baseline_reg and norm_baseline_reg > 0):
            parts.append(lambda_val_reg * (val_loss_reg / norm_baseline_reg))
        # if train_sz.sh and (val_loss_sz is not None) and (norm_baseline_sz and norm_baseline_sz > 0):
        #     parts.append(lambda_val_sz * (val_loss_sz / norm_baseline_sz))

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

        msg = [f"[Epoch {epoch + 1:03d}/{args.fewshot_epoch}] time={elapsed:.2f}s"]

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
        print(
            f"Loaded best model from epoch {best_state['epoch'] + 1}, best_score={best_state['best_score']:.4f}")

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
            normalize=False,
            title="Cognitive TD - Confusion Matrix (8 classes)",
            save_path=figure_save_path_confusion_matrix,
        )
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
    #     y_true_sz = np.array(val_metrics["val_true_sz"])
    #     y_pred_sz = np.array(val_metrics["val_pred_sz"])
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
    #     y_true_sz = np.array(val_metrics["val_true_sz"])
    #     y_prob_sz = np.array(val_metrics["val_prob_sz"])  # 正类概率
    #
    #     plot_roc_curve(
    #         y_true=y_true_sz,
    #         y_prob=y_prob_sz,
    #         title="SZ Classification ROC Curve",
    #         save_path=ROC_figure
    #     )

    try:
        if train_scaler_cls is not None:
            return model, train_scaler_cls
    except NameError:
        pass  # 如果变量没定义会走这里

    return model, None

