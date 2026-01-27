import time
import torch
import torch.nn as nn
import os
import os.path as osp

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
from dataset.pre_data import StandardScaler_1, split_datasets, create_dataloaders, SlidingWindowDataset
# from sklearn.preprocessing import StandardScaler
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
def check_for_anomalies(ts_data, name=""):
    for i, ts in enumerate(ts_data):
        if np.all(ts == 0):
            print(f"⚠️ Subject {i} in {name} is all zeros.")
        elif np.isnan(ts).any():
            print(f"❌ Subject {i} in {name} contains NaN.")
        elif np.std(ts) < 1e-6:
            print(f"⚠️ Subject {i} in {name} has near-constant values.")

def tsne_plot(feats, labels, tasks=None, title="t-SNE of validation embeddings",
              save_path="tsne_val.png", max_points=5000):
    """
    feats: (N, D)
    labels: (N,)
    tasks: (N,) 或 None
    """
    N = feats.shape[0]
    idx = np.arange(N)
    if N > max_points:
        # 下采样加速，可选
        rng = np.random.default_rng(42)
        idx = rng.choice(N, size=max_points, replace=False)

    X = feats[idx]
    y = labels[idx]
    t = tasks[idx] if tasks is not None else None

    # 标准化 + 可选 PCA(先压到50维，有助于 t-SNE 稳定/加速)
    X = StandardScaler().fit_transform(X)
    if X.shape[1] > 50:
        X = PCA(n_components=50, random_state=42).fit_transform(X)

    tsne = TSNE(n_components=2, init="pca", learning_rate="auto",
                perplexity=min(30, X.shape[0]-1), random_state=42)
    X2 = tsne.fit_transform(X)

    plt.figure(figsize=(7, 6))
    # 不指定颜色（让 matplotlib 自选），二分类直接按 label 分两类画
    # 若带 task，可用不同 marker 辅助区分任务；否则单一 marker
    markers = ['o', 's', '^', 'D', 'P', 'X', '*', 'v']  # 最多8类task
    if t is None:
        # 仅按 label 分组
        classes = np.unique(y)
        for cls in classes:
            mask = (y == cls)
            plt.scatter(X2[mask, 0], X2[mask, 1], s=10, alpha=0.7, label=f"label={cls}")
        plt.legend()
    else:
        # 同时编码 label 与 task（label 用颜色，task 用 marker）
        classes = np.unique(y)
        unique_tasks = np.unique(t)
        for i, cls in enumerate(classes):
            for j, tk in enumerate(unique_tasks):
                mask = (y == cls) & (t == tk)
                if not np.any(mask):
                    continue
                marker = markers[j % len(markers)]
                plt.scatter(X2[mask, 0], X2[mask, 1], s=10, alpha=0.7,
                            marker=marker, label=f"y={cls}")
        # 简化图例：可选合并/去重
        handles, labels_leg = plt.gca().get_legend_handles_labels()
        # 去重
        uniq = dict(zip(labels_leg, handles))
        plt.legend(uniq.values(), uniq.keys(), fontsize=8, ncol=2)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[t-SNE] Saved to {save_path}")

# def map_sub_idx_to_label(csv_path, sub_idx_list):
#     try:
#         df = pd.read_csv(csv_path)
#         idx_to_label = dict(zip(df['sub_idx'], df['Label']))
#         label_list = [idx_to_label.get(i, -1) for i in sub_idx_list]
#         return label_list
#     except Exception as e:
#         print(f"❌ 映射失败: {e}")
#         return []


if __name__ == "__main__":
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=config.epoch, help="epoch")
    parser.add_argument("--batch_size", type=int, default=config.batch_size, help="batch_size")
    parser.add_argument('--lr', type=float, default=config.lr, help='learning rate, default=0.01')

    parser.add_argument("--window_size", type=int, default=config.window_size, help="window_size")
    parser.add_argument("--sequence_length", type=int, default=config.sequence_length, help="sequence_length")
    parser.add_argument("--condition", type=str, default=config.condition, help="type_classification")
    parser.add_argument("--main_path", type=str, default=config.main_path, help="main_path")
    parser.add_argument("--dataset", type=str, default=config.dataset, help="dataset")
    parser.add_argument("--train_ratio", type=int, default=config.train_ratio, help="important: train_ratio")
    # parser.add_argument("--test_ratio", type=int, default=config.test_ratio, help="important: test_ratio")
    # parser.add_argument("--val_ratio", type=int, default=config.val_ratio, help="important: val_ratio")
    parser.add_argument("--gpu", type=int, default=True, help="using gpu or not")
    parser.add_argument("--seed", type=int, default=config.seed, help="random seed")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")

    config.timestring = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    model_config_print = config.__dict__.copy()
    print("Loaded full config successfully:")
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[Step 1] Configurations")
    print("using: {}".format(config.device))
    for item in config.__dict__.items():
        if item[0][0] == "_":
            continue
        print("{}: {}".format(item[0], item[1]))
    device = config.device

    # ========= 这里开始：根据任务做最小化分支 =========
    # 只做“区分任务”的轻量 if，不改你的整体流程
    if config.task == "activity":
        task_name = "Task Activity Classification"
        num_classes = 8
        is_regression = False
        # loss_fn = torch.nn.CrossEntropyLoss()
        # label_dim = num_classes

    elif config.task == "fingerprint":
        task_name = "Fingerprinting"
        num_classes = 770
        is_regression = False
        # loss_fn = torch.nn.CrossEntropyLoss()
        # label_dim = num_classes

    elif config.task == "SZ":
        task_name = "Schizophrenia Classification"
        num_classes = 2
        is_regression = False
        # loss_fn = torch.nn.CrossEntropyLoss()
        # label_dim = num_classes

    elif config.task == "rt":
        task_name = "Reaction Time Prediction"
        is_regression = True
        num_classes = None
        # loss_fn = torch.nn.MSELoss()
        # label_dim = 1

    else:
        raise ValueError(f"Unknown task: {config.task}")

    print(f"[Step 2] Selected task: {task_name} | regression={is_regression} | num_classes={num_classes}")
    # ========= 分支结束 =========

    print("[Step 3] Initializing model")
    batch = 'batch_size' + str(args.batch_size)
    lr = 'lr' + str(args.lr)
    window_size = 'window_size' + str(args.window_size)
    task_tag = config.task if hasattr(config, "task") else task_name.replace(" ", "_")
    path = ('-').join([config.timestring, task_tag, window_size, batch, lr])
    main_save_path = osp.join(config.main_path, "saves", path)
    if not os.path.exists(main_save_path):
        os.makedirs(main_save_path)

    model_save_path = osp.join(main_save_path, "model_full.pth")
    model_state_dict = osp.join(main_save_path, "model_path.pth")
    figure_save_path_loss_and_acc = osp.join(main_save_path, "loss_and_acc.png")
    figure_save_path_confusion_matrix = osp.join(main_save_path, "confusion_matrix.png")
    # per_task_accuracy_vs_rest_length = os.path.join(main_save_path, "per_task_accuracy_vs_rest_length.png")
    # metrics_vs_rest_length = os.path.join(main_save_path, "metrics_vs_rest_length.png")
    sequence_length_vs_performance_and_overfit = os.path.join(main_save_path, "sequence_length_vs_performance_and_overfit.png")
    per_task_accuracy_bar = os.path.join(main_save_path, "per_task_accuracy_bar.png")
    sequence_length_window_size_search_results = os.path.join(main_save_path, "sequence_length_window_size_search_results.csv")
    sequence_length_metrics_summary = os.path.join(main_save_path, "sequence_length_metrics_summary.csv")
    regression_result_test_true = f"{main_save_path}/test_true.npy"
    regression_result_test_pred = f"{main_save_path}/test_pred.npy"
    # figure_attention_map = osp.join(main_save_path, "attention_map.png")
    # figure_regression_train_path = f"{main_save_path}/regression_train.png"
    # figure_regression_val_path = f"{main_save_path}/regression_val.png"
    # figure_regression_test_path = f"{main_save_path}/regression_test.png"
    print("main_save_path: {}".format(main_save_path))
    print("model_save_path: {}".format(model_save_path))
    print("figure_save_path_confusion_matrix: {}".format(figure_save_path_confusion_matrix))
    print("figure_save_path_loss_whole: {}".format(figure_save_path_loss_and_acc))
    print("regression_result_test_true: {}".format(regression_result_test_true))
    print("regression_result_test_pred: {}".format(regression_result_test_pred))
    # print("figure_regression_train_path: {}".format(figure_regression_train_path))
    # print("figure_regression_val_path: {}".format(figure_regression_val_path))
    # print("figure_regression_test_path: {}".format(figure_regression_test_path))

    print("[Step 2] Preparing dataset...")
    task_accuracy_series = {i: [] for i in range(8)}
    test_task_accuracy_series = {i: [] for i in range(8)}  # task_id: list of acc
    seq_lengths = [70]
    results = {}
    results_list = []
    seeds = [1, 526, 888]
    for seed in seeds:
        args.seed = seed
        f1s = []
        accs = []
        seq_len = []
        test_f1s = []
        test_accs = []
        for seq_length in seq_lengths:
            args.window_size = 1
            args.sequence_length = seq_length
            print("window size:", args.window_size,"sequence length:", args.sequence_length)
            train_set, val_set = load_and_split_subjects_train_sz(
                data_dir=args.dataset,
                map_sub_label='./dataset/Schizophrenia_mapping.csv',
                condition=args.condition,
                train_ratio=args.train_ratio,
                seed=args.seed
            )
            all_concat = np.concatenate([ts.flatten() for _, _, ts in train_set])
            mean = all_concat.mean()
            std = all_concat.std()
            train_scaler = StandardScaler_1(mean=mean, std=std)
            train_dataset = SlidingWindowDataset(train_set, sequence_length=args.sequence_length,
                                                 window_size=args.window_size, scaler=train_scaler)
            val_dataset = SlidingWindowDataset(val_set, sequence_length=args.sequence_length,
                                               window_size=args.window_size,
                                               scaler=train_scaler)
            print(f"Train dataset size: {len(train_dataset)}")
            print(f"Val   dataset size: {len(val_dataset)}")

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                      pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                    pin_memory=True)

            print("[Step 5] Initializing model")
            model = MambaClassifier_sz(input_dim=86, d_state=2, d_conv=2, expand=1, num_classes=2).to(device)
            # model = torch.load("./saves/20250902_214047_235622-window_size1-batch_size32-lr0.0005/model_full.pth", map_location="cpu")
            criterion = nn.CrossEntropyLoss()
            # optimizer = optim.Adam(model.parameters(), lr=args.lr)
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
            # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)
            print(model)

            # 计算模型的总参数数量
            total_params = sum(p.numel() for p in model.parameters())
            # 计算可训练的参数数量
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            print(f"Total parameters: {total_params}")
            print(f"Trainable parameters: {trainable_params}")

            print("[Step 6] Training...")
            epoch_train_loss_list = []
            epoch_train_acc_list = []
            epoch_val_loss_list = []
            epoch_val_acc_list = []
            # early stopping
            best_val_loss = float("inf")
            best_model_state = None
            patience = 1  # 当验证集损失在连续5个epoch中没有改善时停止训练
            counter = 0  # 追踪验证损失没有改善的epoch数量

            for epoch in range(args.epoch):
                epoch_start_time = time.time()
                # Training
                train_loss, train_acc = utils.train(model, train_loader, criterion, optimizer, scheduler, device)
                epoch_train_loss_list.append(train_loss)
                epoch_train_acc_list.append(train_acc)

                # Validation
                val_loss, val_acc, val_true_list, val_pred_list, taskwise_acc = utils.test(model, val_loader, criterion,
                                                                                           device)
                epoch_val_loss_list.append(val_loss)
                epoch_val_acc_list.append(val_acc)
                scheduler.step()

                # Logging
                print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
                      (epoch + 1, args.epoch, time.time() - epoch_start_time, \
                       train_acc, train_loss, val_acc, val_loss))

                # Check if early stopping condition is met
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    counter = 0
                    best_model_state = model.state_dict()  # 保存最佳模型的状态
                else:
                    counter += 1
                    if counter >= patience:
                        print(f"Early stopping triggered after {epoch + 1} epochs.")
                        epoch = epoch + 1
                        break  # Early stopping

            if best_model_state is not None:
                model.load_state_dict(best_model_state)

            torch.save(model, model_save_path)
            torch.save(model.state_dict(), model_state_dict)

            test_set = load_and_split_subjects_test_sz(
                data_dir=args.dataset,
                map_sub_label='./dataset/Schizophrenia_mapping.csv',
                condition=args.condition,
                train_ratio=0.0,
                seed=args.seed
            )
            test_dataset = SlidingWindowDataset(test_set, sequence_length=180,
                                               window_size=1,
                                               scaler=train_scaler)
            print(f"test   dataset size: {len(test_dataset)}")

            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                    pin_memory=True)
            _, _, _, test_pred_list, batch_task = utils.HR_test(model, test_loader, criterion, device)
            print("batch_task",batch_task,"\nHignResk",test_pred_list)
            feats, labels, tasks, logits = extract_feats_forward(model, val_loader, device)
            tsne_plot(feats, labels, tasks, title="t-SNE (val) - classifier input", save_path="tsne_val.png")

            epochs = range(1, len(epoch_train_loss_list) + 1)
            val_true_list = np.asarray(val_true_list)
            val_pred_list = np.asarray(val_pred_list)

            np.save(regression_result_test_true, val_true_list)
            np.save(regression_result_test_pred, val_pred_list)
            # print("epochs", epochs, "\nepoch_val_loss_list",epoch_val_loss_list, "\nepoch_val_acc_list", epoch_val_acc_list)

            plt.figure(figsize=(20, 6))
            plt.subplot(1, 2, 1)
            plt.plot(epochs, epoch_train_loss_list, 'b', label='Training loss')
            plt.plot(epochs, epoch_val_loss_list, 'r', label='Validation loss')  # 添加验证损失的曲线
            plt.title('Training and Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(epochs, epoch_train_acc_list, 'b', label='Training acc')
            plt.plot(epochs, epoch_val_acc_list, 'r', label='Validation acc')  # 添加验证损失的曲线
            plt.title('Training and Validation accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('accuracy')
            plt.legend()

            if figure_save_path_loss_and_acc is not None:
                plt.savefig(figure_save_path_loss_and_acc)
                plt.close()
            else:
                plt.show()

            overfit_score = epoch_train_acc_list[-1] - val_acc
            f1 = f1_score(val_true_list, val_pred_list, average='weighted')
            recall = recall_score(val_true_list, val_pred_list, average='weighted')
            accuracy = accuracy_score(val_true_list, val_pred_list)
            print("f1", f1, "\naccuracy", accuracy, "recall (weighted)", recall, "overfit", overfit_score)
            f1s.append(f1)
            accs.append(accuracy)
            seq_len.append(args.sequence_length)

            results = {
                'seed': args.seed,
                'window_size': args.window_size,
                'sequence_length': args.sequence_length,
                'mean': train_scaler.mean,
                'std': train_scaler.std,
                'f1': f1,
                'acc': accuracy,
                'recall': recall,
                'overfit': overfit_score,
                'epoch': epoch,
            }
            print("Per Task Accuracy:")
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

            for task_id in range(8):
                task_name = task_names[task_id]
                acc = taskwise_acc.get(task_id, 0.0)
                task_accuracy_series[task_id].append(acc)
                print(f"{task_name} (Task {task_id}): {acc:.4f}")
                task_labels.append(task_name)
                task_values.append(acc)
                results[task_name] = acc

            plt.figure(figsize=(10, 6))
            bars = plt.bar(task_labels, task_values)
            # 在柱状图上方标出数值
            for bar, value in zip(bars, task_values):
                plt.text(bar.get_x() + bar.get_width() / 2,
                         value + 0.01,  # 数值显示在柱子上方
                         f"{value:.4f}",
                         ha='center', va='bottom', fontsize=10)

            plt.ylim(0.3, 1.05)
            plt.title(f"Per Task Accuracy_HCP-task (win={args.window_size}, seq={args.sequence_length}, Seed={args.seed})")
            plt.ylabel("Accuracy")
            plt.xlabel("Task")
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plot_filename = f"Per_Task_Accuracy_HCP_task_win{args.window_size}_seq{args.sequence_length}_Seed{args.seed}.png"
            plot_path = os.path.join(main_save_path, plot_filename)
            plt.savefig(plot_path)

            results_list.append(results)

        plt.figure(figsize=(10, 5))
        plt.plot(seq_len, f1s, marker='o', label='F1-score')
        # plt.plot(seq_len, test_f1s, marker='d', label='Test F1-score')
        plt.plot(seq_len, accs, marker='s', label='Accuracy')
        # plt.plot(seq_len, test_accs, marker='*', label='Test Accuracy')
        plt.xlabel('seq_length')
        plt.ylabel('Score')
        plt.title(f'Effect of seq_length on Model Performance (Seed={args.seed})')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plot_filename = f'windowsize_vs_performance_seed{args.seed}.png'
        plot_path = os.path.join(main_save_path, plot_filename)
        plt.savefig(plot_path)

    df_results = pd.DataFrame(results_list)
    df_results.to_csv(sequence_length_window_size_search_results, index=False)
    metrics = ['f1', 'acc']
    # 按 sequence_length 分组，计算均值和标准差
    summary = df_results.groupby('sequence_length')[metrics].agg(['mean', 'std']).reset_index()
    # 展示结果
    print(summary)
    # 可选：保存为新 CSV
    summary.to_csv(sequence_length_metrics_summary, index=False)