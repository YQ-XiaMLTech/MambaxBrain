import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
from collections import defaultdict
import torch.nn.functional as F
from itertools import cycle
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from itertools import cycle
import torch
import torch.nn.functional as F
import math
# def train_epoch(model, cls_loader, reg_loader, optimizer, device,
#                 lambda_770=1.0, lambda_8=1.0, lambda_reg=1.0,
#                 reg_every=2,
#                 scheduler=None, scheduler_step_every_batch=False,
#                 y_scaler=None):  # ← 新增：回归标签的 StandardScaler（训练集拟合好的）
#
#     model.train()
#     reg_iter = cycle(reg_loader)
#
#     # 分类累计（逐样本加权）
#     total_loss_cls_sum = 0.0
#     total_cls_samples  = 0
#     correct_770 = correct_8 = 0
#     total_770 = total_8 = 0
#
#     # 回归累计（逐样本加权）
#     total_loss_reg_sum = 0.0
#     total_reg_samples  = 0
#     # 同时保留 scaled 与 original
#     y_true_reg_scaled, y_pred_reg_scaled = [], []
#     y_true_reg_orig,   y_pred_reg_orig   = [], []
#     all_tasks = []
#     for step, cls_batch in enumerate(cls_loader, start=1):
#         # ====== 分类一步 ======
#         x  = cls_batch["x"].to(device)
#         fc = cls_batch["fc"].to(device)
#         y770 = cls_batch["label_770"].to(device)
#         y8   = cls_batch["label_8"].to(device)
#         bs = x.size(0)
#
#         logits_770, logits_8, _ = model(x, fc, task='cls')
#         loss_770 = F.cross_entropy(logits_770, y770, reduction='mean')
#         loss_8v  = F.cross_entropy(logits_8,   y8,   reduction='mean')
#         loss_cls = lambda_770 * loss_770 + lambda_8 * loss_8v
#
#         optimizer.zero_grad()
#         loss_cls.backward()
#         optimizer.step()
#         if scheduler and scheduler_step_every_batch:
#             scheduler.step()
#
#         total_loss_cls_sum += loss_cls.item() * bs
#         total_cls_samples  += bs
#         _, pred_770 = torch.max(logits_770, 1)  # [B]
#         c770 = (pred_770 == y770).sum().item()  # 这一批里预测对的数量
#         t770 = y770.size(0)
#         _, pred_8 = torch.max(logits_8, 1)
#         c8 = (pred_8 == y8).sum().item()
#         t8 = y8.size(0)
#         correct_770 += c770
#         total_770 += t770
#         correct_8   += c8
#         total_8   += t8
#
#         # ====== 回归一步（按频率） ======
#         if step % reg_every == 0:
#             rb  = next(reg_iter)
#             xr  = rb["x"].to(device)
#             fcr = rb["fc"].to(device)
#             yr_scaled = rb["label_reg_scale"].to(device).float()   # 训练用 scaled 标签
#             yr_orig   = rb["label_reg"].to(device).float()       # 原始标签（未缩放）
#             label_task = rb["label_task"].to(device).float()
#             bsr = xr.size(0)
#
#             pred_scaled, _ = model(xr, fcr, task='reg')     # 模型输出是 scaled 空间
#             loss_reg = lambda_reg * F.l1_loss(pred_scaled, yr_scaled, reduction='mean')
#
#             optimizer.zero_grad()
#             loss_reg.backward()
#             optimizer.step()
#             if scheduler and scheduler_step_every_batch:
#                 scheduler.step()
#
#             total_loss_reg_sum += loss_reg.item() * bsr
#             total_reg_samples  += bsr
#
#             # ---- 收集 scaled 与 original 两份 ----
#             # scaled
#             y_true_reg_scaled.append(yr_scaled.detach().cpu())
#             y_pred_reg_scaled.append(pred_scaled.detach().cpu())
#
#             # original（需要 inverse_transform）
#             if y_scaler is not None:
#                 y_pred_np = pred_scaled.detach().cpu().numpy().reshape(-1, 1)
#                 y_pred_orig_np = y_scaler.inverse_transform(y_pred_np).reshape(-1)
#                 y_pred_reg_orig.append(torch.from_numpy(y_pred_orig_np))
#             else:
#                 # 没有 scaler 就直接当作原标度（不推荐，但做兜底）
#                 y_pred_reg_orig.append(pred_scaled.detach().cpu())
#
#             y_true_reg_orig.append(yr_orig.detach().cpu())
#             all_tasks.append(label_task.cpu())
#
#     # ====== 汇总 ======
#     avg_loss_cls = (total_loss_cls_sum / max(total_cls_samples, 1)) if total_cls_samples > 0 else 0.0
#     acc_770 = (correct_770 / max(total_770, 1)) if total_770 > 0 else 0.0
#     acc_8   = (correct_8   / max(total_8,   1)) if total_8   > 0 else 0.0
#
#     avg_loss_reg = (total_loss_reg_sum / max(total_reg_samples, 1)) if total_reg_samples > 0 else 0.0
#
#     y_true_reg_scaled = torch.cat(y_true_reg_scaled)
#     y_pred_reg_scaled = torch.cat(y_pred_reg_scaled)
#     y_true_reg_orig   = torch.cat(y_true_reg_orig)
#     y_pred_reg_orig   = torch.cat(y_pred_reg_orig)
#     task_ids = torch.cat(all_tasks)
#
#     return {
#         "loss_cls": avg_loss_cls,
#         "acc_770": acc_770,
#         "acc_8": acc_8,
#         "loss_reg": avg_loss_reg,
#         # scaled 空间（用于和训练 loss 对齐）
#         "y_true_reg_scaled": y_true_reg_scaled,
#         "y_pred_reg_scaled": y_pred_reg_scaled,
#         # 原始空间（用于画图/指标）
#         "y_true_reg": y_true_reg_orig,
#         "y_pred_reg": y_pred_reg_orig,
#         "task_ids": task_ids
#     }

def train_epoch(
    model,
    cls_loader=None,         # Fingerprinting + Cognitive 共用
    rt_loader=None,          # RT 回归
    # sz_loader=None,          # SZ 二分类（可选）
    optimizer=None,
    device=None,
    train_finger=True,       # 是否训练 770-way 指纹头
    train_cogn=True,
    # 分类/回归权重
    lambda_770=1.0, lambda_8=1.0, lambda_reg=1.0,
    # 交替频率
    rt_every=2,              # 每多少个 cls step 来一次 RT
    sz_every=2,              # 每多少个 cls step 来一次 SZ（可按需调整）
    # 调度器
    scheduler=None, scheduler_step_every_batch=False,
    # 标签缩放器（只用于 RT）
    y_scaler=None,
):
    model.train()

    # 判定哪些任务参与训练（依据 loader 是否提供）
    do_cls = (cls_loader is not None) and (train_finger or train_cogn)
    do_rt  = rt_loader is not None
    # do_sz  = sz_loader is not None

    if not (do_cls or do_rt):
        raise ValueError("At least one of cls_loader / rt_loader / sz_loader must be provided.")

    # 若需要交替，构造循环迭代器
    rt_iter = cycle(rt_loader) if do_rt and do_cls else None
    # sz_iter = cycle(sz_loader) if do_sz and do_cls else None

    # ===== 分类累计（逐样本加权）=====
    total_loss_cls_sum = 0.0
    total_770loss= 0.0
    total_8loss = 0.0
    total_cls_samples  = 0
    correct_770 = correct_8 = 0
    total_770 = total_8 = 0

    # ===== 回归累计 =====
    total_loss_reg_sum = 0.0
    total_reg_samples  = 0
    y_true_reg_scaled, y_pred_reg_scaled = [], []
    y_true_reg_orig,   y_pred_reg_orig   = [], []
    all_tasks = []  # 可选：收集 RT 的任务标签（如果有）

    # ===== SZ 累计 =====
    total_loss_sz_sum = 0.0
    total_sz_samples  = 0
    correct_sz = 0

    # ===== 训练主循环 =====
    # 情况 A：有 cls_loader -> 以 cls 为主循环，中途穿插 RT/SZ
    if do_cls:
        for step, cls_batch in enumerate(cls_loader, start=1):
            # ---- 分类一步 ----
            x   = cls_batch["x"].to(device)
            fc  = cls_batch["fc"].to(device)
            bs = x.size(0)

            logits_770, logits_8, _ = model(x, fc, task='cls')
            loss_parts = []
            # 770-way
            if train_finger:
                y770 = cls_batch["label_770"].to(device)
                loss_770 = F.cross_entropy(logits_770, y770, reduction='mean')
                loss_parts.append(lambda_770 * loss_770)

                # 统计指标
                _, pred_770 = torch.max(logits_770, 1)
                correct_770 += (pred_770 == y770).sum().item()
                total_770 += y770.size(0)

                total_770loss += loss_770.item() * bs
            # 8-way
            if train_cogn:
                y8 = cls_batch["label_8"].to(device)
                loss_8v = F.cross_entropy(logits_8, y8, reduction='mean')
                loss_parts.append(lambda_8 * loss_8v)

                # 统计指标
                _, pred_8 = torch.max(logits_8, 1)
                correct_8 += (pred_8 == y8).sum().item()
                total_8 += y8.size(0)
                total_8loss += loss_8v.item() * bs


            # 如果两个分类头都未启用，这一批就不做分类反传（极端情况）
            if len(loss_parts) > 0:
                loss_cls = sum(loss_parts)
                optimizer.zero_grad()
                loss_cls.backward()
                optimizer.step()
                if scheduler and scheduler_step_every_batch:
                    scheduler.step()

                total_loss_cls_sum += loss_cls.item() * bs
                total_cls_samples += bs  # 用 batch 计数作为归一化底数（与之前一致）

            # ---- 按频率穿插 RT ----
            if do_rt and (step % rt_every == 0):
                rb  = next(rt_iter)
                xr  = rb["x"].to(device)
                fcr = rb["fc"].to(device)
                yr_scaled = rb["label_reg_scale"].to(device).float()
                yr_orig   = rb["label_reg"].to(device).float()
                label_task = rb.get("label_task", None)
                bsr = xr.size(0)

                pred_scaled, _ = model(xr, fcr, task='reg')  # 输出在 scaled 空间
                loss_reg = lambda_reg * F.l1_loss(pred_scaled, yr_scaled, reduction='mean')

                optimizer.zero_grad()
                loss_reg.backward()
                optimizer.step()
                if scheduler and scheduler_step_every_batch:
                    scheduler.step()

                total_loss_reg_sum += loss_reg.item() * bsr
                total_reg_samples  += bsr

                y_true_reg_scaled.append(yr_scaled.detach().cpu())
                y_pred_reg_scaled.append(pred_scaled.detach().cpu())

                if y_scaler is not None:
                    y_pred_np = pred_scaled.detach().cpu().numpy().reshape(-1, 1)
                    y_pred_orig_np = y_scaler.inverse_transform(y_pred_np).reshape(-1)
                    y_pred_reg_orig.append(torch.from_numpy(y_pred_orig_np))
                else:
                    y_pred_reg_orig.append(pred_scaled.detach().cpu())

                y_true_reg_orig.append(yr_orig.detach().cpu())
                if label_task is not None:
                    all_tasks.append(label_task.detach().cpu())

            # ---- 按频率穿插 SZ ----
            # if do_sz and (step % sz_every == 0):
            #     sb = next(sz_iter)
            #     xs  = sb["x"].to(device)
            #     fcs = sb["fc"].to(device)
            #     ys  = sb["label"].to(device).long()
            #     bss = xs.size(0)
            #
            #     logits_sz, _ = model(xs, fcs, task='sz')    # 假设输出 logits_sz: [B, 2]
            #     loss_sz = lambda_sz * F.cross_entropy(logits_sz, ys, reduction='mean')
            #
            #     optimizer.zero_grad()
            #     loss_sz.backward()
            #     optimizer.step()
            #     if scheduler and scheduler_step_every_batch:
            #         scheduler.step()
            #
            #     total_loss_sz_sum += loss_sz.item() * bss
            #     total_sz_samples  += bss
            #     pred_sz = torch.argmax(logits_sz, dim=1)
            #     correct_sz += (pred_sz == ys).sum().item()

    # 情况 B：没有 cls_loader，只训练 RT 和/或 SZ
    else:
        # 统一把两个 loader zip/cycle；按较长迭代
        max_len = 0
        if do_rt: max_len = max(max_len, len(rt_loader))
        # if do_sz: max_len = max(max_len, len(sz_loader))
        if max_len == 0:
            raise ValueError("No data to iterate.")

        rt_it = cycle(rt_loader) if do_rt else None
        # sz_it = cycle(sz_loader) if do_sz else None

        for step in range(1, max_len + 1):
            # 交替/并行：这里给一个简洁策略——每步都各自走一小步（都在就都走）
            if do_rt:
                rb  = next(rt_it)
                xr  = rb["x"].to(device)
                fcr = rb["fc"].to(device)
                yr_scaled = rb["label_reg_scale"].to(device).float()
                yr_orig   = rb["label_reg"].to(device).float()
                label_task = rb.get("label_task", None)
                bsr = xr.size(0)

                pred_scaled, _ = model(xr, fcr, task='reg')
                loss_reg = lambda_reg * F.l1_loss(pred_scaled, yr_scaled, reduction='mean')

                optimizer.zero_grad()
                loss_reg.backward()
                optimizer.step()
                if scheduler and scheduler_step_every_batch:
                    scheduler.step()

                total_loss_reg_sum += loss_reg.item() * bsr
                total_reg_samples  += bsr

                y_true_reg_scaled.append(yr_scaled.detach().cpu())
                y_pred_reg_scaled.append(pred_scaled.detach().cpu())

                if y_scaler is not None:
                    y_pred_np = pred_scaled.detach().cpu().numpy().reshape(-1, 1)
                    y_pred_orig_np = y_scaler.inverse_transform(y_pred_np).reshape(-1)
                    y_pred_reg_orig.append(torch.from_numpy(y_pred_orig_np))
                else:
                    y_pred_reg_orig.append(pred_scaled.detach().cpu())

                y_true_reg_orig.append(yr_orig.detach().cpu())
                if label_task is not None:
                    all_tasks.append(label_task.detach().cpu())

            # if do_sz:
            #     sb = next(sz_it)
            #     xs  = sb["x"].to(device)
            #     fcs = sb["fc"].to(device)
            #     ys  = sb["label"].to(device).long()
            #     bss = xs.size(0)
            #
            #     logits_sz, _ = model(xs, fcs, task='sz')
            #     loss_sz = lambda_sz * F.cross_entropy(logits_sz, ys, reduction='mean')
            #
            #     optimizer.zero_grad()
            #     loss_sz.backward()
            #     optimizer.step()
            #     if scheduler and scheduler_step_every_batch:
            #         scheduler.step()
            #
            #     total_loss_sz_sum += loss_sz.item() * bss
            #     total_sz_samples  += bss
            #     pred_sz = torch.argmax(logits_sz, dim=1)
            #     correct_sz += (pred_sz == ys).sum().item()

    # ===== 汇总 =====
    # 分类
    if total_cls_samples > 0:
        avg_loss_cls = total_loss_cls_sum / total_cls_samples
        avg_loss_fing = total_770loss / total_cls_samples
        avg_loss_cogn = total_8loss / total_cls_samples
    else:
        avg_loss_cls = 0.0
        avg_loss_fing = 0.0
        avg_loss_cogn = 0.0
    acc_770 = (correct_770 / max(total_770, 1)) if train_finger else None
    acc_8 = (correct_8 / max(total_8, 1)) if train_cogn else None

    # 回归
    avg_loss_reg = (total_loss_reg_sum / max(total_reg_samples, 1)) if do_rt else 0.0
    if do_rt and len(y_true_reg_scaled) > 0:
        y_true_reg_scaled = torch.cat(y_true_reg_scaled)
        y_pred_reg_scaled = torch.cat(y_pred_reg_scaled)
        y_true_reg_orig   = torch.cat(y_true_reg_orig)
        y_pred_reg_orig   = torch.cat(y_pred_reg_orig)
        task_ids = torch.cat(all_tasks) if len(all_tasks) > 0 else None
    else:
        y_true_reg_scaled = y_pred_reg_scaled = y_true_reg_orig = y_pred_reg_orig = None
        task_ids = None

    # SZ
    # avg_loss_sz = (total_loss_sz_sum / max(total_sz_samples, 1)) if do_sz else 0.0
    # acc_sz      = (correct_sz / max(total_sz_samples, 1)) if do_sz else 0.0

    return {
        # 分类
        "loss_cls": avg_loss_cls,
        "loss_finger": avg_loss_fing,
        "loss_cogn": avg_loss_cogn,
        "acc_770": acc_770,
        "acc_8":   acc_8,
        # 回归
        "loss_reg": avg_loss_reg,
        "y_true_reg_scaled": y_true_reg_scaled,
        "y_pred_reg_scaled": y_pred_reg_scaled,
        "y_true_reg": y_true_reg_orig,
        "y_pred_reg": y_pred_reg_orig,
        "task_ids": task_ids,
        # SZ
        # "loss_sz": avg_loss_sz,
        # "acc_sz":  acc_sz,
    }

import torch
import torch.nn.functional as F
from collections import defaultdict

@torch.no_grad()
def val_epoch(
    model,
    device,
    cls_val_loader=None,         # Fingerprinting + Cognitive 共用，可为 None
    reg_val_loader=None,         # RT 回归验证，可为 None
    # sz_val_loader=None,          # SZ 验证（可选）
    # 任务开关（与 train 对齐）
    train_finger=True,           # 是否评估 770-way 指纹头
    train_cogn=True,             # 是否评估 8-way 任务解码头
    # loss 权重
    lambda_770=1.0,
    lambda_8=1.0,
    lambda_reg=1.0,
    # lambda_sz=1.0,
    # 仅用于 RT 反归一化
    y_scaler=None,
):
    model.eval()

    # ==================== 分类验证（770 / 8） ====================
    do_cls = (cls_val_loader is not None) and (train_finger or train_cogn)

    total_loss_cls_sum = 0.0
    total_cls_samples  = 0
    total_770_samples = 0
    total_8_samples = 0
    total_loss_770_sum = 0.0
    total_loss_8_sum = 0.0

    # 770 指标
    correct_770 = 0
    total_770   = 0
    val_true_770, val_pred_770 = [], []

    # 8 指标
    correct_8 = 0
    total_8   = 0
    val_true_8, val_pred_8 = [], []
    embeddings_8 = []
    labels_8 = []

    # 可选：以 8 类为 task_id 做 task-wise 统计
    task_total = defaultdict(int)
    task_correct = defaultdict(int)

    if do_cls:
        for batch in cls_val_loader:
            x  = batch["x"].to(device)             # [B, L, R]
            fc = batch["fc"].to(device)            # [B, E]
            bs = x.size(0)
            y770 = batch["label_770"].to(device)
            y8 = batch["label_8"].to(device)
            # 你的模型：返回 (logits_770, logits_8, features)
            # logits_770, logits_8, _ = model(x, fc, task='cls')
            with torch.no_grad():
                feat = model.encode_shared(x, fc)  # [B, H]
                _, logits_8, _ = model(x, fc, task='cls')  # 正常前向

            loss_parts = []
            if train_finger:
                logits_770, _, _ = model(x, fc, task='cls')
                loss_770 = F.cross_entropy(logits_770, y770, reduction='mean')
                loss_parts.append(lambda_770 * loss_770)

                _, pred_770 = torch.max(logits_770, 1)
                correct_770 += (pred_770 == y770).sum().item()
                total_770   += y770.size(0)
                val_true_770.extend(y770.cpu().tolist())
                val_pred_770.extend(pred_770.cpu().tolist())
                for i in range(y8.size(0)):
                    task_id = int(y8[i].item())
                    task_total[task_id] += 1
                    if pred_770[i] == y770[i]:
                        task_correct[task_id] += 1

                total_loss_770_sum += loss_770.item() * bs
                total_770_samples += bs

            if train_cogn:
                loss_8v = F.cross_entropy(logits_8, y8, reduction='mean')
                loss_parts.append(lambda_8 * loss_8v)

                _, pred_8 = torch.max(logits_8, 1)
                correct_8 += (pred_8 == y8).sum().item()
                total_8   += y8.size(0)
                val_true_8.extend(y8.cpu().tolist())
                val_pred_8.extend(pred_8.cpu().tolist())

                total_loss_8_sum += loss_8v.item() * bs
                total_8_samples += bs

                embeddings_8.append(feat.detach().cpu())
                labels_8.append(y8.detach().cpu())

            if len(loss_parts) > 0:
                loss_cls = sum(loss_parts)
                total_loss_cls_sum += loss_cls.item() * bs
                total_cls_samples  += bs

    avg_loss_cls = (total_loss_cls_sum / total_cls_samples) if total_cls_samples > 0 else None
    avg_loss_fing = (total_loss_770_sum / total_770_samples) if total_770_samples > 0 else None
    avg_loss_cogn = (total_loss_8_sum / total_8_samples) if total_8_samples > 0 else None
    acc_770 = (correct_770 / total_770) if (train_finger and total_770 > 0) else None
    acc_8   = (correct_8   / total_8)   if (train_cogn   and total_8   > 0) else None
    taskwise_accuracy = None
    if train_finger and len(task_total) > 0:
        taskwise_accuracy = {k: (task_correct[k] / task_total[k]) if task_total[k] > 0 else 0.0
                             for k in range(8)}
    # 拼接 embedding
    embeddings_8 = torch.cat(embeddings_8, dim=0) if len(embeddings_8) > 0 else None
    labels_8 = torch.cat(labels_8, dim=0) if len(labels_8) > 0 else None
    # ==================== 回归验证（RT） ====================
    do_reg = reg_val_loader is not None
    total_loss_reg_sum = 0.0
    total_reg_samples  = 0
    y_true_scaled_list, y_pred_scaled_list = [], []
    y_true_orig_list,   y_pred_orig_list   = [], []
    all_tasks = []

    if do_reg:
        for batch in reg_val_loader:
            x   = batch["x"].to(device)                    # [B, L*, R]
            fc  = batch["fc"].to(device)                   # [B, E]
            y_s = batch["label_reg_scale"].to(device).float().view(-1)  # [B]
            y_o = batch["label_reg"].to(device).float().view(-1)        # [B]
            label_task = batch.get("label_task", None)
            bs  = x.size(0)

            pred_s, _ = model(x, fc, task='reg')           # [B] or [B,1]
            pred_s = pred_s.view(-1)

            loss_reg = lambda_reg * F.l1_loss(pred_s, y_s, reduction='mean')
            total_loss_reg_sum += loss_reg.item() * bs
            total_reg_samples  += bs

            # scaled
            y_true_scaled_list.append(y_s.detach().cpu())
            y_pred_scaled_list.append(pred_s.detach().cpu())

            # original
            if y_scaler is not None:
                y_pred_np = pred_s.detach().cpu().view(-1, 1).numpy()
                y_pred_orig = y_scaler.inverse_transform(y_pred_np).reshape(-1)
                y_pred_orig_list.append(torch.from_numpy(y_pred_orig))
            else:
                y_pred_orig_list.append(pred_s.detach().cpu())

            y_true_orig_list.append(y_o.detach().cpu())
            if label_task is not None:
                all_tasks.append(label_task.cpu())

    avg_loss_reg = (total_loss_reg_sum / total_reg_samples) if total_reg_samples > 0 else None
    if total_reg_samples > 0:
        y_true_orig = torch.cat(y_true_orig_list).numpy()
        y_pred_orig = torch.cat(y_pred_orig_list).numpy()

        mae  = mean_absolute_error(y_true_orig, y_pred_orig)
        rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
        r2   = r2_score(y_true_orig, y_pred_orig)
        y_true_scaled = torch.cat(y_true_scaled_list).numpy()
        y_pred_scaled = torch.cat(y_pred_scaled_list).numpy()
        task_ids = torch.cat(all_tasks).numpy() if len(all_tasks) > 0 else None
    else:
        y_true_orig = y_pred_orig = y_true_scaled = y_pred_scaled = None
        mae = rmse = r2 = None
        task_ids = None

    # # ==================== SZ 二分类验证 ====================
    # do_sz = sz_val_loader is not None
    # total_loss_sz_sum = 0.0
    # total_sz_samples  = 0
    # correct_sz = 0
    # val_true_sz, val_pred_sz, val_prob_sz = [], [], []
    # if do_sz:
    #     for batch in sz_val_loader:
    #         xs  = batch["x"].to(device)
    #         fcs = batch["fc"].to(device)
    #         ys  = batch["label"].to(device).long()
    #         bs  = xs.size(0)
    #
    #         logits_sz, _ = model(xs, fcs, task='sz')   # [B, 2]
    #         loss_sz = lambda_sz * F.cross_entropy(logits_sz, ys, reduction='mean')
    #
    #         total_loss_sz_sum += loss_sz.item() * bs
    #         total_sz_samples  += bs
    #
    #         pred_sz = torch.argmax(logits_sz, dim=1)
    #         prob_sz = F.softmax(logits_sz, dim=1)[:, 1]  # SZ=1 的概率
    #         correct_sz += (pred_sz == ys).sum().item()
    #         val_true_sz.extend(ys.cpu().tolist())
    #         val_pred_sz.extend(pred_sz.cpu().tolist())
    #         val_prob_sz.extend(prob_sz.detach().cpu().tolist())
    #
    # avg_loss_sz = (total_loss_sz_sum / total_sz_samples) if total_sz_samples > 0 else None
    # acc_sz      = (correct_sz / total_sz_samples) if total_sz_samples > 0 else None

    # ==================== 打包返回 ====================
    return {
        # 分类（按开关）
        "loss_cls": avg_loss_cls,
        "loss_finger": avg_loss_fing,
        "loss_cogn":avg_loss_cogn,
        "acc_770":  acc_770,                # None 表示未评估
        "acc_8":    acc_8,                  # None 表示未评估
        "taskwise_accuracy": taskwise_accuracy,
        "val_true_770": val_true_770 if train_finger else None,
        "val_pred_770": val_pred_770 if train_finger else None,
        "val_true_8":   val_true_8   if train_cogn   else None,
        "val_pred_8":   val_pred_8   if train_cogn   else None,
        "embeddings_8": embeddings_8,
        "labels_8": labels_8,

        # 回归
        "loss_reg": avg_loss_reg,
        "mae": mae, "rmse": rmse, "r2": r2,
        "y_true_reg": y_true_orig,
        "y_pred_reg": y_pred_orig,
        "y_true_reg_scaled": y_true_scaled,
        "y_pred_reg_scaled": y_pred_scaled,
        "task_ids": task_ids,

        # # SZ
        # "loss_sz": avg_loss_sz,
        # "acc_sz":  acc_sz,
        # "val_true_sz": val_true_sz if do_sz else None,
        # "val_pred_sz": val_pred_sz if do_sz else None,
        # "val_prob_sz": val_prob_sz if do_sz else None,
    }


# @torch.no_grad()
# def val_epoch(model,
#               device,
#               cls_val_loader=None,     # 可为 None
#               reg_val_loader=None,     # 可为 None
#               lambda_770=1.0,
#               lambda_8=1.0,
#               lambda_reg=1.0,
#               y_scaler=None):
#
#     model.eval()
#
#     # ==================== 分类验证 ====================
#     total_loss_cls_sum = 0.0
#     total_cls_samples  = 0
#     correct_770 = correct_8 = 0
#     total_770   = total_8   = 0
#     task_total = defaultdict(int)
#     task_correct = defaultdict(int)
#     val_true_770, val_pred_770 = [], []
#     val_true_8,   val_pred_8   = [], []
#     all_tasks = []
#     if cls_val_loader is not None:
#         for batch in cls_val_loader:
#             x  = batch["x"].to(device)            # [B, L, R]
#             fc = batch["fc"].to(device)           # [B, E]
#             y770 = batch["label_770"].to(device)  # [B]
#             y8   = batch["label_8"].to(device)    # [B]
#             bs   = x.size(0)
#
#             logits_770, logits_8, _ = model(x, fc, task='cls')
#             loss_770 = F.cross_entropy(logits_770, y770, reduction='mean')
#             loss_8v  = F.cross_entropy(logits_8,   y8,   reduction='mean')
#             loss_cls = lambda_770 * loss_770 + lambda_8 * loss_8v
#
#             total_loss_cls_sum += loss_cls.item() * bs
#             total_cls_samples  += bs
#
#             # 770
#             _, pred_770 = torch.max(logits_770, 1)
#             correct_770 += (pred_770 == y770).sum().item()
#             total_770   += y770.size(0)
#             val_true_770.extend(y770.cpu().tolist())
#             val_pred_770.extend(pred_770.cpu().tolist())
#
#             # 8
#             _, pred_8 = torch.max(logits_8, 1)
#             correct_8 += (pred_8 == y8).sum().item()
#             total_8   += y8.size(0)
#             val_true_8.extend(y8.cpu().tolist())
#             val_pred_8.extend(pred_8.cpu().tolist())
#
#             # task-wise（以 8 类为 task_id）
#             for i in range(y770.size(0)):
#                 task_id = int(y8[i].item())
#                 task_total[task_id] += 1
#                 if pred_770[i] == y770[i]:
#                     task_correct[task_id] += 1
#
#     avg_loss_cls = (total_loss_cls_sum / total_cls_samples) if total_cls_samples > 0 else 0.0
#     acc_770 = (correct_770 / total_770) if total_770 > 0 else 0.0
#     acc_8   = (correct_8   / total_8)   if total_8   > 0 else 0.0
#     taskwise_accuracy = {
#         task: (task_correct[task] / task_total[task]) if task_total[task] > 0 else 0.0
#         for task in range(8)
#     }
#
#     # ==================== 回归验证 ====================
#     total_loss_reg_sum = 0.0
#     total_reg_samples  = 0
#     y_true_scaled_list, y_pred_scaled_list = [], []
#     y_true_orig_list,   y_pred_orig_list   = [], []
#
#     if reg_val_loader is not None:
#         for batch in reg_val_loader:
#             x   = batch["x"].to(device)                   # [B, L*, R]
#             fc  = batch["fc"].to(device)                  # [B, E]
#             y_s = batch["label_reg_scale"].to(device).float().view(-1)  # [B]
#             y_o = batch["label_reg"].to(device).float().view(-1)        # [B]
#             label_task = batch["label_task"].to(device).float()
#             bs  = x.size(0)
#
#             pred_s, _ = model(x, fc, task='reg')          # 可能 [B,1] 或 [B]
#             pred_s = pred_s.view(-1)                      # 对齐到 [B]
#
#             # 这里以 L1 为例，如需 MSE/SmoothL1 自行替换
#             loss_reg = lambda_reg * F.l1_loss(pred_s, y_s, reduction='mean')
#
#             total_loss_reg_sum += loss_reg.item() * bs
#             total_reg_samples  += bs
#
#             # scaled
#             y_true_scaled_list.append(y_s.detach().cpu())
#             y_pred_scaled_list.append(pred_s.detach().cpu())
#
#             # original（inverse_transform）
#             if y_scaler is not None:
#                 y_pred_np = pred_s.detach().cpu().view(-1, 1).numpy()
#                 y_pred_orig = y_scaler.inverse_transform(y_pred_np).reshape(-1)
#                 y_pred_orig_list.append(torch.from_numpy(y_pred_orig))
#             else:
#                 y_pred_orig_list.append(pred_s.detach().cpu())
#
#             y_true_orig_list.append(y_o.detach().cpu())
#             all_tasks.append(label_task.cpu())
#
#     avg_loss_reg = (total_loss_reg_sum / total_reg_samples) if total_reg_samples > 0 else 0.0
#
#     if len(y_true_orig_list) > 0:
#         y_true_orig = torch.cat(y_true_orig_list).numpy()
#         y_pred_orig = torch.cat(y_pred_orig_list).numpy()
#         mae  = mean_absolute_error(y_true_orig, y_pred_orig)
#         rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
#         r2   = r2_score(y_true_orig, y_pred_orig)
#     else:
#         y_true_orig = np.array([])
#         y_pred_orig = np.array([])
#         mae = rmse = r2 = 0.0
#
#     y_true_scaled = torch.cat(y_true_scaled_list).numpy() if len(y_true_scaled_list) > 0 else np.array([])
#     y_pred_scaled = torch.cat(y_pred_scaled_list).numpy() if len(y_pred_scaled_list) > 0 else np.array([])
#     task_ids = torch.cat(all_tasks) if len(all_tasks) > 0 else np.array([])
#     return {
#         # 分类
#         "loss_cls": avg_loss_cls,
#         "acc_770": acc_770,
#         "acc_8": acc_8,
#         "taskwise_accuracy": taskwise_accuracy,
#         "val_true_770": val_true_770,
#         "val_pred_770": val_pred_770,
#         "val_true_8": val_true_8,
#         "val_pred_8": val_pred_8,
#
#         # 回归
#         "loss_reg": avg_loss_reg,
#         "mae": mae,
#         "rmse": rmse,
#         "r2": r2,
#
#         # 回归预测/真值
#         "y_true_reg": y_true_orig,
#         "y_pred_reg": y_pred_orig,
#         "y_true_reg_scaled": y_true_scaled,
#         "y_pred_reg_scaled": y_pred_scaled,
#         "task_ids": task_ids
#     }

def train(model, train_loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch_x, batch_y, batch_task, batch_dfc in train_loader:
        batch_x, batch_y, batch_dfc = batch_x.to(device), batch_y.to(device), batch_dfc.to(device)

        optimizer.zero_grad()
        output,_ = model(batch_x, batch_dfc)  # shape: (B, num_classes)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)  # 累积总loss（乘batch size）
        _, predicted = torch.max(output, 1)
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)

    scheduler.step()
    train_loss = total_loss / total
    train_acc = correct / total
    return train_loss, train_acc


# 定义验证函数
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            output, _ = model(batch_x)
            loss = criterion(output, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(output, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

    val_loss = total_loss / total
    val_acc = correct / total

    return val_loss, val_acc

def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    test_true_list = []
    test_pred_list = []

    task_correct = defaultdict(int)
    task_total = defaultdict(int)
    with torch.no_grad():
        for batch_x, batch_y, batch_task, batch_dfc in test_loader:
            batch_x, batch_y, batch_task, batch_dfc = batch_x.to(device), batch_y.to(device), batch_task.to(device), batch_dfc.to(device)

            output,_ = model(batch_x, batch_dfc)
            loss = criterion(output, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(output, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

            test_true_list.extend(batch_y.cpu().tolist())
            test_pred_list.extend(predicted.cpu().tolist())

            for i in range(batch_y.size(0)):
                task_id = batch_task[i].item()
                task_total[task_id] += 1
                if predicted[i] == batch_y[i]:
                    task_correct[task_id] += 1

    test_loss = total_loss / total
    test_acc = correct / total

    taskwise_accuracy = {
        task: task_correct[task] / task_total[task] if task_total[task] > 0 else 0.0
        for task in range(8)  # 0 到 7 类
    }
    return test_loss, test_acc, test_true_list, test_pred_list, taskwise_accuracy


import torch
from functools import partial
from typing import Callable

def custom_amp_decorator(dec: Callable, cuda_amp_deprecated: bool):
    def decorator(*args, **kwargs):
        if cuda_amp_deprecated:
            kwargs["device_type"] = "cuda"
        return dec(*args, **kwargs)
    return decorator


if hasattr(torch.amp, "custom_fwd"): # type: ignore[attr-defined]
    deprecated = True
    from torch.amp import custom_fwd, custom_bwd # type: ignore[attr-defined]
else:
    deprecated = False
    from torch.cuda.amp import custom_fwd, custom_bwd

custom_fwd = custom_amp_decorator(custom_fwd, deprecated)
custom_bwd = custom_amp_decorator(custom_bwd, deprecated)
from collections import defaultdict
import torch

def test_with_embeddings(model, test_loader, criterion, device, classifier_attr="classifier"):
    """
    评估分类性能，并收集分类器前一层的 embedding（分类器输入）
    返回：
        test_loss, test_acc, y_true_list, y_pred_list, taskwise_acc,
        embeddings (Tensor[N, H]), labels (Tensor[N]), tasks (Tensor[N])
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    y_true_list, y_pred_list = [], []
    task_correct = defaultdict(int)
    task_total = defaultdict(int)

    # --- 准备 hook：抓分类器的输入作为 embedding ---
    if not hasattr(model, classifier_attr):
        raise AttributeError(f"model 没有属性 '{classifier_attr}'，请把 classifier_attr 改成你的分类头名字")
    clf_module = getattr(model, classifier_attr)

    embeddings_chunks = []
    def grab_classifier_input(module, inputs):
        # inputs[0].shape == (B, H)
        embeddings_chunks.append(inputs[0].detach().cpu())

    hook = clf_module.register_forward_pre_hook(grab_classifier_input)

    labels_chunks = []
    tasks_chunks  = []

    with torch.no_grad():
        for batch in test_loader:
            # 兼容 (x, y) 或 (x, y, task)
            if isinstance(batch, (list, tuple)):
                if len(batch) == 3:
                    batch_x, batch_y, batch_task = batch
                elif len(batch) == 2:
                    batch_x, batch_y = batch
                    batch_task = torch.zeros_like(batch_y)  # 若无 task，就用 0 占位
                else:
                    raise ValueError("test_loader 的 batch 形如 (x, y) 或 (x, y, task)")
            else:
                raise ValueError("期望 test_loader 返回 tuple/list")

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_task = batch_task.to(device)

            logits = model(batch_x)                       # 触发 hook，收集到本 batch 的 embedding
            loss = criterion(logits, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(logits, dim=1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

            # 记录标签/预测
            y_true_list.extend(batch_y.cpu().tolist())
            y_pred_list.extend(predicted.cpu().tolist())

            # task-wise 统计
            for i in range(batch_y.size(0)):
                t = int(batch_task[i].item())
                task_total[t] += 1
                if predicted[i] == batch_y[i]:
                    task_correct[t] += 1

            # 汇总可视化所需的标签与 task
            labels_chunks.append(batch_y.detach().cpu())
            tasks_chunks.append(batch_task.detach().cpu())

    # 移除 hook
    hook.remove()

    test_loss = total_loss / max(total, 1)
    test_acc = correct / max(total, 1)

    # 组织 task-wise accuracy（根据你数据里的 task 数量调整 range 上限）
    max_task_id = max(task_total.keys()) if len(task_total) > 0 else -1
    taskwise_accuracy = {
        t: (task_correct[t] / task_total[t] if task_total[t] > 0 else 0.0)
        for t in range(max_task_id + 1)
    }

    # 拼接所有 batch 的 embedding / label / task
    embeddings = torch.cat(embeddings_chunks, dim=0) if embeddings_chunks else torch.empty(0, 0)
    labels     = torch.cat(labels_chunks,     dim=0) if labels_chunks     else torch.empty(0, dtype=torch.long)
    tasks      = torch.cat(tasks_chunks,      dim=0) if tasks_chunks      else torch.empty(0, dtype=torch.long)

    return test_loss, test_acc, y_true_list, y_pred_list, taskwise_accuracy, embeddings, labels, tasks

@torch.no_grad()
def extract_feats_forward(model, loader, device):
    model.eval()
    all_feats, all_labels, all_tasks, all_logits = [], [], [], []
    for batch in loader:
        if len(batch) == 2:
            batch_x, batch_y = batch
            batch_task = None
        else:
            batch_x, batch_y, batch_task, batch_dfc = batch

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_dfc = batch_dfc.to(device)
        if batch_task is not None:
            batch_task = batch_task.to(device)

        logits, feat = model(batch_x, batch_dfc)
        all_feats.append(feat.detach().cpu())
        all_logits.append(logits.detach().cpu())
        all_labels.append(batch_y.detach().cpu())
        if batch_task is not None:
            all_tasks.append(batch_task.detach().cpu())

    feats  = torch.cat(all_feats, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    logits = torch.cat(all_logits, dim=0).numpy()
    tasks  = torch.cat(all_tasks, dim=0).numpy() if all_tasks else None
    return feats, labels, tasks, logits



# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device).eval()
#
# test_loss, test_acc, y_true, y_pred, task_acc, emb, lab, task = \
#     test_with_embeddings(model, val_loader, criterion, device, classifier_attr="classifier")
#
# print(emb.shape, lab.shape, task.shape)  # (N, H), (N,), (N,)
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# import numpy as np
#
# Z = TSNE(n_components=2, init="pca", random_state=42, perplexity=30).fit_transform(emb.numpy())
# colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}
#
# plt.figure(figsize=(8, 6))
# for c in np.unique(lab.numpy()):
#     m = (lab.numpy() == c)
#     plt.scatter(Z[m, 0], Z[m, 1], s=12, alpha=0.7, c=colors.get(int(c), None), label=f"Class {int(c)}")
# plt.legend(frameon=False)
# plt.title("2D Embedding (t-SNE) of Penultimate Layer")
# plt.xlabel("Dim 1");
# plt.ylabel("Dim 2");
# plt.grid(alpha=0.3, linestyle="--")
# plt.tight_layout();
# plt.show()
