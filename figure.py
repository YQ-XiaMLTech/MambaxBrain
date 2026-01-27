import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def plot_roc_curve(y_true, y_prob, title="ROC Curve", save_path=None, dpi=300):
    """
    绘制 ROC 曲线 (二分类)
    :param y_true: 真实标签 (list or np.array)，0/1
    :param y_prob: 预测概率 (list or np.array)，取正类的概率
    :param title: 图标题
    :param save_path: 保存路径 (str)，None 表示不保存
    :param dpi: 保存图片的分辨率
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=dpi)
    plt.show()

def plot_loss_curves(
    y_lists,
    x_list,
    color_list,
    line_style_list,
    legend_list=None,
    legend_location="auto",
    legend_bbox_to_anchor=(0.515, 1.11),
    legend_ncol=3,
    legend_fontsize=15,
    fig_title=None,
    legend_loc="best",
    fig_x_label="time",
    fig_y_label="val",
    show_flag=True,
    save_flag=False,
    save_path=None,
    save_dpi=300,
    fig_title_size=20,
    fig_grid=False,
    marker_size=0,
    line_width=2,
    x_label_size=15,
    y_label_size=15,
    number_label_size=15,
    fig_size=(8, 6),
    x_ticks_set_flag=False,
    x_ticks=None,
    y_ticks_set_flag=False,
    y_ticks=None,
    tight_layout_flag=True,
    x_ticks_dress=None,
    y_ticks_dress=None
) -> None:
    """
    Draw a 2D plot of several lines
    :param y_lists: (list[list]) y value of lines, each list in which is one line. e.g., [[2,3,4,5], [2,1,0,-1], [1,4,9,16]]
    :param x_list: (list) x value shared by all lines. e.g., [1,2,3,4]
    :param color_list: (list) color of each line. e.g., ["red", "blue", "green"]
    :param line_style_list: (list) line style of each line. e.g., ["solid", "dotted", "dashed"]
    :param legend_list: (list) legend of each line, which CAN BE LESS THAN NUMBER of LINES. e.g., ["red line", "blue line", "green line"]
    :param legend_fontsize: (float) legend fontsize. e.g., 15
    :param fig_title: (string) title of the figure. e.g., "Anonymous"
    :param fig_x_label: (string) x label of the figure. e.g., "time"
    :param fig_y_label: (string) y label of the figure. e.g., "val"
    :param show_flag: (boolean) whether you want to show the figure. e.g., True
    :param save_flag: (boolean) whether you want to save the figure. e.g., False
    :param save_path: (string) If you want to save the figure, give the save path. e.g., "./test.png"
    :param save_dpi: (integer) If you want to save the figure, give the save dpi. e.g., 300
    :param fig_title_size: (float) figure title size. e.g., 20
    :param fig_grid: (boolean) whether you want to display the grid. e.g., True
    :param marker_size: (float) marker size. e.g., 0
    :param line_width: (float) line width. e.g., 1
    :param x_label_size: (float) x label size. e.g., 15
    :param y_label_size: (float) y label size. e.g., 15
    :param number_label_size: (float) number label size. e.g., 15
    :param fig_size: (tuple) figure size. e.g., (8, 6)
    :param x_ticks: (list) list of x_ticks. e.g., range(2, 21, 1)
    :param x_ticks_set_flag: (boolean) whether to set x_ticks. e.g., False
    :param y_ticks: (list) list of y_ticks. e.g., range(2, 21, 1)
    :param y_ticks_set_flag: (boolean) whether to set y_ticks. e.g., False
    :return:
    """
    assert len(list(y_lists[0])) == len(list(x_list)), "Dimension of y should be same to that of x"
    assert len(y_lists) == len(line_style_list) == len(color_list), "number of lines should be fixed"
    y_count = len(y_lists)
    plt.figure(figsize=fig_size)
    for i in range(y_count):
        plt.plot(x_list, y_lists[i], markersize=marker_size, linewidth=line_width, c=color_list[i], linestyle=line_style_list[i])
    plt.xlabel(fig_x_label, fontsize=x_label_size)
    plt.ylabel(fig_y_label, fontsize=y_label_size)
    if x_ticks_set_flag:
        if x_ticks_dress:
            plt.xticks(x_ticks, x_ticks_dress)
        else:
            plt.xticks(x_ticks)
    if y_ticks_set_flag:
        if y_ticks_dress:
            plt.yticks(y_ticks, y_ticks_dress)
        else:
            plt.yticks(y_ticks)
    plt.tick_params(labelsize=number_label_size)
    if legend_list:
        if legend_location == "fixed":
            plt.legend(legend_list, fontsize=legend_fontsize, bbox_to_anchor=legend_bbox_to_anchor, fancybox=True, ncol=legend_ncol, loc=legend_loc)
        else:
            plt.legend(legend_list, fontsize=legend_fontsize, loc=legend_loc)
    if fig_title:
        plt.title(fig_title, fontsize=fig_title_size)
    if fig_grid:
        plt.grid(True)
    if tight_layout_flag:
        plt.tight_layout()
    if save_flag:
        plt.savefig(save_path, dpi=save_dpi)
    if show_flag:
        plt.show()
    plt.clf()
    plt.close()

# def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=False,
#                           title="Confusion Matrix", save_path=None, dpi=300):
#     """
#     绘制混淆矩阵
#     :param y_true: 真实标签 (list or np.array)
#     :param y_pred: 预测标签 (list or np.array)
#     :param class_names: 类别名称 (list)，长度=类别数
#     :param normalize: 是否对混淆矩阵进行归一化 (按行百分比)
#     :param title: 图标题
#     :param save_path: 保存路径 (str)，None 表示不保存
#     :param dpi: 保存图片的分辨率
#     """
#     cm = confusion_matrix(y_true, y_pred)
#     if normalize:
#         cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
#         fmt = ".2f"
#     else:
#         fmt = "d"
#
#     plt.figure(figsize=(6, 5))
#     sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
#                 xticklabels=class_names if class_names is not None else "auto",
#                 yticklabels=class_names if class_names is not None else "auto",
#                 annot_kws={"size": 12})
#     plt.title(title, fontsize=16)
#     plt.ylabel("True Label", fontsize=12)
#     plt.xlabel("Predicted Label", fontsize=12)
#     plt.tight_layout()
#
#     if save_path:
#         plt.savefig(save_path, dpi=dpi)
#     plt.close()
#
def plot_confusion_matrix(y_true, y_pred, class_names=None,
                          title="Confusion Matrix", save_path=None, dpi=300):
    """
    绘制混淆矩阵（颜色按百分比归一化，但标注显示样本数）
    """
    cm_raw = confusion_matrix(y_true, y_pred)
    cm_norm = cm_raw.astype("float") / cm_raw.sum(axis=1, keepdims=True)  # 按行归一化

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=cm_raw, fmt="d", cmap="Blues",
                xticklabels=class_names if class_names is not None else "auto",
                yticklabels=class_names if class_names is not None else "auto",
                cbar_kws={"label": "Proportion"},
                annot_kws={"size": 12})

    # plt.title(title, fontsize=16)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()

# def plot_multi_task_regression(save, y_true, y_pred, task_ids, task_to_index,
#                                fig_size=(8, 6),
#                                title="Multi-task Regression Plot",
#                                title_size=18,
#                                x_label_size=15,
#                                y_label_size=15,
#                                number_label_size=12,
#                                marker_size=30,
#                                alpha=0.6,
#                                line_width=2,
#                                grid_flag=False,
#                                tight_layout_flag=True):
#     """
#     把所有任务的回归散点图画在一个图里，不同任务用颜色区分
#     """
#     index_to_task = {v: k for k, v in task_to_index.items()}
#
#     # 全局 R²
#     global_r2 = r2_score(y_true, y_pred)
#
#     plt.figure(figsize=fig_size)
#
#     # 不同任务用不同颜色
#     palette = sns.color_palette("Set2", len(index_to_task))
#
#     for task_id, task_name in index_to_task.items():
#         mask = (task_ids == task_id)
#         y_t = y_true[mask]
#         y_p = y_pred[mask]
#         sns.scatterplot(x=y_t, y=y_p, alpha=alpha, s=marker_size,
#                         label=task_name, color=palette[task_id])
#
#     # 理想预测线
#     min_val = min(y_true.min(), y_pred.min())
#     max_val = max(y_true.max(), y_pred.max())
#     plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=line_width)
#
#     # 统一 x/y 范围，保证方形
#     plt.xlim(min_val, max_val)
#     plt.ylim(min_val, max_val)
#     plt.gca().set_aspect('equal', adjustable='box')
#
#     plt.xlabel("True Values", fontsize=x_label_size)
#     plt.ylabel("Predicted Values", fontsize=y_label_size)
#     plt.title(f"{title}\nGlobal R² = {global_r2:.3f}", fontsize=title_size)
#     plt.xticks(fontsize=number_label_size)
#     plt.yticks(fontsize=number_label_size)
#     plt.legend(title="Task")
#
#     if grid_flag:
#         plt.grid(True)
#     if tight_layout_flag:
#         plt.tight_layout()
#
#     plt.savefig(save)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score

def plot_multi_task_regression(save, y_true, y_pred, task_ids, task_to_index,
                               task_colors=None,  # <--- 支持自定义 RGB 颜色
                               fig_size=(8, 6),
                               title="Multi-task Regression Plot",
                               title_size=18,
                               x_label_size=15,
                               y_label_size=15,
                               number_label_size=12,
                               marker_size=30,
                               alpha=0.6,
                               line_width=2,
                               grid_flag=False,
                               tight_layout_flag=True):
    """
    绘制多任务回归散点图，每个任务用不同颜色区分。
    支持 task_colors={"TASK": (R, G, B)}，RGB ∈ [0,255]
    """

    index_to_task = {v: k for k, v in task_to_index.items()}
    global_r2 = r2_score(y_true, y_pred)

    plt.figure(figsize=fig_size)

    # === 设置颜色 ===
    if task_colors is None:
        # 默认调色板（如果没传入）
        palette = sns.color_palette("Set2", len(index_to_task))
        task_colors = {name: palette[i] for i, name in index_to_task.items()}
    else:
        # 把 0-255 转成 0-1
        for k, v in task_colors.items():
            task_colors[k] = tuple(np.array(v) / 255.0)

    # === 绘制任务散点 ===
    for task_id, task_name in index_to_task.items():
        mask = (task_ids == task_id)
        y_t = y_true[mask]
        y_p = y_pred[mask]
        color = task_colors.get(task_name, (0.5, 0.5, 0.5))  # fallback 灰色
        sns.scatterplot(x=y_t, y=y_p, alpha=alpha, s=marker_size,
                        label=task_name, color=color)

    # === 理想线 ===
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=line_width)

    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.xlabel("True Values", fontsize=x_label_size)
    plt.ylabel("Predicted Values", fontsize=y_label_size)
    plt.title(f"{title}\nGlobal R² = {global_r2:.3f}", fontsize=title_size)
    plt.xticks(fontsize=number_label_size)
    plt.yticks(fontsize=number_label_size)
    plt.legend(title="Task")

    if grid_flag:
        plt.grid(True)
    if tight_layout_flag:
        plt.tight_layout()

    plt.savefig(save)
    plt.close()


