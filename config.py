

# config.py
import os

class Config:
    task_modes = {
        "fingerprint": "test",
        "cognitive": "test",
        "rt": "test"
    }
    main_path = "./"
    condition = "natural"
    dataset = main_path + "dataset/processed_timeseries_114x1"
    # dataset = main_path + 'dataset/small'
    # SZ_dataset = main_path + "dataset/MathalonSZ_86x1"
    train_ratio = 0.7
    # SZ_train_ratio = 0.6
    fewshot_train_ratio = 1 - train_ratio
    # SZ_fewshot_train_ratio = 1 - SZ_train_ratio
    # val/test 比例你原来注释掉了，这里同样保持不显式设定
    window_size = 1
    sequence_length = 100
    # SZ_sequence_length = 70
    epoch = 60
    fewshot_epoch = 15
    batch_size = 64
    lr = 0.0001
    fewshot_lr = 0.0001
    seed = 123
    lambda_770 = 1.0
    lambda_8 = 1.0
    lambda_reg = 1.5
    # # 根据任务分配配置
    # if task == "activity":
    #     # 1) Task Activity Classification：8类
    #     dataset = main_path + "dataset/processed_timeseries_114x1"
    #     train_ratio = 0.6
    #     val_ratio = 0.2
    #     test_ratio = 0.2
    #     window_size = 100
    #     sequence_length = 100  # 为了 argparse 有这个字段
    #     delta = 100
    #     epoch = 30
    #     batch_size = 32
    #     lr = 0.0005
    #     seed = 9
    #
    #
    # elif task == "fingerprint":
    #     # 2) Fingerprinting：770人分类
    #     dataset = main_path + "dataset/processed_timeseries_114x1"
    #     train_ratio = 0.7
    #     # val/test 比例你原来注释掉了，这里同样保持不显式设定
    #     window_size = 1
    #     sequence_length = 100
    #     epoch = 100
    #     batch_size = 32
    #     lr = 0.0005
    #     seed = 888
    #
    #     # 为了保持字段齐全（某些代码里可能会访问到）
    #     val_ratio = None
    #     test_ratio = None
    #     delta = None
    #
    # elif task == "SZ":
    #     # 3) Schizophrenia Classification：二分类
    #     dataset = main_path + "dataset/MathalonSZ_86x1"
    #     train_ratio = 0.6
    #     window_size = 1
    #     sequence_length = 70
    #     epoch = 10
    #     batch_size = 128
    #     lr = 0.0008
    #     seed = 888
    #
    #     val_ratio = None
    #     test_ratio = None
    #     delta = None
    #
    # elif task == "rt":
    #     # 4) Reaction-Time Prediction：回归
    #     dataset = main_path + "dataset/processed_timeseries_114x1"
    #     train_ratio = 0.8
    #     window_size = 548
    #     sequence_length = 548
    #     epoch = 100
    #     batch_size = 64
    #     lr = 0.0001
    #     seed = 9
    #
    #     val_ratio = None
    #     test_ratio = None
    #     delta = None
    #
    # else:
    #     raise ValueError(f"Unknown task: {task}")

config = Config
