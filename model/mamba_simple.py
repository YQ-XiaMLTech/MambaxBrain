from mamba_ssm.modules.mamba_simple import Mamba
import torch
import torch.nn as nn
import torch.nn.functional as F

class CorrNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128):
        super(CorrNN, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.ReLU(inplace=True)

        # self.fc2 = nn.Linear(hidden_dim, output_dim)
        # self.bn2 = nn.BatchNorm1d(output_dim)
        # self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, input_dim)
        Returns: Tensor of shape (batch_size, output_dim)
        """
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act1(x)
        #
        # x = self.fc2(x)
        # x = self.bn2(x)
        # x = self.act2(x)

        return x
class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand):
        super().__init__()
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.mamba(x)
        return self.norm(x + residual)
        # return x

class MambaClassifierMultiHead(nn.Module):
    def __init__(self,
                 input_dim,            # R，ROI 数（也是 Mamba 输出通道）
                 d_state, d_conv, expand,
                 corr_feat_dim=6441,   # E，上三角边数
                 mamba_layers=1,
                 corr_hidden=128,       # CorrNN 输出维
                 fusion_hidden=128,    # 融合后的中间维度
                 num_classes_770=770,
                 num_classes_8=8,
                 dropout_p=0.5):
        super().__init__()

        self.blocks = nn.ModuleList([
            MambaBlock(input_dim, d_state, d_conv, expand)
            for _ in range(mamba_layers)
        ])
        # self.proj_R = nn.Linear(86, input_dim)

        self.corrnn = CorrNN(input_dim=corr_feat_dim, hidden_dim=corr_hidden)


        fused_dim = input_dim + corr_hidden
        self.dropout = nn.Dropout(dropout_p)
        self.fuse = nn.Sequential(
            nn.Linear(fused_dim, 2 * fused_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        # ---- 多任务 heads ----
        self.head_770 = nn.Linear(2 *fused_dim, num_classes_770)
        self.head_8 = nn.Linear(2 * fused_dim, num_classes_8)
        self.head_reg = nn.Linear(2 *fused_dim, 1)
        self.head_sz = nn.Linear(2 *fused_dim, 2)
        # self.head_reg = nn.Sequential(
        #     nn.Linear(fused_dim, 128),
        #     nn.GELU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(128, 1)
        # )
        # self.head_reg = nn.Sequential(
        #     nn.Linear(fused_dim, 2 * fused_dim),
        #     nn.GELU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(2 * fused_dim, 1)
        # )

    def encode_shared(self, x, fc):
        """
        x:  [B, L, R]
        fc: [B, E]
        -> feat: [B, fusion_hidden]
        """
        # Mamba 分支 + mean 池化（无 mask）
        h = x
        for blk in self.blocks:
            h = blk(h)              # [B, L, R]
        h = self.dropout(h)
        mamba_feat = h.mean(dim=1)  # [B, R]

        # CorrNN 分支
        corr_feat = self.corrnn(fc) # [B, corr_hidden]

        # 融合
        feat = torch.cat([mamba_feat, corr_feat], dim=1)  # [B, R + corr_hidden]
        feat = self.fuse(feat)                            # [B, fusion_hidden]
        return feat

    def forward(self, x, fc, task=None):
        """
        task:
          - 'cls' -> 返回 (out_770, out_8, feat)
          - 'reg' -> 返回 (out_reg, feat)
          - None  -> 返回 {'cls770', 'cls8', 'reg', 'feat'} 便于验证
        """
        feat = self.encode_shared(x, fc)

        if task == 'cls':
            out_770 = self.head_770(feat)  # [B, 770]
            out_8 = self.head_8(feat)  # [B, 8]
            return out_770, out_8, feat

        if task == 'reg':
            out_reg = self.head_reg(feat).squeeze(-1)  # [B]
            return out_reg, feat
        if task == 'sz':
            out_sz = self.head_sz(feat)
            return out_sz, feat

        # 默认：全返回（用于验证/可视化）
        return {
            "cls770": self.head_770(feat),
            "cls8": self.head_8(feat),
            "reg": self.head_reg(feat),
            "sz": self.head_sz(feat),
            "feat": feat,
        }


class MambaClassifierMultiHead_rag(nn.Module):
    def __init__(self,
                 input_dim,
                 d_state, d_conv, expand,
                 corr_feat_dim=6441,
                 mamba_layers=1,
                 corr_hidden=128,
                 num_classes_770=770,
                 num_classes_8=8,
                 dropout_p=0.5):
        super().__init__()

        self.blocks = nn.ModuleList([
            MambaBlock(input_dim, d_state, d_conv, expand)
            for _ in range(mamba_layers)
        ])

        self.corrnn = CorrNN(input_dim=corr_feat_dim, hidden_dim=corr_hidden)

        fused_dim = input_dim + corr_hidden
        self.dropout = nn.Dropout(dropout_p)
        self.head_770 = nn.Linear(fused_dim, num_classes_770)
        self.head_8 = nn.Linear(fused_dim, num_classes_8)
        # self.head_reg = nn.Linear(2 *fused_dim, 1)
        self.head_sz = nn.Linear(fused_dim, 2)
        # self.head_reg = nn.Linear(fused_dim, 1)
        self.head_reg = nn.Sequential(
            nn.Linear(fused_dim, 2*fused_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(2*fused_dim, 1)
        )

    def encode_shared(self, x, fc):
        """
        x:  [B, L, R]
        fc: [B, E]
        -> feat: [B, R + corr_hidden]
        """
        h = x
        for blk in self.blocks:
            h = blk(h)              # [B, L, R]
        h = self.dropout(h)
        mamba_feat = h.mean(dim=1)  # [B, R]

        corr_feat = self.corrnn(fc) # [B, corr_hidden]

        feat = torch.cat([mamba_feat, corr_feat], dim=1)  # [B, fused_dim]
        return feat

    def forward(self, x, fc, task=None):
        feat = self.encode_shared(x, fc)
        if task == 'cls':
            out_770 = self.head_770(feat)  # [B, 770]
            out_8 = self.head_8(feat)  # [B, 8]
            return out_770, out_8, feat

        if task == 'reg':
            out_reg = self.head_reg(feat).squeeze(-1)  # [B]
            return out_reg, feat
        if task == 'sz':
            out_sz = self.head_sz(feat)
            return out_sz, feat

        # 默认：全返回（用于验证/可视化）
        return {
            "cls770": self.head_770(feat),
            "cls8": self.head_8(feat),
            "reg": self.head_reg(feat),
            "sz": self.head_sz(feat),
            "feat": feat,
        }

import numpy as np
class MambaClassifier_sz(nn.Module):
    def __init__(self, input_dim, d_state, d_conv, expand, num_classes, num_layers=1, corr_feat_dim=3655, hidden_dim=64, tran_dim=512):
        super().__init__()
        self.blocks = nn.ModuleList([
            MambaBlock(input_dim, d_state, d_conv, expand) for _ in range(num_layers)
        ])
        self.corrnn = CorrNN(input_dim=corr_feat_dim, hidden_dim=hidden_dim)
        self.classifier = nn.Linear(hidden_dim+input_dim, num_classes)
        self.dropout = nn.Dropout(0.5)


    def forward(self, x):
        B, L, D = x.shape
        for blk in self.blocks:
            x = blk(x)
        out = self.dropout(x)
        mamba_feat = out.mean(dim=1)   # (B, D)，时序平均池化
        #
        fc_features = []
        for i in range(B):
            ts = x[i].detach().cpu().numpy().T  # shape: (D=114, L=100)
            fc = np.corrcoef(ts)

            # 提取上三角
            upper = fc[np.triu_indices(D, k=1)]
            fc_features.append(upper.astype(np.float32))

        fc_tensor = torch.tensor(np.stack(fc_features)).to(x.device)
        fc_feat = self.corrnn(fc_tensor)  # (B, D_corr)
        # feat = self.fusion(mamba_feat, corr_feat)
        feat = torch.cat([mamba_feat, fc_feat], dim=1)
        joint_feat = self.dropout(feat)
        joint_feat = self.classifier(joint_feat)  # (B, num_classes)
        return joint_feat, feat