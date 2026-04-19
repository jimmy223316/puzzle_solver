"""
model.py — 神經網路架構：Residual CNN
============================================

架構設計理念：
- 使用卷積神經網路 (CNN) 提取盤面的空間局部特徵
- 加入殘差連接 (Residual Connections) 加速訓練、緩解梯度消失
- 全局平均池化 (Global Average Pooling) 讓模型對位置有一定的不變性
- 最終透過全連接層輸出 4 個動作的 Logits

維度流向（單筆樣本）：
  輸入:  (3, 12, 12)   ← [dx, dy, mask]
    ↓  Conv Block 1:  (3,12,12) → (64,12,12)
    ↓  Conv Block 2:  (64,12,12) → (128,12,12)
    ↓  Residual Block 1: (128,12,12) → (128,12,12)
    ↓  Conv Block 3:  (128,12,12) → (256,12,12)
    ↓  Residual Block 2: (256,12,12) → (256,12,12)
    ↓  Conv Block 4:  (256,12,12) → (256,12,12)
    ↓  GlobalAvgPool:  (256,12,12) → (256,)
    ↓  FC Head:        (256,) → (128,) → (4,)
  輸出:  (4,) Logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    殘差區塊：兩層 3×3 卷積 + BatchNorm + ReLU，帶 skip connection。
    
    維度不變：(B, C, H, W) → (B, C, H, W)
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        return: (B, C, H, W)
        """
        residual = x                          # (B, C, H, W) — 保存原始輸入
        out = F.relu(self.bn1(self.conv1(x)))  # (B, C, H, W)
        out = self.bn2(self.conv2(out))        # (B, C, H, W)
        out = F.relu(out + residual)           # (B, C, H, W) — 加回殘差
        return out


class PuzzleSolverCNN(nn.Module):
    """
    數字華容道解題 CNN 模型。
    
    設計特點：
    - 不使用下取樣 (stride=1, no pooling between blocks)
      因為 12×12 已經很小，且每個格子的資訊都很重要
    - 殘差連接確保深層梯度流暢
    - 全局平均池化將空間維度壓縮為 1D 特徵向量
    - 輕量 FC head 防止過擬合
    """

    def __init__(self, in_channels: int = 3, num_actions: int = 4):
        """
        Args:
            in_channels: 輸入通道數（預設 3 = dx + dy + mask）
            num_actions: 輸出動作數（預設 4 = 上下左右）
        """
        super().__init__()

        # ============ 卷積主幹 (Backbone) ============
        
        # Block 1: 淺層特徵提取
        # (B, 3, 12, 12) → (B, 64, 12, 12)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Block 2: 升維
        # (B, 64, 12, 12) → (B, 128, 12, 12)
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Residual Block 1: 深化特徵
        # (B, 128, 12, 12) → (B, 128, 12, 12)
        self.res_block1 = ResidualBlock(128)

        # Block 3: 再次升維
        # (B, 128, 12, 12) → (B, 256, 12, 12)
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Residual Block 2: 深化特徵
        # (B, 256, 12, 12) → (B, 256, 12, 12)
        self.res_block2 = ResidualBlock(256)

        # Block 4: 最終卷積
        # (B, 256, 12, 12) → (B, 256, 12, 12)
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # ============ 全局平均池化 ============
        # (B, 256, 12, 12) → (B, 256, 1, 1) → (B, 256)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # ============ 分類頭 (Classification Head) ============
        # (B, 256) → (B, 128) → (B, 4)
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_actions),
        )

        # 參數初始化
        self._init_weights()

    def _init_weights(self):
        """Kaiming 初始化：適合 ReLU 的權重初始化方法。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播。
        
        Args:
            x: (B, 3, 12, 12) — 編碼後的盤面特徵
        
        Returns:
            (B, 4) — 四個方向的動作 Logits
        """
        # 卷積主幹
        x = self.block1(x)        # (B, 3,   12, 12) → (B, 64,  12, 12)
        x = self.block2(x)        # (B, 64,  12, 12) → (B, 128, 12, 12)
        x = self.res_block1(x)    # (B, 128, 12, 12) → (B, 128, 12, 12)
        x = self.block3(x)        # (B, 128, 12, 12) → (B, 256, 12, 12)
        x = self.res_block2(x)    # (B, 256, 12, 12) → (B, 256, 12, 12)
        x = self.block4(x)        # (B, 256, 12, 12) → (B, 256, 12, 12)

        # 全局平均池化 + 展平
        x = self.global_pool(x)   # (B, 256, 12, 12) → (B, 256, 1, 1)
        x = x.view(x.size(0), -1) # (B, 256, 1, 1) → (B, 256)

        # 分類頭
        logits = self.head(x)     # (B, 256) → (B, 4)
        return logits


# =========================================================
# 模型資訊列印
# =========================================================
if __name__ == "__main__":
    model = PuzzleSolverCNN()

    # 計算參數量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("📐 PuzzleSolverCNN 模型架構")
    print("=" * 60)
    print(model)
    print(f"\n📊 參數統計：")
    print(f"   總參數量:   {total_params:>10,}")
    print(f"   可訓練參數: {trainable_params:>10,}")

    # 測試前向傳播
    dummy_input = torch.randn(4, 3, 12, 12)  # batch_size=4
    output = model(dummy_input)
    print(f"\n🧪 測試前向傳播：")
    print(f"   輸入維度: {list(dummy_input.shape)}")
    print(f"   輸出維度: {list(output.shape)}")
    print(f"   輸出範例: {output[0].detach().tolist()}")
