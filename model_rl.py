"""
model_rl.py — Actor-Critic CNN（PPO 用）
==========================================

架構：共用 Backbone（與 model.py 完全相同），分叉出：
  - Actor Head：輸出 4 個動作的 Logits（繼承 BC 的 head 權重）
  - Critic Head：輸出標量 V(s)（隨機初始化）

維度流向（單筆樣本）：
  輸入 (3, 12, 12)
      │
      ▼ [共用 Backbone]
  (B, 256) ← GlobalAvgPool + Flatten
      │
      ├──────────────────────────────┐
      ▼                              ▼
  Actor Head                    Critic Head
  Linear(256→128)→ReLU           Linear(256→64)→ReLU
  Linear(128→4)  → Logits        Linear(64→1)  → V(s)

BC 權重載入策略：
  - Backbone (block1~4, res_block1/2, global_pool) → 直接複製
  - Actor Head (head.*) → 從 BC 的 head.* 完整複製
  - Critic Head → 隨機初始化（全新學習）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# 從 model.py 重用 ResidualBlock（保持架構一致）
class ResidualBlock(nn.Module):
    """殘差區塊：(B, C, H, W) → (B, C, H, W)，維度不變。"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class ActorCriticCNN(nn.Module):
    """
    Actor-Critic 模型，用於 PPO 訓練。

    方法：
      forward(x)     → (logits, value)  供訓練時計算 loss 用
      get_action(x)  → (action, log_prob, entropy, value)  供 rollout 採樣用
    """

    def __init__(self, in_channels: int = 3, num_actions: int = 4):
        super().__init__()

        # ============ 共用 Backbone（與 model.py 完全相同）============
        # (B, 3, 12, 12) → (B, 64, 12, 12)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        # (B, 64, 12, 12) → (B, 128, 12, 12)
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        # (B, 128, 12, 12) → (B, 128, 12, 12)
        self.res_block1 = ResidualBlock(128)
        # (B, 128, 12, 12) → (B, 256, 12, 12)
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        # (B, 256, 12, 12) → (B, 256, 12, 12)
        self.res_block2 = ResidualBlock(256)
        # (B, 256, 12, 12) → (B, 256, 12, 12)
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        # GlobalAvgPool: (B, 256, 12, 12) → (B, 256)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # ============ Actor Head（從 BC 的 head.* 繼承）============
        # (B, 256) → (B, 128) → (B, 4)
        self.actor_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),          # RL 微調時 Dropout 稍微降低，不太需要正則化
            nn.Linear(128, num_actions),
        )

        # ============ Critic Head（全新隨機初始化）============
        # (B, 256) → (B, 64) → (B, 1)
        self.critic_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        # 初始化 Critic Head（Orthogonal 初始化對 RL 更穩定）
        self._init_critic()

    def _init_critic(self):
        """對 Critic Head 使用 Orthogonal 初始化（RL 標準做法）。"""
        for m in self.critic_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0)
        # Actor 最後一層用較小的 gain，讓初始 policy 比較均勻
        last_linear = self.actor_head[-1]
        nn.init.orthogonal_(last_linear.weight, gain=0.01)
        nn.init.constant_(last_linear.bias, 0)

    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        """
        共用特徵提取。

        輸入:  (B, 3, 12, 12)
        輸出:  (B, 256)   ← GlobalAvgPool + Flatten
        """
        x = self.block1(x)        # → (B, 64,  12, 12)
        x = self.block2(x)        # → (B, 128, 12, 12)
        x = self.res_block1(x)    # → (B, 128, 12, 12)
        x = self.block3(x)        # → (B, 256, 12, 12)
        x = self.res_block2(x)    # → (B, 256, 12, 12)
        x = self.block4(x)        # → (B, 256, 12, 12)
        x = self.global_pool(x)   # → (B, 256, 1, 1)
        return x.view(x.size(0), -1)  # → (B, 256)

    def forward(self, x: torch.Tensor):
        """
        前向傳播（訓練時使用）。

        Args:
            x: (B, 3, 12, 12)

        Returns:
            logits: (B, 4)   ← Actor 輸出（未過 softmax）
            value:  (B, 1)   ← Critic 輸出 V(s)
        """
        features = self._backbone(x)          # (B, 256)
        logits   = self.actor_head(features)  # (B, 4)
        value    = self.critic_head(features) # (B, 1)
        return logits, value

    @torch.no_grad()
    def get_action(self, x: torch.Tensor,
                   legal_mask: torch.Tensor = None):
        """
        採樣動作（rollout 時使用，不計算梯度）。

        Args:
            x:          (B, 3, 12, 12)
            legal_mask: (B, 4) bool Tensor，True=合法
                        若為 None，所有動作都合法

        Returns:
            action:   (B,) int64
            log_prob: (B,) float32
            entropy:  (B,) float32
            value:    (B,) float32
        """
        features = self._backbone(x)
        logits   = self.actor_head(features)  # (B, 4)
        value    = self.critic_head(features).squeeze(-1)  # (B,)

        # 過濾非法動作：把非法動作的 logit 設為 -inf
        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask, float('-inf'))

        dist     = Categorical(logits=logits)
        action   = dist.sample()         # (B,)
        log_prob = dist.log_prob(action) # (B,)
        entropy  = dist.entropy()        # (B,)

        return action, log_prob, entropy, value

    def evaluate_actions(self, x: torch.Tensor,
                         actions: torch.Tensor,
                         legal_mask: torch.Tensor = None):
        """
        重新評估舊動作的 log_prob 和 entropy（PPO update 時使用）。

        Args:
            x:          (B, 3, 12, 12)
            actions:    (B,) int64，舊的採樣動作
            legal_mask: (B, 4) bool

        Returns:
            log_prob: (B,)
            entropy:  (B,)
            value:    (B,)
        """
        features = self._backbone(x)
        logits   = self.actor_head(features)
        value    = self.critic_head(features).squeeze(-1)

        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask, float('-inf'))

        dist     = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy  = dist.entropy()

        return log_prob, entropy, value


# =========================================================
# BC 權重載入函式
# =========================================================
def load_bc_weights(model: ActorCriticCNN,
                    bc_checkpoint_path: str,
                    device: torch.device) -> ActorCriticCNN:
    """
    從 BC 訓練的 best_model.pth 載入 Backbone + Actor 權重。

    映射規則（BC model.py → model_rl.py）：
      block1.*      → block1.*       ✅ 直接複製
      block2.*      → block2.*       ✅
      res_block1.*  → res_block1.*   ✅
      block3.*      → block3.*       ✅
      res_block2.*  → res_block2.*   ✅
      block4.*      → block4.*       ✅
      global_pool.* → global_pool.*  ✅
      head.0.*      → actor_head.0.* ✅  (Linear 256→128)
      head.3.*      → actor_head.3.* ✅  (Linear 128→4)
      [無對應]      → critic_head.*  🔄 保持隨機初始化

    Args:
        model:               ActorCriticCNN 實例
        bc_checkpoint_path:  BC 訓練存檔路徑（best_model.pth）
        device:              計算裝置

    Returns:
        model: 已載入權重的模型
    """
    print(f"📂 從 BC 模型載入權重：{bc_checkpoint_path}")
    checkpoint = torch.load(bc_checkpoint_path, map_location=device, weights_only=True)
    bc_state   = checkpoint["model_state_dict"]

    # 建立名稱映射：BC key → RL key
    key_mapping = {
        # Backbone（直接對應）
        "block1.0.weight": "block1.0.weight",
        "block1.1.weight": "block1.1.weight",
        "block1.1.bias":   "block1.1.bias",
        "block1.1.running_mean": "block1.1.running_mean",
        "block1.1.running_var":  "block1.1.running_var",
        "block1.1.num_batches_tracked": "block1.1.num_batches_tracked",
        "block2.0.weight": "block2.0.weight",
        "block2.1.weight": "block2.1.weight",
        "block2.1.bias":   "block2.1.bias",
        "block2.1.running_mean": "block2.1.running_mean",
        "block2.1.running_var":  "block2.1.running_var",
        "block2.1.num_batches_tracked": "block2.1.num_batches_tracked",
        "block3.0.weight": "block3.0.weight",
        "block3.1.weight": "block3.1.weight",
        "block3.1.bias":   "block3.1.bias",
        "block3.1.running_mean": "block3.1.running_mean",
        "block3.1.running_var":  "block3.1.running_var",
        "block3.1.num_batches_tracked": "block3.1.num_batches_tracked",
        "block4.0.weight": "block4.0.weight",
        "block4.1.weight": "block4.1.weight",
        "block4.1.bias":   "block4.1.bias",
        "block4.1.running_mean": "block4.1.running_mean",
        "block4.1.running_var":  "block4.1.running_var",
        "block4.1.num_batches_tracked": "block4.1.num_batches_tracked",
    }

    # ResidualBlock 的 key 動態映射
    for rb in ["res_block1", "res_block2"]:
        for part in ["conv1.weight", "bn1.weight", "bn1.bias",
                     "bn1.running_mean", "bn1.running_var", "bn1.num_batches_tracked",
                     "conv2.weight", "bn2.weight", "bn2.bias",
                     "bn2.running_mean", "bn2.running_var", "bn2.num_batches_tracked"]:
            key_mapping[f"{rb}.{part}"] = f"{rb}.{part}"

    # Actor Head 的映射（BC head.* → actor_head.*）
    actor_head_map = {
        "head.0.weight": "actor_head.0.weight",
        "head.0.bias":   "actor_head.0.bias",
        "head.3.weight": "actor_head.3.weight",
        "head.3.bias":   "actor_head.3.bias",
    }
    key_mapping.update(actor_head_map)

    # 取得 RL 模型的 state_dict，選擇性覆蓋
    rl_state = model.state_dict()
    loaded_keys   = []
    skipped_keys  = []

    for bc_key, rl_key in key_mapping.items():
        if bc_key in bc_state and rl_key in rl_state:
            if bc_state[bc_key].shape == rl_state[rl_key].shape:
                rl_state[rl_key] = bc_state[bc_key]
                loaded_keys.append(rl_key)
            else:
                skipped_keys.append(f"{bc_key} (shape mismatch)")
        else:
            skipped_keys.append(bc_key)

    model.load_state_dict(rl_state)

    print(f"  ✅ 成功載入 {len(loaded_keys)} 個權重 (Backbone + Actor Head)")
    if skipped_keys:
        print(f"  ⚠️  跳過 {len(skipped_keys)} 個 key：{skipped_keys[:5]}...")
    print(f"  🔄 Critic Head 保持隨機初始化")
    print(f"  📊 來自 Epoch {checkpoint.get('epoch', '?')}, "
          f"BC Val Acc: {checkpoint.get('val_acc', 0)*100:.2f}%")
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"裝置: {device}")

    model = ActorCriticCNN().to(device)

    # 測試前向傳播
    x = torch.randn(4, 3, 12, 12, device=device)
    logits, value = model(x)
    print(f"✅ forward()  logits: {list(logits.shape)}, value: {list(value.shape)}")

    # 測試 get_action
    legal_mask = torch.ones(4, 4, dtype=torch.bool, device=device)
    legal_mask[0, 0] = False  # 假設 batch 0 的動作 0 非法
    action, log_prob, entropy, val = model.get_action(x, legal_mask)
    print(f"✅ get_action() action: {action.tolist()}, log_prob: {log_prob.tolist()}")

    # 測試載入 BC 權重
    try:
        model = load_bc_weights(model, "best_model.pth", device)
    except FileNotFoundError:
        print("⚠️  找不到 best_model.pth，跳過載入測試。")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 總參數量: {total_params:,}")
