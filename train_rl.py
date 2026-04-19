"""
train_rl.py — PPO 強化學習訓練腳本
====================================

演算法：Proximal Policy Optimization (PPO) + GAE

訓練流程：
  1. 載入 best_model.pth → 初始化 ActorCriticCNN
  2. 重複 total_updates 次：
     a. 用當前 Policy rollout rollout_steps 步，收集 buffer
        - 每個 step 記錄 (state, action, log_prob, reward, value, done, legal_mask)
     b. 計算 GAE Advantage 和 Return
     c. 正規化 Advantage
     d. 切成 mini_batch，進行 K 次 PPO 更新
  3. 每隔 N 次更新印出統計，定期存檔最佳模型

多尺寸 Curriculum Learning：
  - 前 30% 更新：N ∈ {3, 4, 5}         easy   stage
  - 30 ~ 70%  ：N ∈ {4, 5, 6, 7}       medium stage
  - 後 70%    ：N ∈ {5, 6, 7, 8, 9}    hard   stage

使用方式：
    python train_rl.py                                    # 預設參數
    python train_rl.py --bc_model best_model.pth          # 指定 BC 模型
    python train_rl.py --total_updates 1000 --rollout_steps 4096
    python train_rl.py --max_time_hours 2.0               # 時間限制
"""

import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model_rl import ActorCriticCNN, load_bc_weights
from env_rl import PuzzleRLEnv


# =========================================================
# Rollout Buffer
# =========================================================
class RolloutBuffer:
    """
    儲存一次 rollout 的所有資料。

    每個欄位都是長度 rollout_steps 的 list，最後轉成 Tensor。
    """
    def __init__(self):
        self.states      = []  # numpy (3, 12, 12)
        self.actions     = []  # int
        self.log_probs   = []  # float
        self.rewards     = []  # float
        self.values      = []  # float
        self.dones       = []  # bool
        self.legal_masks = []  # numpy (4,) bool

    def add(self, state, action, log_prob, reward, value, done, legal_mask):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.legal_masks.append(legal_mask)

    def clear(self):
        self.__init__()

    def to_tensors(self, device):
        """將 buffer 資料轉成 PyTorch Tensor，回傳 dict。"""
        return {
            "states":      torch.tensor(np.array(self.states),
                                        dtype=torch.float32, device=device),  # (T, 3, 12, 12)
            "actions":     torch.tensor(self.actions,
                                        dtype=torch.long,    device=device),  # (T,)
            "log_probs":   torch.tensor(self.log_probs,
                                        dtype=torch.float32, device=device),  # (T,)
            "rewards":     torch.tensor(self.rewards,
                                        dtype=torch.float32, device=device),  # (T,)
            "values":      torch.tensor(self.values,
                                        dtype=torch.float32, device=device),  # (T,)
            "dones":       torch.tensor(self.dones,
                                        dtype=torch.float32, device=device),  # (T,)
            "legal_masks": torch.tensor(np.array(self.legal_masks),
                                        dtype=torch.bool,    device=device),  # (T, 4)
        }


# =========================================================
# GAE 計算
# =========================================================
def compute_gae(rewards: torch.Tensor,
                values:  torch.Tensor,
                dones:   torch.Tensor,
                last_value: float,
                gamma: float = 0.99,
                gae_lambda: float = 0.95) -> tuple:
    """
    計算 GAE (Generalized Advantage Estimation)。

    公式：
        δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
        A_t = δ_t + (γλ) * A_{t+1}         ← 反向遞推

    Args:
        rewards:    (T,) 每步獎勵
        values:     (T,) 每步的 V(s) 預測
        dones:      (T,) 是否結束（float: 0 或 1）
        last_value: 最後一步之後的 V(s_{T+1})（若已 done 則為 0）
        gamma:      折扣因子
        gae_lambda: GAE lambda

    Returns:
        advantages: (T,) 優勢估計（已正規化）
        returns:    (T,) TD 目標值，用於訓練 Critic
    """
    T = len(rewards)
    advantages = torch.zeros(T, device=rewards.device)

    last_adv = 0.0
    for t in reversed(range(T)):
        # next_value：若 t 是最後一步，用 last_value；否則用 values[t+1]
        next_value = values[t + 1] if t + 1 < T else last_value
        next_done  = dones[t]

        # TD 殘差
        delta = rewards[t] + gamma * next_value * (1 - next_done) - values[t]

        # GAE 反向遞推
        last_adv = delta + gamma * gae_lambda * (1 - next_done) * last_adv
        advantages[t] = last_adv

    returns = advantages + values  # V 目標值 = A + V_pred
    return advantages, returns


# =========================================================
# PPO 更新
# =========================================================
def ppo_update(model:         ActorCriticCNN,
               optimizer:     optim.Optimizer,
               buffer_data:   dict,
               advantages:    torch.Tensor,
               returns:       torch.Tensor,
               ppo_epochs:    int   = 4,
               mini_batch_size: int = 256,
               clip_epsilon:  float = 0.2,
               value_coef:    float = 0.5,
               entropy_coef:  float = 0.02,
               max_grad_norm: float = 0.5) -> dict:
    """
    PPO 梯度更新。

    損失函數：
      L = -L_CLIP + value_coef * L_VALUE - entropy_coef * H[π]

      L_CLIP  = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
      L_VALUE = MSE(V_pred, V_target)
      H[π]    = -E[log π(a|s)]（越大代表 policy 越均勻，更有探索性）

    Returns:
        stats: dict 含 pg_loss, value_loss, entropy_loss, total_loss
    """
    T     = len(buffer_data["states"])
    idx   = torch.randperm(T)   # 打亂順序

    # 正規化 Advantage（提升訓練穩定性）
    adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    stats = {"pg_loss": 0, "value_loss": 0, "entropy": 0, "total_loss": 0}
    n_batches = 0

    for _ in range(ppo_epochs):
        for start in range(0, T, mini_batch_size):
            b_idx = idx[start: start + mini_batch_size]

            b_states      = buffer_data["states"][b_idx]         # (B, 3, 12, 12)
            b_actions     = buffer_data["actions"][b_idx]        # (B,)
            b_old_lp      = buffer_data["log_probs"][b_idx]      # (B,) 舊的 log_prob
            b_legal_masks = buffer_data["legal_masks"][b_idx]    # (B, 4)
            b_adv         = adv_norm[b_idx]                      # (B,)
            b_returns     = returns[b_idx]                       # (B,)

            # 用新的 policy 重新評估這批動作
            new_log_prob, entropy, new_value = model.evaluate_actions(
                b_states, b_actions, b_legal_masks
            )

            # PPO 比值 r_t = π_new / π_old
            log_ratio = new_log_prob - b_old_lp
            ratio = torch.exp(log_ratio)

            # PPO Clip Loss（L_CLIP）
            # 【重要】A > 0 時：鼓勵增大 r；A < 0 時：鼓勵縮小 r
            # clip 限制更新幅度，避免策略突變
            surrogate1  = ratio * b_adv
            surrogate2  = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * b_adv
            pg_loss     = -torch.min(surrogate1, surrogate2).mean()

            # Value Loss（L_VALUE = MSE）
            value_loss  = nn.functional.mse_loss(new_value, b_returns)

            # Entropy Bonus（鼓勵探索）
            entropy_loss = -entropy.mean()

            # 總損失
            total_loss = pg_loss + value_coef * value_loss + entropy_coef * entropy_loss

            optimizer.zero_grad()
            total_loss.backward()
            # 梯度裁剪（PPO 的重要穩定技巧）
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            stats["pg_loss"]    += pg_loss.item()
            stats["value_loss"] += value_loss.item()
            stats["entropy"]    += (-entropy_loss).item()
            stats["total_loss"] += total_loss.item()
            n_batches += 1

    # 平均
    for k in stats:
        stats[k] /= max(n_batches, 1)
    return stats


# =========================================================
# 主訓練流程
# =========================================================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  使用裝置: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # ------ 1. 初始化模型與優化器 ------
    model = ActorCriticCNN().to(device)

    if args.bc_model:
        model = load_bc_weights(model, args.bc_model, device)
    else:
        print("⚠️  未指定 BC 模型，從隨機初始化開始訓練")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    # ------ 2. 初始化環境與 Buffer ------
    env    = PuzzleRLEnv()
    buffer = RolloutBuffer()

    # ------ 3. 訓練統計 ------
    best_mean_reward = -float("inf")
    episode_rewards  = []   # 每完成一個 episode 的總 reward
    episode_lengths  = []   # 每 episode 步數
    episode_success  = []   # 是否成功

    recent_rewards = []     # 最近 100 個 episode 的 reward（用於 best model 判斷）

    global_step  = 0
    global_start = time.time()

    # 初始 reset
    curriculum_stage = "easy"
    env.set_curriculum(curriculum_stage)
    obs     = env.reset()
    ep_reward = 0.0

    print(f"\n{'='*70}")
    print(f"🚀 開始 PPO 訓練 | 總更新次數: {args.total_updates} | "
          f"Rollout: {args.rollout_steps} steps")
    print(f"   Curriculum 起始: {curriculum_stage} ({env.sizes})")
    if args.max_time_hours > 0:
        print(f"   時間上限: {args.max_time_hours} 小時")
    print(f"{'='*70}\n")

    for update in range(1, args.total_updates + 1):
        # ---- Curriculum 調整 ----
        progress = update / args.total_updates
        if progress < 0.30:
            stage = "easy"
        elif progress < 0.70:
            stage = "medium"
        else:
            stage = "hard"

        if stage != curriculum_stage:
            curriculum_stage = stage
            env.set_curriculum(stage)
            print(f"  📈 課程升級 → {stage} ({env.sizes})")

        # ---- A. 收集 Rollout ----
        buffer.clear()
        model.eval()

        for _ in range(args.rollout_steps):
            # 編碼並轉成 Tensor
            obs_tensor   = torch.tensor(obs, dtype=torch.float32,
                                        device=device).unsqueeze(0)  # (1, 3, 12, 12)
            legal_mask_np = env.get_legal_mask()
            legal_mask_t  = torch.tensor(legal_mask_np, dtype=torch.bool,
                                         device=device).unsqueeze(0)  # (1, 4)

            # 採樣動作
            action, log_prob, entropy, value = model.get_action(obs_tensor, legal_mask_t)
            action_int = action.item()

            # 執行一步
            next_obs, reward, done, info = env.step(action_int)
            ep_reward += reward

            # 存入 buffer
            buffer.add(
                state      = obs,
                action     = action_int,
                log_prob   = log_prob.item(),
                reward     = reward,
                value      = value.item(),
                done       = float(done),
                legal_mask = legal_mask_np,
            )

            global_step += 1
            obs = next_obs

            if done:
                episode_rewards.append(ep_reward)
                episode_lengths.append(info["steps"])
                episode_success.append(float(info["success"]))
                recent_rewards.append(ep_reward)
                if len(recent_rewards) > 100:
                    recent_rewards.pop(0)
                ep_reward = 0.0
                obs = env.reset()

        # ---- B. 計算最後一步的 V(s_T+1) ----
        with torch.no_grad():
            obs_tensor  = torch.tensor(obs, dtype=torch.float32,
                                       device=device).unsqueeze(0)
            _, last_val = model(obs_tensor)
            last_value  = last_val.item() if not done else 0.0

        # ---- C. 轉 Tensor 並計算 GAE ----
        buf = buffer.to_tensors(device)
        advantages, returns = compute_gae(
            rewards    = buf["rewards"],
            values     = buf["values"],
            dones      = buf["dones"],
            last_value = last_value,
            gamma      = args.gamma,
            gae_lambda = args.gae_lambda,
        )

        # ---- D. PPO 梯度更新 ----
        model.train()
        stats = ppo_update(
            model            = model,
            optimizer        = optimizer,
            buffer_data      = buf,
            advantages       = advantages,
            returns          = returns,
            ppo_epochs       = args.ppo_epochs,
            mini_batch_size  = args.mini_batch_size,
            clip_epsilon     = args.clip_epsilon,
            value_coef       = args.value_coef,
            entropy_coef     = args.entropy_coef,
            max_grad_norm    = args.max_grad_norm,
        )

        # ---- E. 日誌與存檔 ----
        if update % args.log_interval == 0:
            elapsed = time.time() - global_start
            n_eps   = len(episode_rewards)
            mean_r  = np.mean(recent_rewards) if recent_rewards else 0
            mean_l  = np.mean(episode_lengths[-100:]) if episode_lengths else 0
            succ_r  = np.mean(episode_success[-100:]) * 100 if episode_success else 0

            print(f"  Update {update:>5d}/{args.total_updates} | "
                  f"Step: {global_step:>8d} | "
                  f"MeanR: {mean_r:>7.2f} | "
                  f"MeanLen: {mean_l:>6.1f} | "
                  f"Succ%: {succ_r:>5.1f}% | "
                  f"Entropy: {stats['entropy']:.3f} | "
                  f"PGLoss: {stats['pg_loss']:.4f} | "
                  f"VLoss: {stats['value_loss']:.4f} | "
                  f"{elapsed/60:.1f}min")

            # 存檔最佳模型（以最近 100 episode 的平均 reward 衡量）
            if recent_rewards and mean_r > best_mean_reward:
                best_mean_reward = mean_r
                torch.save({
                    "update":            update,
                    "model_state_dict":  model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "mean_reward":       mean_r,
                    "success_rate":      succ_r,
                    "curriculum_stage":  curriculum_stage,
                }, args.rl_model)
                print(f"    ⭐ 新最佳模型已存檔！MeanR={mean_r:.2f}")

        # ---- F. 時間上限檢查 ----
        if args.max_time_hours > 0:
            elapsed_h = (time.time() - global_start) / 3600.0
            if elapsed_h >= args.max_time_hours:
                print(f"\n⏱️  已達時間上限 ({args.max_time_hours}h)，結束訓練！")
                break

    # ------ 7. 訓練結束統計 ------
    elapsed = time.time() - global_start
    print(f"\n{'='*70}")
    print(f"✅ PPO 訓練完成！")
    print(f"   總更新: {update} | 總步數: {global_step:,} | 耗時: {elapsed/60:.1f}min")
    n_eps = len(episode_rewards)
    if n_eps > 0:
        print(f"   完成 Episode 數: {n_eps}")
        print(f"   最終平均 Reward (近100): {np.mean(recent_rewards):.2f}")
        print(f"   最終成功率 (近100): {np.mean(episode_success[-100:])*100:.1f}%")
    print(f"   最佳模型存檔: {args.rl_model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="華容道 PPO 強化學習訓練")
    parser.add_argument("--bc_model",       type=str,   default="best_model.pth",    help="BC 預訓練模型路徑")
    parser.add_argument("--rl_model",       type=str,   default="best_rl_model.pth", help="RL 最佳模型存檔路徑")
    parser.add_argument("--total_updates",  type=int,   default=3000,                help="總 PPO 更新次數")
    parser.add_argument("--rollout_steps",  type=int,   default=2048,                help="每次收集步數")
    parser.add_argument("--ppo_epochs",     type=int,   default=4,                   help="每批資料 PPO 更新次數")
    parser.add_argument("--mini_batch_size",type=int,   default=256,                 help="Mini-batch 大小")
    parser.add_argument("--gamma",          type=float, default=0.99,                help="折扣因子")
    parser.add_argument("--gae_lambda",     type=float, default=0.95,                help="GAE lambda")
    parser.add_argument("--clip_epsilon",   type=float, default=0.2,                 help="PPO clip 範圍")
    parser.add_argument("--value_coef",     type=float, default=0.5,                 help="Critic Loss 係數")
    parser.add_argument("--entropy_coef",   type=float, default=0.02,               help="Entropy Bonus 係數")
    parser.add_argument("--max_grad_norm",  type=float, default=0.5,                 help="梯度裁剪上限")
    parser.add_argument("--lr",             type=float, default=3e-5,               help="學習率（比 BC 小 10 倍）")
    parser.add_argument("--weight_decay",   type=float, default=1e-4,                help="Weight Decay")
    parser.add_argument("--log_interval",   type=int,   default=10,                  help="每隔幾次 update 印日誌")
    parser.add_argument("--max_time_hours", type=float, default=0,                   help="最大訓練時間 (小時)，0=無限")
    args = parser.parse_args()
    main(args)
