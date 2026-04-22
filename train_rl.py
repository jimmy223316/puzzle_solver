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
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model_rl import ActorCriticCNN, load_bc_weights
from env_rl import PuzzleRLEnv
from subproc_vec_env import SubprocVecEnv

# RTX 5060 Ti Tensor Core 加速：這行讓矩陣計算自動使用 TF32，在 Ampere+ 架構上可及 20~30% 加速
torch.set_float32_matmul_precision('high')

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
                                        dtype=torch.float32, device=device),
            "actions":     torch.tensor(np.array(self.actions),
                                        dtype=torch.long,    device=device),
            "log_probs":   torch.tensor(np.array(self.log_probs),
                                        dtype=torch.float32, device=device),
            "rewards":     torch.tensor(np.array(self.rewards),
                                        dtype=torch.float32, device=device),
            "values":      torch.tensor(np.array(self.values),
                                        dtype=torch.float32, device=device),
            "dones":       torch.tensor(np.array(self.dones),
                                        dtype=torch.float32, device=device),
            "legal_masks": torch.tensor(np.array(self.legal_masks),
                                        dtype=torch.bool,    device=device),
        }


# =========================================================
# GAE 計算
# =========================================================
def compute_gae(rewards: torch.Tensor,
                values:  torch.Tensor,
                dones:   torch.Tensor,
                last_value: torch.Tensor,
                gamma: float = 0.99,
                gae_lambda: float = 0.95) -> tuple:
    """
    計算 GAE (Generalized Advantage Estimation)。
    """
    T = rewards.shape[0]
    num_envs = rewards.shape[1]
    advantages = torch.zeros_like(rewards, device=rewards.device)

    last_adv = torch.zeros(num_envs, device=rewards.device)
    for t in reversed(range(T)):
        next_value = values[t + 1] if t + 1 < T else last_value
        next_done  = dones[t]

        delta = rewards[t] + gamma * next_value * (1 - next_done) - values[t]

        last_adv = delta + gamma * gae_lambda * (1 - next_done) * last_adv
        advantages[t] = last_adv

    returns = advantages + values  # V 目標值 = A + V_pred
    # 回傳壓平的 tensor 給 PPO
    return advantages.reshape(-1), returns.reshape(-1)


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

            b_states      = buffer_data["states"][b_idx]
            b_actions     = buffer_data["actions"][b_idx]
            b_old_lp      = buffer_data["log_probs"][b_idx]
            b_legal_masks = buffer_data["legal_masks"][b_idx]
            b_adv         = adv_norm[b_idx]
            b_returns     = returns[b_idx]

            new_log_prob, entropy, new_value = model.evaluate_actions(
                b_states, b_actions, b_legal_masks
            )

            log_ratio = new_log_prob - b_old_lp
            ratio = torch.exp(log_ratio)

            surrogate1  = ratio * b_adv
            surrogate2  = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * b_adv
            pg_loss     = -torch.min(surrogate1, surrogate2).mean()

            value_loss  = nn.functional.smooth_l1_loss(new_value, b_returns)

            # 🛡️【Entropy Floor 保護】：當 entropy 掉得太低時，強制加大探索係數
            # 這防止了 Entropy Collapse - 一旦模型開始確定化，立刻給它一記警醒
            current_entropy = entropy.mean().item()
            effective_entropy_coef = entropy_coef
            if current_entropy < 0.05:
                effective_entropy_coef = entropy_coef * 3.0  # 三倍懲罰讓策略重新分散

            entropy_loss = -entropy.mean()
            total_loss = pg_loss + value_coef * value_loss + effective_entropy_coef * entropy_loss

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            stats["pg_loss"]    += pg_loss.item()
            stats["value_loss"] += value_loss.item()
            stats["entropy"]    += current_entropy
            stats["total_loss"] += total_loss.item()
            n_batches += 1

    # 平均
    for k in stats:
        stats[k] /= max(n_batches, 1)
    return stats


# =========================================================
# 主訓練流程
# =========================================================
def make_env():
    return PuzzleRLEnv()

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

    # ------ 1.5 分離學習率 (Separate Learning Rates) ------
    # - Backbone 和 Actor Head 已經訓練得很好，使用極小的學習率保護權重 (例如 5e-6)
    # - Critic Head 是全新初始化的，需要較大的學習率趕快收斂 (例如 3e-4)
    actor_params = []
    critic_params = []
    for name, param in model.named_parameters():
        if "critic_head" in name:
            critic_params.append(param)
        else:
            actor_params.append(param)

    optimizer = optim.AdamW([
        {"params": actor_params, "lr": args.actor_lr},
        {"params": critic_params, "lr": args.critic_lr}
    ], weight_decay=args.weight_decay)

    # ------ 2. 初始化環境與 Buffer ------
    envs = [make_env for _ in range(args.num_envs)]
    env = SubprocVecEnv(envs)
    buffer = RolloutBuffer()

    # ------ 3. 訓練統計 ------
    best_mean_reward = -float("inf")
    episode_rewards  = []   # 每完成一個 episode 的總 reward
    episode_lengths  = []   # 每 episode 步數
    episode_success  = []   # 是否成功

    recent_rewards = []     # 最近 100 個 episode 的 reward（用於 best model 判斷）

    global_step  = 0
    global_start = time.time()

    # 追蹤統計數據（用於畫圖）
    history = {
        "updates": [],
        "rewards": [],
        "lengths": [],
        "v_loss":  [],
        "pg_loss": [],
        "success": []
    }

    # 初始 reset
    curriculum_stage = "easy"
    if args.focus_size > 0:
        env.reset(n=args.focus_size)  # 強制指定尺寸
        print(f"  🎯 模式：指定尺寸訓練 (N={args.focus_size})")
    else:
        env.set_curriculum(curriculum_stage)
        print(f"  📈 模式：Curriculum Learning 起始: {curriculum_stage} ({PuzzleRLEnv.CURRICULUM[curriculum_stage]})")
    
    obs     = env.reset(n=args.focus_size if args.focus_size > 0 else None)
    ep_rewards = [0.0] * args.num_envs

    print(f"\n{'='*70}")
    print(f"🚀 開始 PPO 訓練 | 總更新次數: {args.total_updates} | "
          f"Rollout: {args.rollout_steps} steps")
    print(f"   Curriculum 起始: {curriculum_stage} ({PuzzleRLEnv.CURRICULUM[curriculum_stage]})")
    if args.max_time_hours > 0:
        print(f"   時間上限: {args.max_time_hours} 小時")
    print(f"{'='*70}\n")

    for update in range(1, args.total_updates + 1):
        # ---- 0. 評論家預熱 (Critic Warm-up) ----
        is_warmup = update <= args.warmup_updates
        for name, param in model.named_parameters():
            if "critic_head" not in name:
                param.requires_grad = not is_warmup

        if is_warmup and update == 1:
            print(f"  🔥 開始 Critic Warm-up：凍結 Actor 權重 (前 {args.warmup_updates} 次更新)，僅訓練 Critic！")
        elif not is_warmup and update == args.warmup_updates + 1:
            print(f"  🔓 解除凍結：預熱結束，Actor 權重加入訓練！")

        # ---- Curriculum 調整 (若非 focus_size 模式) ----
        if args.focus_size == 0:
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
                print(f"  📈 課程升級 → {stage} ({PuzzleRLEnv.CURRICULUM[stage]})")

        # ---- A. 收集 Rollout ----
        buffer.clear()
        
        rollout_steps_per_env = args.rollout_steps // args.num_envs

        with torch.inference_mode():
            model.eval()
            for _ in range(rollout_steps_per_env):
                # 編碼並轉成 Tensor
                obs_tensor   = torch.tensor(obs, dtype=torch.float32, device=device)  # (num_envs, 3, 12, 12)
                legal_mask_np = env.get_legal_masks()
                legal_mask_t  = torch.tensor(legal_mask_np, dtype=torch.bool, device=device)  # (num_envs, 4)

                # 採樣動作 (Batch inference)
                actions, log_probs, entropies, values = model.get_action(obs_tensor, legal_mask_t)
                actions_np = actions.cpu().numpy()

                # 執行一步 (Asynchronous multiple envs)
                next_obs, rewards, dones, infos = env.step(actions_np)
                
                for i in range(args.num_envs):
                    ep_rewards[i] += rewards[i]
                    if dones[i]:
                        episode_rewards.append(ep_rewards[i])
                        episode_lengths.append(infos[i]["steps"])
                        episode_success.append(float(infos[i]["success"]))
                        recent_rewards.append(ep_rewards[i])
                        if len(recent_rewards) > 100:
                            recent_rewards.pop(0)
                        ep_rewards[i] = 0.0

                # 存入 buffer
                buffer.add(
                    state      = obs,
                    action     = actions_np,
                    log_prob   = log_probs.cpu().numpy(),
                    reward     = rewards,
                    value      = values.cpu().numpy(),
                    done       = dones.astype(np.float32),
                    legal_mask = legal_mask_np,
                )

                global_step += args.num_envs
                obs = next_obs

            # ---- B. 計算最後一步的 V(s_T+1) ----
            obs_tensor  = torch.tensor(obs, dtype=torch.float32, device=device)
            _, last_values_t = model(obs_tensor)
            last_value = last_values_t.squeeze(-1)

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

        # 平展 buffer 中的各個欄位以供 PPO 更新使用
        buf["states"]      = buf["states"].reshape(-1, 3, 12, 12)
        buf["actions"]     = buf["actions"].reshape(-1)
        buf["log_probs"]   = buf["log_probs"].reshape(-1)
        buf["legal_masks"] = buf["legal_masks"].reshape(-1, 4)

        # ---- D. PPO 梯度更新 ----
        model.train()
        
        # 🚧【神級修復】：強制凍結所有 BatchNorm 層的統計數據！
        # 即使呼叫了 model.train()，我們也必須讓 BN 層保持 eval 模式，
        # 否則強化學習的新資料分佈會瞬間摧毀 BC 預訓練好的特徵提取能力。
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                
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

            # 記錄歷史數據
            history["updates"].append(update)
            history["rewards"].append(mean_r)
            history["lengths"].append(mean_l)
            history["v_loss"].append(stats["value_loss"])
            history["pg_loss"].append(stats["pg_loss"])
            history["success"].append(succ_r)

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

    # ------ 8. 繪製結果圖表 ------
    if len(history["updates"]) > 0:
        print("\n📊 正在生成訓練結果圖表...")
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Reward & Success
        plt.subplot(1, 3, 1)
        plt.title("Reward & Success Rate")
        plt.plot(history["updates"], history["rewards"], label="Mean Reward", color="blue")
        plt.xlabel("Updates")
        plt.ylabel("Reward")
        ax2 = plt.gca().twinx()
        ax2.plot(history["updates"], history["success"], label="Success %", color="green", linestyle="--")
        ax2.set_ylabel("Success Rate %")
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Loss
        plt.subplot(1, 3, 2)
        plt.title("Loss Scaling")
        plt.plot(history["updates"], history["v_loss"], label="Value Loss", color="red")
        plt.plot(history["updates"], history["pg_loss"], label="PG Loss", color="orange")
        plt.yscale("log")
        plt.xlabel("Updates")
        plt.ylabel("Loss (Log Scale)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Mean Length
        plt.subplot(1, 3, 3)
        plt.title("Mean Episode Length")
        plt.plot(history["updates"], history["lengths"], color="purple")
        plt.xlabel("Updates")
        plt.ylabel("Steps")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = "training_results.png"
        plt.savefig(plot_path)
        print(f"✅ 訓練圖表已存檔至: {plot_path}")
        # 如果在有 GUI 的環境，可以考慮 plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="華容道 PPO 強化學習訓練")
    parser.add_argument("--bc_model",       type=str,   default="best_model.pth",    help="BC 預訓練模型路徑")
    parser.add_argument("--rl_model",       type=str,   default="best_rl_model.pth", help="RL 最佳模型存檔路徑")
    parser.add_argument("--num_envs",       type=int,   default=16,                  help="平行環境數量")
    parser.add_argument("--total_updates",  type=int,   default=100000,              help="總 PPO 更新次數")
    parser.add_argument("--rollout_steps",  type=int,   default=4096,                help="每次收集步數 (加大以提高 GPU 利用率)")
    parser.add_argument("--ppo_epochs",     type=int,   default=4,                   help="每批資料 PPO 更新次數")
    parser.add_argument("--mini_batch_size",type=int,   default=1024,                help="Mini-batch 大小 (加大以提高 GPU 利用率，16GB VRAM 夠用)")
    parser.add_argument("--gamma",          type=float, default=0.99,                help="折扣因子 (0.99 適合稀疏獎勵環境)")
    parser.add_argument("--gae_lambda",     type=float, default=0.95,                help="GAE lambda")
    parser.add_argument("--clip_epsilon",   type=float, default=0.1,                 help="PPO clip 範圍")
    parser.add_argument("--value_coef",     type=float, default=0.5,                 help="Critic Loss 係數")
    parser.add_argument("--entropy_coef",   type=float, default=0.05,                help="Entropy Bonus 係數 (調高以防 Entropy Collapse)")
    parser.add_argument("--max_grad_norm",  type=float, default=0.5,                 help="梯度裁剪上限")
    parser.add_argument("--actor_lr",       type=float, default=5e-6,                help="Actor (預訓練) 學習率（極小）")
    parser.add_argument("--critic_lr",      type=float, default=1e-4,                help="Critic (全新) 學習率（適中）")
    parser.add_argument("--warmup_updates", type=int,   default=30,                  help="Critic 預熱階段的更新次數 (加長以保護 Actor)")
    parser.add_argument("--focus_size",     type=int,   default=3,                   help="強制指定單一尺寸進行訓練 (0=使用 Curriculum)")
    parser.add_argument("--weight_decay",   type=float, default=1e-4,                help="Weight Decay")
    parser.add_argument("--log_interval",   type=int,   default=10,                  help="每隔幾次 update 印日誌")
    parser.add_argument("--max_time_hours", type=float, default=20,                  help="最大訓練時間 (小時)，0=無限")
    args = parser.parse_args()
    main(args)
