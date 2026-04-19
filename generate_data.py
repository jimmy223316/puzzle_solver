"""
generate_data.py — 資料生成器
================================

功能：
1. 使用 VirtualPuzzleEnv + 降階法解題器，自動產生大量 (state, action) 訓練資料
2. 實作 Curriculum Learning：從小尺寸開始，逐步增加到大尺寸
3. 特徵編碼：將原始盤面轉換為 (3, 12, 12) 的歸一化張量
4. 輸出為 dataset.pt 檔案

資料流維度變化：
  原始盤面 list[N²] 
  → 重塑為 (N, N) 
  → 計算相對座標 (N, N, 2) [dx, dy]
  → 歸一化至 [-1, 1]
  → Zero-Pad 到 (12, 12, 2) 
  → 加入 Mask 通道 → (12, 12, 3) 
  → 轉置為 (3, 12, 12)
"""

import torch
import random
import time
import sys
from env import VirtualPuzzleEnv, solve_puzzle_virtual

# 最大盤面尺寸，所有輸入都會 Pad 到這個大小
MAX_N = 12


# =========================================================
# 特徵編碼器：盤面 → (3, 12, 12) 張量
# =========================================================
def encode_state(state: list, n: int) -> torch.Tensor:
    """
    將 1D 盤面狀態編碼為 (3, 12, 12) 的特徵張量（極速向量化版）。
    """
    feature = torch.zeros(3, MAX_N, MAX_N, dtype=torch.float32)
    s = torch.tensor(state, dtype=torch.float32).view(n, n)
    is_zero = (s == 0)
    
    # 建立網格座標
    r, c = torch.meshgrid(torch.arange(n), torch.arange(n), indexing='ij')
    
    # 計算每個格子的目標位置
    target_r = (s - 1) // n
    target_c = (s - 1) % n
    
    # 計算位移差
    dy = r.float() - target_r
    dx = c.float() - target_c
    
    # 對空格(0)的位移差設為 0
    dy[is_zero] = 0.0
    dx[is_zero] = 0.0
    
    # 歸一化到 [-1, 1] 並填入特徵圖
    feature[0, :n, :n] = dx / (MAX_N - 1)
    feature[1, :n, :n] = dy / (MAX_N - 1)
    feature[2, :n, :n] = 1.0  # Mask
    
    return feature


# =========================================================
# Curriculum Learning 尺寸選擇器
# =========================================================
def get_puzzle_size(episode: int, total_episodes: int) -> int:
    """
    根據當前 episode 決定要生成的盤面尺寸。
    
    Curriculum 策略（4 個階段）：
      - Stage 1 (0~20%):   N ∈ {3, 4, 5}       — 入門
      - Stage 2 (20~50%):  N ∈ {4, 5, 6, 7}     — 進階
      - Stage 3 (50~80%):  N ∈ {6, 7, 8, 9}     — 挑戰
      - Stage 4 (80~100%): N ∈ {8, 9, 10, 11, 12} — 大師
    
    Args:
        episode:        當前第幾局
        total_episodes: 總局數
    
    Returns:
        int: 盤面尺寸 N
    """
    progress = episode / total_episodes

    if progress < 0.2:
        return random.choice([3, 4, 5])
    elif progress < 0.5:
        return random.choice([4, 5, 6, 7])
    elif progress < 0.8:
        return random.choice([6, 7, 8, 9])
    else:
        return random.choice([8, 9, 10, 11, 12])


# =========================================================
# 主生成迴圈
# =========================================================
def generate_dataset(total_episodes: int = 50000,
                     output_path: str = "dataset.pt",
                     save_interval: int = 5000) -> None:
    """
    生成完整的訓練資料集。
    
    流程：
    1. 根據 Curriculum 選擇尺寸 N
    2. 建立 VirtualPuzzleEnv(N)，隨機打亂
    3. 用降階法解題，記錄每一步的 (state, action)
    4. 將所有步的 state 編碼為 (3,12,12) 張量
    5. 定期與最終存檔為 .pt 檔案
    
    Args:
        total_episodes: 總局數
        output_path:    輸出檔案路徑
        save_interval:  每隔多少局自動存檔一次
    """
    all_states = []    # list of (3,12,12) tensors
    all_actions = []   # list of int

    total_steps = 0
    success_count = 0
    fail_count = 0

    # 分尺寸統計
    size_stats = {n: {"games": 0, "steps": 0, "fails": 0} for n in range(3, 13)}

    start_time = time.time()
    print(f"🚀 開始生成資料：共 {total_episodes} 局")
    print(f"{'='*60}")

    for ep in range(total_episodes):
        n = get_puzzle_size(ep, total_episodes)

        env = VirtualPuzzleEnv(n)
        env.reset()
        state = env.get_state()
        trajectory = []

        # 解題並記錄軌跡
        success = solve_puzzle_virtual(state, n, trajectory)

        if success and len(trajectory) > 0:
            success_count += 1
            size_stats[n]["games"] += 1
            size_stats[n]["steps"] += len(trajectory)

            # 編碼每一步的狀態
            for (step_state, step_action) in trajectory:
                encoded = encode_state(step_state, n)
                all_states.append(encoded)
                all_actions.append(step_action)

            total_steps += len(trajectory)
        else:
            fail_count += 1
            size_stats[n]["fails"] += 1

        # 進度報告
        if (ep + 1) % 500 == 0:
            elapsed = time.time() - start_time
            eps_per_sec = (ep + 1) / elapsed
            eta = (total_episodes - ep - 1) / eps_per_sec
            print(f"  📊 Episode {ep+1:>6d}/{total_episodes} | "
                  f"累計步數: {total_steps:>8d} | "
                  f"成功率: {success_count/(success_count+fail_count)*100:.1f}% | "
                  f"速度: {eps_per_sec:.1f} ep/s | "
                  f"ETA: {eta/60:.1f} min")

        # 定期存檔（防止意外中斷遺失資料）
        if (ep + 1) % save_interval == 0:
            _save_dataset(all_states, all_actions, output_path, ep + 1, total_episodes)

    # 最終存檔
    _save_dataset(all_states, all_actions, output_path, total_episodes, total_episodes)

    # 印出最終統計
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✅ 資料生成完成！")
    print(f"   總遊戲數: {success_count + fail_count}")
    print(f"   成功: {success_count}  |  失敗: {fail_count}")
    print(f"   總步數 (samples): {total_steps:,}")
    print(f"   總耗時: {elapsed/60:.1f} 分鐘")
    print(f"\n📊 各尺寸統計：")
    print(f"   {'N':>3s} | {'遊戲數':>7s} | {'總步數':>9s} | {'平均步數':>8s} | {'失敗':>4s}")
    print(f"   {'---':>3s} | {'-------':>7s} | {'---------':>9s} | {'--------':>8s} | {'----':>4s}")
    for n_size in range(3, 13):
        s = size_stats[n_size]
        avg = s["steps"] / s["games"] if s["games"] > 0 else 0
        print(f"   {n_size:>3d} | {s['games']:>7d} | {s['steps']:>9d} | {avg:>8.1f} | {s['fails']:>4d}")


def _save_dataset(states: list, actions: list, path: str,
                  current_ep: int, total_ep: int) -> None:
    """
    將累積的資料存為 .pt 檔案。
    
    存檔格式：
    {
        "states":  Tensor[M, 3, 12, 12]  (float32)
        "actions": Tensor[M]             (int64)
    }
    其中 M = 總步數
    """
    if len(states) == 0:
        print(f"  ⚠️  跳過存檔：尚無資料")
        return

    print(f"  💾 存檔中... ({current_ep}/{total_ep}, {len(states):,} samples)")

    states_tensor = torch.stack(states)         # (M, 3, 12, 12)
    actions_tensor = torch.tensor(actions, dtype=torch.long)  # (M,)

    dataset = {
        "states": states_tensor,
        "actions": actions_tensor,
    }

    torch.save(dataset, path)
    size_mb = states_tensor.element_size() * states_tensor.nelement() / (1024 * 1024)
    print(f"  ✅ 已存檔至 {path} | States: {list(states_tensor.shape)} | "
          f"Actions: {list(actions_tensor.shape)} | ~{size_mb:.1f} MB")


# =========================================================
# 入口點
# =========================================================
if __name__ == "__main__":
    # 可從命令列傳入總局數，預設 50000
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
    generate_dataset(total_episodes=total)
