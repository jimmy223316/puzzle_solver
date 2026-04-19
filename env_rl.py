"""
env_rl.py — 強化學習環境封裝
================================

封裝 VirtualPuzzleEnv，提供 RL 標準介面：
  reset(n) → encoded_state (3, 12, 12) numpy array
  step(action) → (next_state, reward, done, info)

Reward Shaping：
  每步懲罰  : -0.1
  曼哈頓進度: (prev_manhattan - curr_manhattan) * 0.5
  過關獎勵  : +100.0
  超時懲罰  : -10.0（到達步數上限時）

動作空間：
  0=上 1=下 2=左 3=右（空格移動方向，與 env.py 一致）

注意：
  - encode_state 引用自 generate_data.py，保持編碼方式完全一致
  - 所有 state 回傳為 numpy array (3, 12, 12)，在 train_rl.py 中再轉成 Tensor
"""

import random
import numpy as np
import torch
from env import VirtualPuzzleEnv, ACTION_DELTAS
from generate_data import encode_state

MAX_N = 12


class PuzzleRLEnv:
    """
    RL 環境封裝，支援動態尺寸 (N=3~9)。

    屬性：
        n           : 當前盤面尺寸
        state       : 當前盤面 (1D list，長度 N²)
        step_count  : 當前 episode 的步數
        max_steps   : 步數上限
        prev_manhattan : 上一步的曼哈頓距離（用於計算進度獎勵）
    """

    # RL 訓練使用的尺寸範圍（不碰 10~12，BC 已夠好）
    CURRICULUM = {
        "easy":   [3, 4, 5],
        "medium": [4, 5, 6, 7],
        "hard":   [5, 6, 7, 8, 9],
    }

    def __init__(self, sizes: list = None):
        """
        Args:
            sizes: 允許的盤面尺寸列表。若 None，從 easy 課程開始。
        """
        self.sizes = sizes or self.CURRICULUM["easy"]
        self.n = self.sizes[0]
        self.env = VirtualPuzzleEnv(self.n)
        self.state = None
        self.step_count = 0
        self.max_steps = 0
        self.prev_manhattan = 0

    def set_curriculum(self, stage: str):
        """切換課程難度：'easy' / 'medium' / 'hard'"""
        self.sizes = self.CURRICULUM[stage]

    def reset(self, n: int = None) -> np.ndarray:
        """
        重置環境，生成隨機合法盤面。

        Args:
            n: 指定盤面尺寸；若 None 則從 self.sizes 隨機選取

        Returns:
            encoded_state: numpy array (3, 12, 12) float32
        """
        self.n = n if n is not None else random.choice(self.sizes)
        self.env = VirtualPuzzleEnv(self.n)
        self.env.reset()
        self.state = self.env.get_state()
        self.step_count = 0
        self.max_steps = self.n * self.n * 40  # 給充足探索空間
        self.prev_manhattan = self._manhattan(self.state, self.n)
        return self._encode(self.state, self.n)

    def step(self, action: int):
        """
        執行動作。

        Args:
            action: 0=上 1=下 2=左 3=右

        Returns:
            next_state: numpy array (3, 12, 12) float32
            reward:     float
            done:       bool
            info:       dict（含 'success', 'steps', 'n'）
        """
        # 確認合法性（非法動作直接給懲罰，不執行）
        legal = self.get_legal_actions()
        if action not in legal:
            # 撞牆：small penalty，state 不變
            return self._encode(self.state, self.n), -1.0, False, {
                "success": False, "steps": self.step_count, "n": self.n,
                "illegal": True
            }

        # 執行動作，更新 state
        self.env.step(action)
        self.state = self.env.get_state()
        self.step_count += 1

        # 計算各項 reward
        curr_manhattan = self._manhattan(self.state, self.n)

        # 進度獎勵：曼哈頓距離縮短 → 正獎勵
        progress = (self.prev_manhattan - curr_manhattan) * 0.5
        self.prev_manhattan = curr_manhattan

        step_penalty = -0.1          # 每步固定扣分

        done    = False
        reward  = step_penalty + progress
        success = False

        if self.env.is_solved():
            # 過關！獎勵要夠大才能蓋過累積的 step penalty
            reward  += 100.0
            done    = True
            success = True
        elif self.step_count >= self.max_steps:
            # 超時
            reward -= 10.0
            done    = True

        info = {
            "success":   success,
            "steps":     self.step_count,
            "n":         self.n,
            "manhattan": curr_manhattan,
        }
        return self._encode(self.state, self.n), reward, done, info

    def get_legal_actions(self) -> list:
        """回傳當前合法動作清單（不會撞牆的方向）。"""
        return self.env.get_legal_actions()

    def get_legal_mask(self) -> np.ndarray:
        """
        回傳合法動作的 bool mask，供 Actor 遮蔽非法動作。

        Returns:
            mask: numpy bool array (4,)，True=合法
        """
        mask = np.zeros(4, dtype=bool)
        for a in self.env.get_legal_actions():
            mask[a] = True
        return mask

    @staticmethod
    def _manhattan(state: list, n: int) -> int:
        """
        計算當前盤面的總曼哈頓距離。

        對每個非零方塊，計算其當前位置到目標位置的 |dx| + |dy|。
        這是衡量「整體解題進度」的關鍵指標。

        公式：
            對值為 v 的方塊（v ≠ 0）：
              目標行 = (v-1) // n，目標列 = (v-1) % n
              當前行 = idx // n，  當前列 = idx % n
              距離 = |當前行 - 目標行| + |當前列 - 目標列|
        """
        total = 0
        for idx, v in enumerate(state):
            if v == 0:
                continue
            curr_r, curr_c = idx // n, idx % n
            goal_r, goal_c = (v - 1) // n, (v - 1) % n
            total += abs(curr_r - goal_r) + abs(curr_c - goal_c)
        return total

    @staticmethod
    def _encode(state: list, n: int) -> np.ndarray:
        """
        將 1D 盤面編碼為 (3, 12, 12) numpy array。

        使用與 generate_data.py 完全相同的 encode_state 函式，
        確保 RL 和 BC 的特徵空間完全一致。

        Returns:
            numpy array (3, 12, 12) float32
        """
        tensor = encode_state(state, n)  # torch.Tensor (3, 12, 12)
        return tensor.numpy()


if __name__ == "__main__":
    print("🧪 測試 PuzzleRLEnv")
    env = PuzzleRLEnv(sizes=[3, 4])

    for n in [3, 4]:
        obs = env.reset(n=n)
        print(f"\n{'='*40}")
        print(f"N={n}: obs shape={obs.shape}, dtype={obs.dtype}")
        print(f"  初始曼哈頓距離: {env.prev_manhattan}")
        print(f"  合法動作: {env.get_legal_actions()}")
        print(f"  合法 mask: {env.get_legal_mask()}")

        total_reward = 0
        for step in range(100):
            action = random.choice(env.get_legal_actions())
            next_obs, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                print(f"  Episode 結束！步數={info['steps']}, "
                      f"成功={info['success']}, 總獎勵={total_reward:.2f}")
                break
        else:
            print(f"  100步後未結束，目前曼哈頓={info['manhattan']}")

    print("\n✅ 測試完成！")
