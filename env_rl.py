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

    # RL 訓練使用的尺寸範圍（現在只針對 3x3 進行訓練）
    CURRICULUM = {
        "easy":   [3],
        "medium": [3],
        "hard":   [3],
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
        self.max_steps = self.n * self.n * 20  # 3x3=180步，夠用且不會拖太久
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
            return self._encode(self.state, self.n), -0.5, False, {
                "success": False, "steps": self.step_count, "n": self.n,
                "illegal": True
            }

        # 執行動作，更新 state
        self.env.step(action)
        self.state = self.env.get_state()
        self.step_count += 1

        # ---- 極簡獎勵設計 ----
        # 核心哲學：只用「步數懲罰 + 過關獎勵」
        # 不使用曼哈頓距離進度獎勵！因為它會導致 AI 不敢繞路，
        # 進而陷入局部最優，最終觸發 Entropy Collapse。
        reward = -0.2   # 每步輕微懲罰（鼓勵縮短步數）

        done    = False
        success = False

        if self.env.is_solved():
            # 過關獎勵：+50 足以蓋過步數懲罰（50步 * -0.2 = -10）
            # 留出空間讓 AI 學習用更少步數獲得更高總分
            reward += 50.0
            done    = True
            success = True
        elif self.step_count >= self.max_steps:
            reward -= 5.0
            done    = True

        info = {
            "success":   success,
            "steps":     self.step_count,
            "n":         self.n,
            "manhattan": self._calculate_manhattan(),
        }
        return self._encode(self.state, self.n), reward, done, info

    def _calculate_manhattan(self) -> int:
        """
        計算當前盤面的總曼哈頓距離（實例方法版）。
        """
        total = 0
        for idx, v in enumerate(self.state):
            if v == 0:
                continue
            curr_r, curr_c = idx // self.n, idx % self.n
            goal_r, goal_c = (v - 1) // self.n, (v - 1) % self.n
            total += abs(curr_r - goal_r) + abs(curr_c - goal_c)
        return total

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
