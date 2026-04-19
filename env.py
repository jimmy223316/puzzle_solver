"""
env.py — 純 Python 虛擬華容道環境 & 降階法解題器（極速版）
==============================================================

效能優化要點：
- 使用 pos 字典 {tile_val: index} 取代 state.index() 的 O(N²) 搜索
- 每次 swap 後即時更新 pos 字典 → O(1) 查找
- forbidden 參數統一使用 set，避免重複轉換

動作定義（空格 0 的移動方向）：
  0 = 上 (Up)    — 空格與上方數字交換
  1 = 下 (Down)  — 空格與下方數字交換
  2 = 左 (Left)  — 空格與左方數字交換
  3 = 右 (Right) — 空格與右方數字交換
"""

import random
from collections import deque

# =========================================================
# 動作常數定義
# =========================================================
ACTION_UP    = 0
ACTION_DOWN  = 1
ACTION_LEFT  = 2
ACTION_RIGHT = 3

# 空格移動方向 → (dr, dc) 映射
ACTION_DELTAS = {
    ACTION_UP:    (-1,  0),
    ACTION_DOWN:  ( 1,  0),
    ACTION_LEFT:  ( 0, -1),
    ACTION_RIGHT: ( 0,  1),
}


# =========================================================
# VirtualPuzzleEnv — 虛擬華容道環境
# =========================================================
class VirtualPuzzleEnv:
    """
    純 Python 的 N×N 數字華容道環境。
    
    state:  長度 N² 的 1D list，索引 = row * N + col
    目標狀態：[1, 2, 3, ..., N²-1, 0]
    """

    def __init__(self, n: int):
        assert 3 <= n <= 12, f"尺寸 N 必須在 3~12 之間，收到 {n}"
        self.n = n
        self.state = list(range(1, n * n)) + [0]
        self.trajectory = []

    def reset(self, min_shuffles: int = None) -> list:
        """以反向隨機走步法產生合法的隨機初始盤面（保證 100% 可解）。"""
        n = self.n
        self.state = list(range(1, n * n)) + [0]
        self.trajectory = []

        if min_shuffles is None:
            min_shuffles = n * n * 20

        zero_idx = len(self.state) - 1
        last_action = -1
        opposite = {ACTION_UP: ACTION_DOWN, ACTION_DOWN: ACTION_UP,
                    ACTION_LEFT: ACTION_RIGHT, ACTION_RIGHT: ACTION_LEFT}

        for _ in range(min_shuffles):
            r, c = zero_idx // n, zero_idx % n
            candidates = []
            for action, (dr, dc) in ACTION_DELTAS.items():
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n:
                    if action != opposite.get(last_action, -1):
                        candidates.append((action, nr * n + nc))
            if not candidates:
                for action, (dr, dc) in ACTION_DELTAS.items():
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n and 0 <= nc < n:
                        candidates.append((action, nr * n + nc))

            action, new_idx = random.choice(candidates)
            self.state[zero_idx], self.state[new_idx] = self.state[new_idx], self.state[zero_idx]
            zero_idx = new_idx
            last_action = action

        return list(self.state)

    def get_legal_actions(self) -> list:
        n = self.n
        zero_idx = self.state.index(0)
        r, c = zero_idx // n, zero_idx % n
        actions = []
        for action, (dr, dc) in ACTION_DELTAS.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n:
                actions.append(action)
        return actions

    def step(self, action: int) -> tuple:
        n = self.n
        zero_idx = self.state.index(0)
        r, c = zero_idx // n, zero_idx % n
        dr, dc = ACTION_DELTAS[action]
        nr, nc = r + dr, c + dc
        assert 0 <= nr < n and 0 <= nc < n, f"非法動作 {action}"
        new_idx = nr * n + nc
        self.state[zero_idx], self.state[new_idx] = self.state[new_idx], self.state[zero_idx]
        return list(self.state), self.is_solved()

    def is_solved(self) -> bool:
        n = self.n
        goal = list(range(1, n * n)) + [0]
        return self.state == goal

    def get_state(self) -> list:
        return list(self.state)


# =========================================================
# 極速降階法解題器（位置字典版）
# =========================================================

def _build_pos(state: list) -> dict:
    """建立 {tile_value: index} 的位置快取字典。"""
    return {val: idx for idx, val in enumerate(state)}


def _click_to_action(pos: dict, num_to_click: int, n: int) -> int:
    """
    將「點擊哪個數字」轉換為「空格移動方向」。
    利用 pos 字典做 O(1) 查找。
    """
    z_idx = pos[0]
    t_idx = pos[num_to_click]
    zr, zc = z_idx // n, z_idx % n
    tr, tc = t_idx // n, t_idx % n
    dr, dc = tr - zr, tc - zc

    if   dr == -1 and dc ==  0: return ACTION_UP
    elif dr ==  1 and dc ==  0: return ACTION_DOWN
    elif dr ==  0 and dc == -1: return ACTION_LEFT
    elif dr ==  0 and dc ==  1: return ACTION_RIGHT
    else:
        raise ValueError(f"數字 {num_to_click} 不與空格相鄰！z=({zr},{zc}), t=({tr},{tc})")


def _swap(state: list, pos: dict, idx_a: int, idx_b: int):
    """交換 state 中兩個位置，並同步更新 pos 字典。O(1) 操作。"""
    val_a, val_b = state[idx_a], state[idx_b]
    state[idx_a], state[idx_b] = val_b, val_a
    pos[val_a], pos[val_b] = idx_b, idx_a


def apply_step_fast(state: list, pos: dict, num_to_click: int, n: int,
                    trajectory: list = None):
    """
    極速版 apply_step：利用 pos 字典做 O(1) 查找和更新。
    
    Args:
        state:        盤面 (1D list，就地修改)
        pos:          位置字典 {val: idx}（就地修改）
        num_to_click: 被「點擊」的數字
        n:            盤面尺寸
        trajectory:   記錄 (state_copy, action) 的列表
    """
    if trajectory is not None:
        action = _click_to_action(pos, num_to_click, n)
        trajectory.append((list(state), action))

    z_idx = pos[0]
    t_idx = pos[num_to_click]
    _swap(state, pos, z_idx, t_idx)


# ---------------------------------------------------------
# BFS 路徑尋找
# ---------------------------------------------------------
def get_path_to(start_idx: int, target_idx: int, n: int, forbidden: set) -> list:
    """BFS 尋找從 start_idx 到 target_idx 的路徑，避開 forbidden 集合。"""
    if start_idx == target_idx:
        return []
    # 如果目標本身在禁區中，直接返回不可達
    if target_idx in forbidden:
        return None
    q = deque([(start_idx, [])])
    visited = {start_idx}
    visited.update(forbidden)
    while q:
        curr, path = q.popleft()
        r, c = curr // n, curr % n
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n:
                nxt = nr * n + nc
                if nxt == target_idx:
                    return path + [nxt]
                if nxt not in visited:
                    visited.add(nxt)
                    q.append((nxt, path + [nxt]))
    return None


def move_zero_to(state: list, pos: dict, target_idx: int, locked: set, n: int,
                 trajectory: list = None) -> bool:
    """將空格移動到 target_idx。"""
    path = get_path_to(pos[0], target_idx, n, locked)
    if path is None:
        return False
    for p in path:
        apply_step_fast(state, pos, state[p], n, trajectory)
    return True


def move_tile(state: list, pos: dict, tile_val: int, target_idx: int,
              locked: set, n: int, trajectory: list = None) -> bool:
    """將 tile_val 方塊推動到 target_idx。使用穩健推動法。加入迭代上限防止無限迴圈。"""
    max_iters = n * n * 200  # 安全上限（需足夠高以應對邊界死角）
    for _ in range(max_iters):
        curr_idx = pos[tile_val]
        if curr_idx == target_idx:
            return True

        path = get_path_to(curr_idx, target_idx, n, locked)
        if not path:
            return False

        next_pos = path[0]
        # 嘗試讓 0 繞過目標方塊
        forbidden_for_zero = locked | {curr_idx}
        zero_path = get_path_to(pos[0], next_pos, n, forbidden_for_zero)

        if zero_path is not None:
            for p in zero_path:
                apply_step_fast(state, pos, state[p], n, trajectory)
            apply_step_fast(state, pos, tile_val, n, trajectory)
        else:
            # 死結：0 繞不過去，踩著目標方塊擠過去
            loose_path = get_path_to(pos[0], next_pos, n, locked)
            if not loose_path:
                return False
            apply_step_fast(state, pos, state[loose_path[0]], n, trajectory)

    return pos[tile_val] == target_idx


# ---------------------------------------------------------
# 主演算法：降階法
# ---------------------------------------------------------
def solve_puzzle_virtual(state: list, n: int, trajectory: list = None) -> bool:
    """
    使用降階法解 N×N 華容道（極速版）。
    
    Args:
        state:      盤面 (1D list，就地修改)
        n:          盤面尺寸
        trajectory: 記錄 (state, action) 對的列表
    
    Returns:
        True = 成功, False = 失敗
    """
    pos = _build_pos(state)  # 建立位置快取
    locked = set()

    # 1. 一層一層剝開（直到剩下 2x2）
    for step in range(n - 2):

        # --- A. 放上面的 Row ---
        row_vals = [step * n + c + 1 for c in range(step, n)]
        normal_vals = row_vals[:-2]
        val1, val2 = row_vals[-2], row_vals[-1]
        idx1, idx2 = val1 - 1, val2 - 1

        for val in normal_vals:
            if not move_tile(state, pos, val, val - 1, locked, n, trajectory):
                return False
            locked.add(val - 1)

        if not (pos[val1] == idx1 and pos[val2] == idx2):
            if pos[val2] in (idx1, idx2):
                if not move_tile(state, pos, val2, idx1 + 2 * n, locked, n, trajectory):
                    return False

            if not move_tile(state, pos, val1, idx2, locked, n, trajectory):
                return False
            locked.add(idx2)

            locked.add(idx1)
            if not move_tile(state, pos, val2, idx2 + n, locked, n, trajectory):
                return False
            locked.add(idx2 + n)

            locked.remove(idx1)
            if not move_zero_to(state, pos, idx1, locked, n, trajectory):
                return False
            locked.remove(idx2)
            locked.remove(idx2 + n)
            apply_step_fast(state, pos, val1, n, trajectory)
            apply_step_fast(state, pos, val2, n, trajectory)

        locked.add(idx1)
        locked.add(idx2)

        # --- B. 放左邊的 Col ---
        col_vals = [r * n + step + 1 for r in range(step + 1, n)]
        normal_vals = col_vals[:-2]
        val1, val2 = col_vals[-2], col_vals[-1]
        idx1, idx2 = val1 - 1, val2 - 1

        for val in normal_vals:
            if not move_tile(state, pos, val, val - 1, locked, n, trajectory):
                return False
            locked.add(val - 1)

        if not (pos[val1] == idx1 and pos[val2] == idx2):
            if pos[val2] in (idx1, idx2):
                if not move_tile(state, pos, val2, idx1 + 2, locked, n, trajectory):
                    return False

            if not move_tile(state, pos, val1, idx2, locked, n, trajectory):
                return False
            locked.add(idx2)

            locked.add(idx1)
            if not move_tile(state, pos, val2, idx2 + 1, locked, n, trajectory):
                return False
            locked.add(idx2 + 1)

            locked.remove(idx1)
            if not move_zero_to(state, pos, idx1, locked, n, trajectory):
                return False
            locked.remove(idx2)
            locked.remove(idx2 + 1)
            apply_step_fast(state, pos, val1, n, trajectory)
            apply_step_fast(state, pos, val2, n, trajectory)

        locked.add(idx1)
        locked.add(idx2)

    # 2. 處理最後的 2x2 核心
    i1, i2 = (n - 2) * n + n - 2, (n - 2) * n + n - 1
    i3, i4 = (n - 1) * n + n - 2, (n - 1) * n + n - 1
    t1, t2, t3 = i1 + 1, i2 + 1, i3 + 1

    for _ in range(20):
        if state[i1] == t1 and state[i2] == t2 and state[i3] == t3:
            if state[i4] != 0:
                move_zero_to(state, pos, i4, set(), n, trajectory)
            return True

        z_idx = pos[0]
        if z_idx == i1:
            apply_step_fast(state, pos, state[i2], n, trajectory)
        elif z_idx == i2:
            apply_step_fast(state, pos, state[i4], n, trajectory)
        elif z_idx == i4:
            apply_step_fast(state, pos, state[i3], n, trajectory)
        elif z_idx == i3:
            apply_step_fast(state, pos, state[i1], n, trajectory)
        else:
            move_zero_to(state, pos, i4, locked, n, trajectory)

    return False


# =========================================================
# 快速測試
# =========================================================
if __name__ == "__main__":
    import time

    print("=" * 50)
    print("VirtualPuzzleEnv 快速測試（極速版）")
    print("=" * 50)

    trials_per_size = 5
    for n in range(3, 13):
        successes = 0
        total_steps = 0
        total_time = 0.0

        for trial in range(trials_per_size):
            env = VirtualPuzzleEnv(n)
            env.reset()
            state = env.get_state()
            trajectory = []

            t0 = time.time()
            success = solve_puzzle_virtual(state, n, trajectory)
            elapsed = time.time() - t0

            if success:
                successes += 1
                total_steps += len(trajectory)
            total_time += elapsed

        avg_steps = total_steps // max(successes, 1)
        rate = successes / trials_per_size * 100
        print(f"  {n:>2d}x{n:<2d}: {successes}/{trials_per_size} 成功 ({rate:.0f}%)  "
              f"平均步數={avg_steps:>6d}  總耗時={total_time:.3f}s")

    print("\n全部測試完成！")
