"""
efficient_solver.py  v5 (FINAL)
================================
支援新版「行/列整排滑動」機制。

核心思路：
  - 沿用 manual_solver.py 已驗證的降階法與 move_tile 邏輯（正確）
  - 只改 move_zero_to：利用新機制讓 0 一步跳多格，減少點擊次數
  - apply_step 改用新的滑動語意

關鍵差異（新 vs 舊）：
  舊：apply_step 只移動緊鄰 0 的那一格，0 靠近 tile 需多步
  新：apply_step 可以從遠處點擊同行/列，整排滑動，0 一步跳到位
  -> 主要節省在「移動 0 到 push_from」的點擊次數
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time, re
from collections import deque

# =========================================================
# 瀏覽器環境
# =========================================================
class WebPuzzleEnv:
    def __init__(self, n=5):
        self.n = n
        self.driver = webdriver.Chrome()
        self.driver.get("https://superglutenman0312.github.io/number_puzzle/")
        self._select_game_mode()
        self.click_count = 0

    def _select_game_mode(self):
        try:
            wait = WebDriverWait(self.driver, 10)
            mode_text = f"{self.n} x {self.n} 拼圖"
            span = wait.until(EC.element_to_be_clickable(
                (By.XPATH, f"//span[contains(text(), '{mode_text}')]")))
            span.click()
            time.sleep(0.5)
            btn = wait.until(EC.element_to_be_clickable((By.ID, "modal-action-btn")))
            btn.click()
            time.sleep(1)
        except Exception: pass

    def get_state(self, name="AI_Bot"):
        try:
            alert = self.driver.switch_to.alert
            alert.send_keys(name)
            alert.accept()
            time.sleep(0.3)
        except Exception: pass

        grid = [[0]*self.n for _ in range(self.n)]
        for tile in self.driver.find_elements(By.CLASS_NAME, "tile-wrapper"):
            raw_text = tile.text
            style = tile.get_attribute("style")
            if not raw_text or not style: continue
            text = raw_text.strip()
            if not text: continue

            unit = 100.0 / self.n
            tm = re.search(r"top:\s*([\d.]+)%", style)
            lm = re.search(r"left:\s*([\d.]+)%", style)
            if tm and lm:
                r = max(0, min(int(round(float(tm.group(1))/unit)), self.n-1))
                c = max(0, min(int(round(float(lm.group(1))/unit)), self.n-1))
                grid[r][c] = int(text)
        return [x for row in grid for x in row]

    def click_tile(self, val):
        try:
            self.driver.find_element(
                By.XPATH,
                f"//div[contains(@class,'tile-wrapper') and normalize-space()='{val}']"
            ).click()
            self.click_count += 1
            time.sleep(0.001)
        except Exception: pass


# =========================================================
# apply_step：新版滑動（0 與 click_idx 同行/列，整排滑動）
# =========================================================
def apply_step(env, state, n, click_idx):
    """
    點擊 click_idx 的方塊（必須與空格同行或同列）。
    所有介於兩者間的方塊整排往 0 方向滑一格，click_idx 變成新空格。
    """
    val = state[click_idx]
    if val == 0: return True
    z = state.index(0)
    rz, cz = z // n, z % n
    rt, ct = click_idx // n, click_idx % n
    if rt != rz and ct != cz: return False

    env.click_tile(val)

    if rt == rz:
        row = rt * n
        if ct < cz:   # tile 在左，整段右移
            state[row+ct+1:row+cz+1] = state[row+ct:row+cz]
            state[row+ct] = 0
        else:          # tile 在右，整段左移
            state[row+cz:row+ct] = state[row+cz+1:row+ct+1]
            state[row+ct] = 0
    else:
        if rt < rz:   # tile 在上，整段下移
            for r in range(rz, rt, -1): state[r*n+ct] = state[(r-1)*n+ct]
            state[rt*n+ct] = 0
        else:          # tile 在下，整段上移
            for r in range(rz, rt): state[r*n+ct] = state[(r+1)*n+ct]
            state[rt*n+ct] = 0
    return True


# =========================================================
# 0 的路徑規劃（新機制：沿行/列跳，但不穿越 forbidden）
# ===========================================================
# 「forbidden」= locked 格 + 正在被推的 tile
# 0 沿一個方向延伸，遇到 forbidden 就停（不可到達，也不能穿越）
# =========================================================
def _zero_slide_targets(pos, n, forbidden):
    r, c = pos // n, pos % n
    targets = []
    for dc in range(1, n-c):
        nxt = r*n+c+dc
        if nxt in forbidden: break
        targets.append(nxt)
    for dc in range(1, c+1):
        nxt = r*n+c-dc
        if nxt in forbidden: break
        targets.append(nxt)
    for dr in range(1, n-r):
        nxt = (r+dr)*n+c
        if nxt in forbidden: break
        targets.append(nxt)
    for dr in range(1, r+1):
        nxt = (r-dr)*n+c
        if nxt in forbidden: break
        targets.append(nxt)
    return targets


def find_zero_path(state, target_idx, n, forbidden):
    """BFS：0 到 target 的最少點擊路徑，不穿越 forbidden。"""
    start = state.index(0)
    if start == target_idx: return []
    q = deque([(start, [])])
    vis = {start}
    while q:
        cur, path = q.popleft()
        for nxt in _zero_slide_targets(cur, n, forbidden):
            np2 = path + [nxt]
            if nxt == target_idx: return np2
            if nxt not in vis:
                vis.add(nxt)
                q.append((nxt, np2))
    return None


def move_zero_to(env, state, target_idx, locked, n):
    path = find_zero_path(state, target_idx, n, locked)
    if path is None: return False
    for p in path:
        apply_step(env, state, n, p)
    return True


# =========================================================
# move_tile：搬動一個 tile 到目標位置
# 邏輯與 manual_solver.py 完全相同，只是 move_zero_to 更快
# =========================================================
def get_path_to(start_idx, target_idx, n, forbidden):
    """BFS：格子移動路徑（每步一格相鄰），forbidden 不可進入。"""
    if start_idx == target_idx: return []
    q = deque([(start_idx, [])])
    vis = {start_idx} | set(forbidden)
    while q:
        cur, path = q.popleft()
        r, c = cur // n, cur % n
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0<=nr<n and 0<=nc<n:
                nxt = nr*n+nc
                if nxt == target_idx: return path + [nxt]
                if nxt not in vis:
                    vis.add(nxt)
                    q.append((nxt, path+[nxt]))
    return None


def move_tile(env, state, tile_val, target_idx, locked, n):
    """
    將 tile_val 推到 target_idx（與 manual_solver 相同邏輯）。
    改進點：find_zero_path 利用滑動機制，移動 0 的點擊次數更少。
    """
    max_iter = n * n * 8
    for _ in range(max_iter):
        curr_idx = state.index(tile_val)
        if curr_idx == target_idx: break

        path = get_path_to(curr_idx, target_idx, n, locked)
        if not path: return False

        next_pos = path[0]
        # 讓 0 繞到 tile 前方（next_pos），不能穿越 tile 本身
        zero_path = find_zero_path(state, next_pos, n, locked | {curr_idx})

        if zero_path is not None:
            for p in zero_path:
                apply_step(env, state, n, p)
            # 點擊 tile 的「索引」（新版遊戲需要索引，不是數值！）
            apply_step(env, state, n, state.index(tile_val))
        else:
            # 死結：讓 0 踩著 tile 擠過去一步，再重新計算
            loose_path = find_zero_path(state, next_pos, n, locked)
            if not loose_path: return False
            apply_step(env, state, n, loose_path[0])

    return state.index(tile_val) == target_idx



# =========================================================
# 降階法主解題（與 manual_solver 相同結構）
# =========================================================
def _print_board(state, n, prefix="   "):
    for i in range(n):
        print(prefix, state[i*n:(i+1)*n])


def solve_puzzle(env, state, n):
    locked = set()
    print(f"[Solver] n={n}")

    for step in range(n - 2):
        print(f"  [Layer {step+1}]")

        # ---- 填上面橫列 ----
        row_vals = [step*n + c + 1 for c in range(step, n)]
        normal_vals = row_vals[:-2]
        v1, v2 = row_vals[-2], row_vals[-1]
        i1, i2 = v1-1, v2-1

        for val in normal_vals:
            move_tile(env, state, val, val-1, locked, n)
            locked.add(val-1)

        if not (state.index(v1)==i1 and state.index(v2)==i2):
            if state.index(v2) in (i1, i2):
                move_tile(env, state, v2, i1+2*n, locked, n)
            move_tile(env, state, v1, i2, locked, n)
            locked.add(i2)
            locked.add(i1)
            move_tile(env, state, v2, i2+n, locked, n)
            locked.remove(i1)
            move_zero_to(env, state, i1, locked, n)
            apply_step(env, state, n, i2)
            apply_step(env, state, n, i2+n)
        locked.add(i1)
        locked.add(i2)

        # ---- 填左側直行 ----
        col_vals = [r*n + step + 1 for r in range(step+1, n)]
        normal_vals = col_vals[:-2]
        v1, v2 = col_vals[-2], col_vals[-1]
        i1, i2 = v1-1, v2-1

        for val in normal_vals:
            move_tile(env, state, val, val-1, locked, n)
            locked.add(val-1)

        if not (state.index(v1)==i1 and state.index(v2)==i2):
            if state.index(v2) in (i1, i2):
                move_tile(env, state, v2, i1+2, locked, n)
            move_tile(env, state, v1, i2, locked, n)
            locked.add(i2)
            locked.add(i1)
            move_tile(env, state, v2, i2+1, locked, n)
            locked.remove(i1)
            move_zero_to(env, state, i1, locked, n)
            apply_step(env, state, n, i2)
            apply_step(env, state, n, i2+1)
        locked.add(i1)
        locked.add(i2)

    # ---- 2x2 核心 ----
    print("  [Core]")
    _print_board(state, n)

    i1 = (n-2)*n + (n-2)
    i2 = (n-2)*n + (n-1)
    i3 = (n-1)*n + (n-2)
    i4 = (n-1)*n + (n-1)
    t1, t2, t3 = i1+1, i2+1, i3+1
    print(f"  target=[{t1},{t2},{t3},0] actual=[{state[i1]},{state[i2]},{state[i3]},{state[i4]}]")

    def core_done():
        return state[i1]==t1 and state[i2]==t2 and state[i3]==t3 and state[i4]==0

    # 把 0 拉進核心
    if state.index(0) not in (i1,i2,i3,i4):
        move_zero_to(env, state, i4, locked, n)

    # BFS 找最短旋轉序列（2x2 最多 12 種狀態）
    slots = [i1, i2, i3, i4]
    adj4 = {0:[1,2], 1:[0,3], 2:[0,3], 3:[1,2]}

    def ctup():
        return tuple(state[ix] for ix in slots)

    target4 = (t1, t2, t3, 0)
    if ctup() != target4:
        bq = deque([(ctup(), [])])
        bvis = {ctup()}
        seq = None
        while bq and not seq:
            tup, moves = bq.popleft()
            zs = tup.index(0)
            for ns in adj4[zs]:
                nl = list(tup)
                nl[zs], nl[ns] = nl[ns], nl[zs]
                nt = tuple(nl)
                nm = moves + [slots[ns]]
                if nt == target4:
                    seq = nm; break
                if nt not in bvis:
                    bvis.add(nt); bq.append((nt, nm))

        if seq:
            for ci in seq:
                apply_step(env, state, n, ci)

    return core_done()


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    n = 12
    env = WebPuzzleEnv(n=n)
    state = env.get_state()
    print(f"Initial: {state}")
    t0 = time.time()
    ok = solve_puzzle(env, state, n)
    elapsed = time.time() - t0
    if ok:
        print(f"\nSUCCESS! Clicks={env.click_count}, Time={elapsed:.1f}s")
        time.sleep(0.5)
        # 留名青史
        env.get_state(name=f"Psyduck_{n}x{n}")
    else:
        print(f"\nFAILED. Clicks={env.click_count}, Time={elapsed:.1f}s")
        _print_board(state, n)
    input("\nPress Enter to close...")
