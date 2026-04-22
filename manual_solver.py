from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import re
from collections import deque

class WebPuzzleEnv:
    def __init__(self, n=3):
        self.n = n
        self.driver = webdriver.Chrome()
        self.driver.get("https://superglutenman0312.github.io/number_puzzle/")
        self._select_game_mode()
        
    def _select_game_mode(self):
        try:
            wait = WebDriverWait(self.driver, 10)
            mode_text = f"{self.n} x {self.n} 拼圖"
            target_span = wait.until(EC.element_to_be_clickable((By.XPATH, f"//span[contains(text(), '{mode_text}')]")))
            target_span.click()
            time.sleep(0.5) 
            start_btn = wait.until(EC.element_to_be_clickable((By.ID, "modal-action-btn")))
            start_btn.click()
            time.sleep(1)
        except Exception: pass

    def get_state(self, name="AI_Bot"):
        try:
            alert = self.driver.switch_to.alert
            alert.send_keys(name)
            alert.accept()
            time.sleep(0.3)
        except Exception: pass
        
        grid = [[0 for _ in range(self.n)] for _ in range(self.n)]
        tiles = self.driver.find_elements(By.CLASS_NAME, "tile-wrapper")
        unit_pct = 100.0 / self.n
        for tile in tiles:
            text = tile.text.strip()
            style = tile.get_attribute("style")
            if not text: continue
            num = int(text)
            top_match = re.search(r"top:\s*([\d.]+)%", style)
            left_match = re.search(r"left:\s*([\d.]+)%", style)
            if top_match and left_match:
                r = int(round(float(top_match.group(1)) / unit_pct))
                c = int(round(float(left_match.group(1)) / unit_pct))
                grid[min(r, self.n-1)][min(c, self.n-1)] = num
        return [item for sublist in grid for item in sublist]

    def click_tile(self, target_number):
        try:
            xpath = f"//div[contains(@class, 'tile-wrapper') and normalize-space()='{target_number}']"
            self.driver.find_element(By.XPATH, xpath).click()
            # 【關鍵修復】加入微小延遲，防止 JS 事件監聽器來不及處理極速連點而漏掉動作
            time.sleep(0.001) 
        except Exception: pass

# ---------------------------------------------------------
# 內部狀態追蹤器與 BFS 演算法 
# ---------------------------------------------------------
def apply_step(env, state, num_to_click):
    env.click_tile(num_to_click)
    z_idx = state.index(0)
    t_idx = state.index(num_to_click)
    state[z_idx], state[t_idx] = state[t_idx], state[z_idx]

def get_path_to(start_idx, target_idx, n, forbidden):
    if start_idx == target_idx: return []
    q = deque([(start_idx, [])])
    visited = {start_idx} | set(forbidden)
    while q:
        curr, path = q.popleft()
        r, c = curr // n, curr % n
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r+dr, c+dc
            if 0<=nr<n and 0<=nc<n:
                nxt = nr*n + nc
                if nxt == target_idx: return path + [nxt]
                if nxt not in visited:
                    visited.add(nxt)
                    q.append((nxt, path + [nxt]))
    return None

def move_zero_to(env, state, target_idx, locked, n):
    path = get_path_to(state.index(0), target_idx, n, locked)
    if path is None: return False
    for p in path:
        apply_step(env, state, state[p])
    return True

def move_tile(env, state, tile_val, target_idx, locked, n):
    """【徹底修復】防止瞬間移動 Bug 的穩健推動法"""
    while True:
        curr_idx = state.index(tile_val)
        if curr_idx == target_idx: break
        
        path = get_path_to(curr_idx, target_idx, n, locked)
        if not path: return False
        
        next_pos = path[0]
        # 嘗試讓 0 繞過目標方塊到達前方
        zero_path = get_path_to(state.index(0), next_pos, n, locked | {curr_idx})
        
        if zero_path is not None:
            # 安全路徑：0 成功繞過去了，推動方塊
            for p in zero_path:
                apply_step(env, state, state[p])
            apply_step(env, state, tile_val)
        else:
            # 死結路徑：0 繞不過去，被目標方塊擋死了。
            # 解法：讓 0 踩著目標方塊「擠」過去一步，然後迴圈重新計算全新路徑！
            loose_path = get_path_to(state.index(0), next_pos, n, locked)
            if not loose_path: return False
            apply_step(env, state, state[loose_path[0]])
            
    return True

# ---------------------------------------------------------
# 主演算法：降階法 (自訂陣列依序法)
# ---------------------------------------------------------
def solve_puzzle(env, state, n):
    locked = set()
    print("\n🚀 開始極速解謎 (自訂陣列依序法) ...")

    # 1. 一層一層剝開 (直到剩下 2x2)
    for step in range(n - 2):
        
        # --- A. 產生這圈上面的 Row ---
        row_vals = [step * n + c + 1 for c in range(step, n)]
        print(f"  👉 準備放上面 Row: {row_vals}")
        
        normal_vals = row_vals[:-2]
        val1, val2 = row_vals[-2], row_vals[-1]
        idx1, idx2 = val1 - 1, val2 - 1  
        
        for val in normal_vals:
            move_tile(env, state, val, val - 1, locked, n)
            locked.add(val - 1)
            
        if not (state.index(val1) == idx1 and state.index(val2) == idx2):
            # 防死角：先把 val2 移出危險區 (往下放兩格以確保安全)
            if state.index(val2) in [idx1, idx2]:
                move_tile(env, state, val2, idx1 + 2*n, locked, n)

            move_tile(env, state, val1, idx2, locked, n)
            locked.add(idx2)
            
            # 🚧 關鍵修復：暫時封鎖 idx1 這個死胡同，避免 0 在推 val2 時跑進去卡死！
            locked.add(idx1) 
            move_tile(env, state, val2, idx2 + n, locked, n) 
            locked.add(idx2 + n)

            locked.remove(idx1) # 取消封鎖
            move_zero_to(env, state, idx1, locked, n)
            locked.remove(idx2)
            locked.remove(idx2 + n)
            apply_step(env, state, val1)
            apply_step(env, state, val2)
        
        locked.add(idx1)
        locked.add(idx2)

        # --- B. 產生這圈左邊的 Col ---
        col_vals = [r * n + step + 1 for r in range(step + 1, n)]
        print(f"  👉 準備放左邊 Col: {col_vals}")
        
        normal_vals = col_vals[:-2]
        val1, val2 = col_vals[-2], col_vals[-1]
        idx1, idx2 = val1 - 1, val2 - 1
        
        for val in normal_vals:
            move_tile(env, state, val, val - 1, locked, n)
            locked.add(val - 1)
            
        if not (state.index(val1) == idx1 and state.index(val2) == idx2):
            # 防死角：先把 val2 移出危險區 (往右放兩格以確保安全)
            if state.index(val2) in [idx1, idx2]:
                move_tile(env, state, val2, idx1 + 2, locked, n)

            move_tile(env, state, val1, idx2, locked, n)
            locked.add(idx2)
            
            # 🚧 關鍵修復：暫時封鎖 idx1 這個死胡同
            locked.add(idx1)
            move_tile(env, state, val2, idx2 + 1, locked, n) 
            locked.add(idx2 + 1)

            locked.remove(idx1) # 取消封鎖
            move_zero_to(env, state, idx1, locked, n)
            locked.remove(idx2)
            locked.remove(idx2 + 1)
            apply_step(env, state, val1)
            apply_step(env, state, val2)
            
        locked.add(idx1)
        locked.add(idx2)

    # 2. 處理最後剩下的 2x2 核心
    print("  🎯 處理最後的 2x2 核心...")
    i1, i2 = (n-2)*n + n-2, (n-2)*n + n-1
    i3, i4 = (n-1)*n + n-2, (n-1)*n + n-1
    t1, t2, t3 = i1 + 1, i2 + 1, i3 + 1

    for _ in range(20):
        if state[i1] == t1 and state[i2] == t2 and state[i3] == t3:
            if state[i4] != 0: move_zero_to(env, state, i4, set(), n)
            return True
        
        z_idx = state.index(0)
        if z_idx == i1: apply_step(env, state, state[i2])
        elif z_idx == i2: apply_step(env, state, state[i4])
        elif z_idx == i4: apply_step(env, state, state[i3])
        elif z_idx == i3: apply_step(env, state, state[i1])
        else: move_zero_to(env, state, i4, locked, n)
        
    return False

if __name__ == "__main__":
    size = 5 # 7x7 大挑戰！
    env = WebPuzzleEnv(n=size)
    
    initial_state = env.get_state()
    print(f"初始狀態: {initial_state}")
    
    success = solve_puzzle(env, initial_state, size)
    
    if success:
        print("🎉 完美過關！")
        time.sleep(0.5)
        # 留名青史
        env.get_state(name=f"Psyduck_{size}x{size}")
    else:
        print("❌ 解題失敗，可能卡住了。")
        
    input("Enter 關閉...")