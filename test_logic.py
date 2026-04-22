"""
test_logic.py v2 - 純邏輯測試（不開瀏覽器）
"""
from collections import deque

class FakeEnv:
    def __init__(self): self.click_count = 0
    def click_tile(self, val): self.click_count += 1

def apply_step(env, state, n, click_idx):
    val = state[click_idx]
    if val == 0: return True
    z = state.index(0)
    rz, cz = z // n, z % n
    rt, ct = click_idx // n, click_idx % n
    if rt != rz and ct != cz: return False
    env.click_tile(val)
    if rt == rz:
        row = rt * n
        if ct < cz:
            state[row+ct+1:row+cz+1] = state[row+ct:row+cz]
            state[row+ct] = 0
        else:
            state[row+cz:row+ct] = state[row+cz+1:row+ct+1]
            state[row+ct] = 0
    else:
        if rt < rz:
            for r in range(rz, rt, -1): state[r*n+ct] = state[(r-1)*n+ct]
            state[rt*n+ct] = 0
        else:
            for r in range(rz, rt): state[r*n+ct] = state[(r+1)*n+ct]
            state[rt*n+ct] = 0
    return True

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

def get_path_to(start_idx, target_idx, n, forbidden):
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
    max_iter = n * n * 8
    for _ in range(max_iter):
        curr_idx = state.index(tile_val)
        if curr_idx == target_idx: break
        path = get_path_to(curr_idx, target_idx, n, locked)
        if not path: return False
        next_pos = path[0]
        zero_path = find_zero_path(state, next_pos, n, locked | {curr_idx})
        if zero_path is not None:
            for p in zero_path:
                apply_step(env, state, n, p)
            # 點擊 tile 的 index（不是數值！）
            apply_step(env, state, n, state.index(tile_val))
        else:
            loose_path = find_zero_path(state, next_pos, n, locked)
            if not loose_path: return False
            apply_step(env, state, n, loose_path[0])
    return state.index(tile_val) == target_idx

def _print_board(state, n):
    for i in range(n): print(f"  {state[i*n:(i+1)*n]}")

def solve_puzzle(env, state, n):
    locked = set()
    print(f"[Solver] n={n}")

    for step in range(n - 2):
        print(f"  [Layer {step+1}]")
        row_vals = [step*n + c + 1 for c in range(step, n)]
        normal_vals = row_vals[:-2]
        v1, v2 = row_vals[-2], row_vals[-1]
        i1, i2 = v1-1, v2-1

        for val in normal_vals:
            ok = move_tile(env, state, val, val-1, locked, n)
            if not ok: print(f"    WARN: failed to place {val}")
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

        col_vals = [r*n + step + 1 for r in range(step+1, n)]
        normal_vals = col_vals[:-2]
        v1, v2 = col_vals[-2], col_vals[-1]
        i1, i2 = v1-1, v2-1

        for val in normal_vals:
            ok = move_tile(env, state, val, val-1, locked, n)
            if not ok: print(f"    WARN: failed to place {val}")
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

    if state.index(0) not in (i1,i2,i3,i4):
        move_zero_to(env, state, i4, locked, n)

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


# ---- Run multiple test cases ----
import random
if __name__ == "__main__":
    n = 5
    goal = list(range(1, n*n)) + [0]
    passed = 0
    failed = 0
    total_clicks = 0

    for trial in range(10):
        state = goal[:]
        # Shuffle by making random valid moves
        z = state.index(0)
        for _ in range(500):
            r, c = z // n, z % n
            moves = []
            if r > 0: moves.append(z - n)
            if r < n-1: moves.append(z + n)
            if c > 0: moves.append(z - 1)
            if c < n-1: moves.append(z + 1)
            pick = random.choice(moves)
            state[z], state[pick] = state[pick], state[z]
            z = pick

        env = FakeEnv()
        print(f"\n=== Trial {trial+1} ===")
        print(f"Initial: {state}")
        state_copy = state[:]
        ok = solve_puzzle(env, state_copy, n)
        if ok:
            print(f"SUCCESS! Clicks={env.click_count}")
            passed += 1
            total_clicks += env.click_count
        else:
            print(f"FAILED! Clicks={env.click_count}")
            _print_board(state_copy, n)
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed}/{passed+failed} passed")
    if passed > 0:
        print(f"Avg clicks (success): {total_clicks//passed}")
