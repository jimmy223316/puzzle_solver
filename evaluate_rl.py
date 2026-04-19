"""
evaluate_rl.py — BC vs RL 對比評估 + 圖表
==========================================

功能：
1. 對相同的隨機盤面，分別用 BC 模型與 RL 微調模型解題
2. 統計成功率、平均步數、步數中位數
3. 輸出 matplotlib 比較圖表

使用方式：
    python evaluate_rl.py                                          # 使用預設路徑
    python evaluate_rl.py --bc best_model.pth --rl best_rl_model.pth
    python evaluate_rl.py --games 50 --sizes 3 4 5 6 7
"""

import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F

from env import VirtualPuzzleEnv, ACTION_DELTAS
from generate_data import encode_state
from model import PuzzleSolverCNN
from model_rl import ActorCriticCNN

MAX_N = 12


# =========================================================
# 合法動作 Mask 工具
# =========================================================
def get_legal_mask_tensor(state: list, n: int, device) -> torch.Tensor:
    """回傳 (4,) bool Tensor，True=合法動作（空格不會撞牆）。"""
    zero_idx = state.index(0)
    r, c = zero_idx // n, zero_idx % n
    mask = torch.zeros(4, dtype=torch.bool, device=device)
    for action, (dr, dc) in ACTION_DELTAS.items():
        nr, nc = r + dr, c + dc
        if 0 <= nr < n and 0 <= nc < n:
            mask[action] = True
    return mask


# =========================================================
# 單模型評估（通用，支援 BC 和 RL 兩種架構）
# =========================================================
@torch.no_grad()
def run_episode(model, model_type: str, initial_state: list, n: int,
                device, max_steps: int = None) -> dict:
    """
    從指定初始盤面出發，讓模型嘗試解題。

    Args:
        model:         BC 或 RL 模型
        model_type:    'bc' | 'rl'
        initial_state: 初始盤面（1D list，長度 N²）
        n:             盤面尺寸
        device:        計算裝置
        max_steps:     步數上限（None 則自動設定）

    Returns:
        dict: success, steps, n
    """
    if max_steps is None:
        max_steps = n * n * 40

    model.eval()
    state = list(initial_state)  # 複製，不污染原始盤面
    goal  = list(range(1, n * n)) + [0]

    for step in range(max_steps):
        if state == goal:
            return {"success": True, "steps": step, "n": n}

        encoded = encode_state(state, n).unsqueeze(0).to(device)  # (1, 3, 12, 12)
        legal_mask = get_legal_mask_tensor(state, n, device)       # (4,)

        if model_type == "bc":
            logits = model(encoded).squeeze(0)                     # (4,)
            masked = logits + legal_mask.float().masked_fill(~legal_mask, float('-inf'))
            # 換個寫法：非法置 -inf
            full_mask = torch.full((4,), float('-inf'), device=device)
            full_mask[legal_mask] = 0.0
            action = (logits + full_mask).argmax().item()

        else:  # rl
            logits, _ = model(encoded)
            logits = logits.squeeze(0)                              # (4,)
            mask_t = legal_mask.unsqueeze(0)                        # (1, 4)
            logits = logits.masked_fill(~legal_mask, float('-inf'))
            action = logits.argmax().item()

        # 執行動作
        zero_idx = state.index(0)
        dr, dc = ACTION_DELTAS[action]
        r, c = zero_idx // n, zero_idx % n
        nr, nc = r + dr, c + dc
        new_idx = nr * n + nc
        state[zero_idx], state[new_idx] = state[new_idx], state[zero_idx]

    # 最後一次檢查
    if state == goal:
        return {"success": True, "steps": max_steps, "n": n}
    return {"success": False, "steps": max_steps, "n": n}


# =========================================================
# 批次評估（公平：兩個模型用同一批盤面）
# =========================================================
def evaluate_size(bc_model, rl_model, n: int, num_games: int,
                  device) -> dict:
    """
    對尺寸 n 生成 num_games 個相同盤面，分別讓 BC 和 RL 模型解題。

    Returns:
        dict: {
            'bc':  {'success': N, 'steps': [...]},
            'rl':  {'success': N, 'steps': [...]},
        }
    """
    results = {
        "bc": {"success": 0, "steps": []},
        "rl": {"success": 0, "steps": []},
    }

    for game_i in range(num_games):
        # 生成初始盤面（兩個模型用同一個）
        env = VirtualPuzzleEnv(n)
        env.reset()
        initial_state = env.get_state()

        for model_name, model, mtype in [
            ("bc", bc_model, "bc"), ("rl", rl_model, "rl")
        ]:
            res = run_episode(model, mtype, initial_state, n, device)
            if res["success"]:
                results[model_name]["success"] += 1
                results[model_name]["steps"].append(res["steps"])

    return results


# =========================================================
# matplotlib 圖表
# =========================================================
def plot_comparison(all_results: dict, sizes: list,
                    output_prefix: str = "comparison"):
    """
    繪製 BC vs RL 對比圖。

    圖 1：成功率比較柱狀圖（每個尺寸一組）
    圖 2：平均步數比較柱狀圖
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # 無 GUI 後端（伺服器/後台友好）
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("⚠️  找不到 matplotlib，跳過圖表輸出。請執行：pip install matplotlib")
        return

    x = np.arange(len(sizes))
    width = 0.35
    colors = {"bc": "#4C9BE8", "rl": "#F4845F"}  # 柔和的藍/橘

    # ---- 圖 1：成功率 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("BC vs RL 模型比較", fontsize=16, fontweight="bold")

    ax1 = axes[0]
    bc_succ = [all_results[n]["bc"]["success"] / max(all_results[n]["total"], 1) * 100
               for n in sizes]
    rl_succ = [all_results[n]["rl"]["success"] / max(all_results[n]["total"], 1) * 100
               for n in sizes]

    bars1 = ax1.bar(x - width/2, bc_succ, width, label="BC 行為複製", color=colors["bc"],
                    alpha=0.85, edgecolor="white")
    bars2 = ax1.bar(x + width/2, rl_succ, width, label="RL PPO 微調", color=colors["rl"],
                    alpha=0.85, edgecolor="white")
    ax1.set_xlabel("盤面尺寸 N")
    ax1.set_ylabel("成功率 (%)")
    ax1.set_title("解題成功率")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{n}×{n}" for n in sizes])
    ax1.set_ylim(0, 110)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    # 在柱子上標數字
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.0f}%",
                 ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.0f}%",
                 ha="center", va="bottom", fontsize=8, color="#c0392b")

    # ---- 圖 2：平均步數 ----
    ax2 = axes[1]
    bc_steps = []
    rl_steps = []
    for n in sizes:
        bc_s = all_results[n]["bc"]["steps"]
        rl_s = all_results[n]["rl"]["steps"]
        bc_steps.append(np.mean(bc_s) if bc_s else 0)
        rl_steps.append(np.mean(rl_s) if rl_s else 0)

    bars3 = ax2.bar(x - width/2, bc_steps, width, label="BC 行為複製", color=colors["bc"],
                    alpha=0.85, edgecolor="white")
    bars4 = ax2.bar(x + width/2, rl_steps, width, label="RL PPO 微調", color=colors["rl"],
                    alpha=0.85, edgecolor="white")
    ax2.set_xlabel("盤面尺寸 N")
    ax2.set_ylabel("平均步數（成功局）")
    ax2.set_title("解題平均步數（越少越好）")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{n}×{n}" for n in sizes])
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    for bar in bars3:
        h = bar.get_height()
        if h > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.0f}",
                     ha="center", va="bottom", fontsize=8)
    for bar in bars4:
        h = bar.get_height()
        if h > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.0f}",
                     ha="center", va="bottom", fontsize=8, color="#c0392b")

    plt.tight_layout()
    out_path = f"{output_prefix}_bar.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  📊 比較圖表已儲存：{out_path}")
    plt.close()


# =========================================================
# 主流程
# =========================================================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  使用裝置: {device}\n")

    # ---- 載入 BC 模型 ----
    print(f"📂 載入 BC 模型: {args.bc}")
    bc_model = PuzzleSolverCNN().to(device)
    bc_ckpt  = torch.load(args.bc, map_location=device, weights_only=True)
    bc_model.load_state_dict(bc_ckpt["model_state_dict"])
    bc_model.eval()
    print(f"   BC Epoch={bc_ckpt.get('epoch','?')}, "
          f"Val Acc={bc_ckpt.get('val_acc',0)*100:.2f}%")

    # ---- 載入 RL 模型 ----
    print(f"📂 載入 RL 模型: {args.rl}")
    rl_model = ActorCriticCNN().to(device)
    try:
        rl_ckpt  = torch.load(args.rl, map_location=device, weights_only=True)
        rl_model.load_state_dict(rl_ckpt["model_state_dict"])
        rl_model.eval()
        print(f"   RL Update={rl_ckpt.get('update','?')}, "
              f"MeanR={rl_ckpt.get('mean_reward',0):.2f}, "
              f"Succ%={rl_ckpt.get('success_rate',0):.1f}%")
    except FileNotFoundError:
        print(f"⚠️  找不到 {args.rl}，請先完成 RL 訓練！")
        return

    sizes = args.sizes or [3, 4, 5, 6, 7]
    print(f"\n🔬 評估設定：尺寸={sizes}, 每尺寸={args.games}局\n")

    all_results = {}

    print(f"{'尺寸':>5s} | {'模型':>5s} | {'成功':>5s}/{' 總計':<5s} | "
          f"{'成功率':>7s} | {'平均步數':>8s} | {'中位步數':>8s}")
    print("─" * 60)

    for n in sizes:
        results = evaluate_size(bc_model, rl_model, n, args.games, device)
        all_results[n] = {**results, "total": args.games}

        for label in ["bc", "rl"]:
            succ  = results[label]["success"]
            steps = results[label]["steps"]
            rate  = succ / args.games * 100
            avg_s = np.mean(steps) if steps else float("nan")
            med_s = np.median(steps) if steps else float("nan")
            tag   = "BC" if label == "bc" else "RL"
            print(f"  {n}×{n} | {tag:>5s} | {succ:>5d}/{args.games:<5d} | "
                  f"{rate:>6.1f}% | {avg_s:>8.1f} | {med_s:>8.1f}")

        # 計算步數優化比率
        bc_s = results["bc"]["steps"]
        rl_s = results["rl"]["steps"]
        if bc_s and rl_s:
            improve = (np.mean(bc_s) - np.mean(rl_s)) / np.mean(bc_s) * 100
            print(f"        → RL步數比BC {'改善' if improve > 0 else '退步'} {abs(improve):.1f}%")
        print()

    # ---- 繪圖 ----
    plot_comparison(all_results, sizes, output_prefix="rl_comparison")

    print("\n✅ 評估完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BC vs RL 模型對比評估")
    parser.add_argument("--bc",    type=str,       default="best_model.pth",    help="BC 模型路徑")
    parser.add_argument("--rl",    type=str,       default="best_rl_model.pth", help="RL 模型路徑")
    parser.add_argument("--games", type=int,       default=30,                  help="每個尺寸測試幾局")
    parser.add_argument("--sizes", type=int, nargs="+",                         help="測試尺寸列表")
    args = parser.parse_args()
    main(args)
