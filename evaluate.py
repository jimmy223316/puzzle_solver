"""
evaluate.py — 模型評估腳本
==============================

功能：
1. 載入訓練好的 best_model.pth
2. 隨機生成多局 N×N 盤面
3. 模型推理 + 合法動作過濾
4. 統計各尺寸的解謎成功率

動作選擇策略：
  模型輸出 4 個 Logits → Softmax → 過濾掉撞牆的非法動作 → 選擇機率最高的合法動作

使用方式：
    python evaluate.py                          # 預設測試
    python evaluate.py --model best_model.pth --games 500
    python evaluate.py --sizes 3 4 5            # 只測試指定尺寸
"""

import argparse
import torch
import torch.nn.functional as F
from env import VirtualPuzzleEnv, ACTION_DELTAS
from generate_data import encode_state
from model import PuzzleSolverCNN


# =========================================================
# 模型推理：選擇最優合法動作
# =========================================================
@torch.no_grad()
def predict_action(model: PuzzleSolverCNN,
                   state: list,
                   n: int,
                   device: torch.device) -> int:
    """
    使用模型預測當前盤面應該執行的動作。
    
    流程：
    1. 將盤面編碼為 (3, 12, 12) 張量
    2. 模型推理得到 4 個 Logits
    3. 計算合法動作 mask
    4. 對非法動作設為 -inf
    5. 取 argmax 得到最優合法動作
    
    Args:
        model:  訓練好的 PuzzleSolverCNN
        state:  當前盤面 (1D list)
        n:      盤面尺寸
        device: 計算裝置
    
    Returns:
        action: 0~3
    """
    # 編碼盤面
    encoded = encode_state(state, n)              # (3, 12, 12)
    encoded = encoded.unsqueeze(0).to(device)     # (1, 3, 12, 12)

    # 模型推理
    logits = model(encoded)                       # (1, 4)
    logits = logits.squeeze(0)                    # (4,)

    # 計算合法動作
    zero_idx = state.index(0)
    r, c = zero_idx // n, zero_idx % n
    legal_mask = torch.full((4,), float("-inf"), device=device)  # 預設全部非法

    for action, (dr, dc) in ACTION_DELTAS.items():
        nr, nc = r + dr, c + dc
        if 0 <= nr < n and 0 <= nc < n:
            legal_mask[action] = 0.0  # 合法的動作不遮蔽

    # 遮蔽非法動作後取 argmax
    masked_logits = logits + legal_mask           # (4,)
    action = masked_logits.argmax().item()

    return action


# =========================================================
# 單局評估
# =========================================================
def evaluate_single_game(model: PuzzleSolverCNN,
                         n: int,
                         device: torch.device,
                         max_steps: int = None,
                         verbose: bool = False) -> tuple:
    """
    評估模型在一局隨機盤面上的表現。
    
    Args:
        model:     訓練好的模型
        n:         盤面尺寸
        device:    計算裝置
        max_steps: 最大步數限制（防止無限循環），None 則自動計算
        verbose:   是否印出每一步
    
    Returns:
        (success, steps): 是否成功解開，以及走了多少步
    """
    if max_steps is None:
        # 經驗公式：大尺寸允許更多步
        max_steps = n * n * 50

    env = VirtualPuzzleEnv(n)
    env.reset()

    if verbose:
        print(f"\n🧩 盤面 {n}×{n} | 最大步數: {max_steps}")
        _print_board(env.get_state(), n)

    for step in range(max_steps):
        if env.is_solved():
            if verbose:
                print(f"  ✅ 成功！步數: {step}")
            return True, step

        state = env.get_state()
        action = predict_action(model, state, n, device)

        env.step(action)

        if verbose and step < 20:  # 只印前 20 步
            action_names = {0: "↑上", 1: "↓下", 2: "←左", 3: "→右"}
            print(f"  Step {step+1}: {action_names[action]}")

    # 最後一次檢查
    if env.is_solved():
        return True, max_steps

    if verbose:
        print(f"  ❌ 達到步數上限 ({max_steps})，未能解開")
        _print_board(env.get_state(), n)

    return False, max_steps


def _print_board(state: list, n: int):
    """印出盤面（格式化）。"""
    width = len(str(n * n))
    print("  ┌" + "─" * ((width + 1) * n + 1) + "┐")
    for r in range(n):
        row_str = "  │ "
        for c in range(n):
            val = state[r * n + c]
            if val == 0:
                row_str += " " * width + " "
            else:
                row_str += f"{val:>{width}d} "
        row_str += "│"
        print(row_str)
    print("  └" + "─" * ((width + 1) * n + 1) + "┘")


# =========================================================
# 批量評估
# =========================================================
def evaluate_batch(model: PuzzleSolverCNN,
                   sizes: list,
                   games_per_size: int,
                   device: torch.device) -> dict:
    """
    對多個尺寸進行批量評估。
    
    Args:
        model:          訓練好的模型
        sizes:          要測試的尺寸列表 (e.g., [3, 4, 5, ...])
        games_per_size: 每個尺寸測試幾局
        device:         計算裝置
    
    Returns:
        results: {n: {"success": int, "total": int, "avg_steps": float}}
    """
    results = {}

    print(f"\n{'='*60}")
    print(f"🔬 批量評估 | 尺寸: {sizes} | 每尺寸 {games_per_size} 局")
    print(f"{'='*60}")

    for n in sizes:
        success_count = 0
        total_steps = 0

        for game in range(games_per_size):
            success, steps = evaluate_single_game(model, n, device)

            if success:
                success_count += 1
                total_steps += steps

            # 進度
            if (game + 1) % max(1, games_per_size // 5) == 0:
                rate = success_count / (game + 1) * 100
                print(f"  {n}×{n}: {game+1}/{games_per_size} 局 | 成功率: {rate:.1f}%")

        avg_steps = total_steps / success_count if success_count > 0 else float("inf")
        results[n] = {
            "success": success_count,
            "total": games_per_size,
            "avg_steps": avg_steps,
        }

    # 印出總結
    print(f"\n{'='*60}")
    print(f"📊 評估結果彙總")
    print(f"{'='*60}")
    print(f"  {'尺寸':>6s} | {'成功':>5s}/{' 總共':<5s} | {'成功率':>7s} | {'平均步數':>8s}")
    print(f"  {'─'*6} | {'─'*11} | {'─'*7} | {'─'*8}")
    for n_size in sorted(results.keys()):
        r = results[n_size]
        rate = r["success"] / r["total"] * 100
        avg = f"{r['avg_steps']:.0f}" if r["success"] > 0 else "N/A"
        print(f"  {n_size:>3d}×{n_size:<2d} | {r['success']:>5d}/{r['total']:<5d} | {rate:>6.1f}% | {avg:>8s}")

    return results


# =========================================================
# 入口點
# =========================================================
def main(args):
    # 裝置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  使用裝置: {device}")

    # 載入模型
    print(f"📂 載入模型: {args.model}")
    checkpoint = torch.load(args.model, map_location=device, weights_only=True)
    model = PuzzleSolverCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"   模型來自 Epoch {checkpoint['epoch']}, Val Acc: {checkpoint['val_acc']*100:.2f}%")

    # 示範一局（帶詳細輸出）
    if args.demo:
        demo_n = args.sizes[0] if args.sizes else 4
        print(f"\n🎮 示範模式：{demo_n}×{demo_n}")
        evaluate_single_game(model, demo_n, device, verbose=True)

    # 批量評估
    sizes = args.sizes if args.sizes else list(range(3, 13))
    evaluate_batch(model, sizes, args.games, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="華容道模仿學習 — 評估腳本")
    parser.add_argument("--model",  type=str,   default="best_model.pth",  help="模型路徑")
    parser.add_argument("--games",  type=int,   default=30,               help="每個尺寸測試幾局")
    parser.add_argument("--sizes",  type=int,   nargs="+",                 help="要測試的尺寸列表")
    parser.add_argument("--demo",   action="store_true",                   help="示範模式（印出詳細步驟）")
    args = parser.parse_args()

    main(args)
