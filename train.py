"""
train.py — 訓練迴圈
=======================

功能：
1. 從 dataset.pt 載入資料
2. 80/20 拆分為訓練集/驗證集
3. 使用 CrossEntropyLoss + AdamW + CosineAnnealingLR 訓練
4. 實作 Early Stopping（patience=5）
5. 存檔最佳模型為 best_model.pth

使用方式：
    python train.py                      # 使用預設參數
    python train.py --epochs 100         # 自訂 epoch 數
    python train.py --batch_size 2048    # 自訂 batch size
"""

import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from model import PuzzleSolverCNN


# =========================================================
# PyTorch Dataset
# =========================================================
class PuzzleDataset(Dataset):
    """
    華容道 (State, Action) 資料集。
    
    從 dataset.pt 讀取：
      - states:  Tensor[M, 3, 12, 12] (float32)
      - actions: Tensor[M]            (int64)
    """

    def __init__(self, states: torch.Tensor, actions: torch.Tensor):
        """
        Args:
            states:  (M, 3, 12, 12) 已編碼的盤面特徵
            actions: (M,) 對應的專家動作 (0~3)
        """
        self.states = states
        self.actions = actions

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns:
            (state, action): state 為 (3,12,12) float32, action 為 int64 scalar
        """
        return self.states[idx], self.actions[idx]


# =========================================================
# 訓練一個 Epoch
# =========================================================
def train_one_epoch(model: nn.Module,
                    dataloader: DataLoader,
                    criterion: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device) -> tuple:
    """
    訓練一個 epoch。
    
    Returns:
        (avg_loss, accuracy): 平均損失 和 準確率
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_states, batch_actions in dataloader:
        # batch_states: (B, 3, 12, 12)
        # batch_actions: (B,)
        batch_states = batch_states.to(device)
        batch_actions = batch_actions.to(device)

        optimizer.zero_grad()

        logits = model(batch_states)               # (B, 4)
        loss = criterion(logits, batch_actions)    # scalar

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_states.size(0)
        predictions = logits.argmax(dim=1)         # (B,)
        correct += (predictions == batch_actions).sum().item()
        total += batch_states.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# =========================================================
# 驗證一個 Epoch
# =========================================================
@torch.no_grad()
def validate(model: nn.Module,
             dataloader: DataLoader,
             criterion: nn.Module,
             device: torch.device) -> tuple:
    """
    驗證（不計算梯度）。
    
    Returns:
        (avg_loss, accuracy): 平均損失 和 準確率
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_states, batch_actions in dataloader:
        batch_states = batch_states.to(device)
        batch_actions = batch_actions.to(device)

        logits = model(batch_states)
        loss = criterion(logits, batch_actions)

        total_loss += loss.item() * batch_states.size(0)
        predictions = logits.argmax(dim=1)
        correct += (predictions == batch_actions).sum().item()
        total += batch_states.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# =========================================================
# 主訓練流程
# =========================================================
def main(args):
    # ------ 1. 裝置偵測 ------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  使用裝置: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # ------ 2. 載入資料 ------
    print(f"\n📂 載入資料集: {args.data_path}")
    data = torch.load(args.data_path, weights_only=True)
    states = data["states"]     # (M, 3, 12, 12)
    actions = data["actions"]   # (M,)
    print(f"   States shape:  {list(states.shape)}")
    print(f"   Actions shape: {list(actions.shape)}")
    print(f"   Action 分佈: {torch.bincount(actions).tolist()}")

    # ------ 3. Train/Val Split (80/20) ------
    full_dataset = PuzzleDataset(states, actions)
    total_size = len(full_dataset)
    train_size = int(total_size * 0.8)
    val_size = total_size - train_size

    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # 固定分割種子
    )
    print(f"\n📊 資料拆分：")
    print(f"   訓練集: {train_size:>10,} samples")
    print(f"   驗證集: {val_size:>10,} samples")

    # ------ 4. DataLoader ------
    # Windows 上 num_workers > 0 需要 if __name__ == "__main__" 保護
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,      # Windows 安全起見用 0
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,  # 驗證時可用更大 batch
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # ------ 5. 模型 / 損失 / 優化器 ------
    model = PuzzleSolverCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 模型參數量: {total_params:,}")

# ------ 6. 訓練迴圈 ------
    print(f"\n{'='*70}")
    print(f"🚀 開始訓練 | Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    if args.max_time_hours > 0:
        print(f"⏱️  設定時間上限: {args.max_time_hours} 小時")
    print(f"{'='*70}")

    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0
    
    global_start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # 訓練
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # 驗證
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # 學習率更新
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        epoch_time = time.time() - epoch_start
        total_elapsed = time.time() - global_start_time

        # 印出結果
        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            # 存檔最佳模型
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, args.model_path)
            marker = " ⭐ BEST"
        else:
            patience_counter += 1

        print(f"  Epoch {epoch:>3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc*100:.2f}% | "
              f"LR: {current_lr:.2e} | "
              f"{epoch_time:.1f}s{marker}")

        # Early Stopping
        if patience_counter >= args.patience:
            print(f"\n⏹️  Early Stopping! 驗證 Loss 已有 {args.patience} 個 epoch 沒有改善。")
            break
            
        # 檢查時間上限
        if args.max_time_hours > 0 and (total_elapsed / 3600.0) >= args.max_time_hours:
            print(f"\n⏱️  已達設定的時間上限 ({args.max_time_hours} 小時)，提前結束訓練！")
            break

    print(f"\n{'='*70}")
    print(f"✅ 訓練完成！最佳模型在 Epoch {best_epoch} (Val Loss: {best_val_loss:.4f})")
    print(f"   模型存檔: {args.model_path}")


# =========================================================
# 入口點
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="華容道模仿學習 — 訓練腳本")
    parser.add_argument("--data_path",    type=str,   default="dataset.pt",     help="資料集路徑")
    parser.add_argument("--model_path",   type=str,   default="best_model.pth", help="最佳模型存檔路徑")
    parser.add_argument("--epochs",       type=int,   default=1000,               help="最大 Epoch 數")
    parser.add_argument("--batch_size",   type=int,   default=1024,             help="Batch Size")
    parser.add_argument("--lr",           type=float, default=3e-4,             help="學習率")
    parser.add_argument("--weight_decay", type=float, default=1e-4,             help="權重衰減")
    parser.add_argument("--patience",     type=int,   default=5,                help="Early Stopping Patience")
    parser.add_argument("--max_time_hours", type=float, default=6,              help="最大訓練時間 (小時)，0代表無限制")
    args = parser.parse_args()

    main(args)
