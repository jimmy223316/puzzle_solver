# puzzle_solver (NeuralSlider) 🧩

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 專案簡介 / Project Overview

**puzzle_solver** 是一個基於深度強化學習（Deep Reinforcement Learning）的高性能數字華容道（N-Puzzle）自動求解器。本專案結合了**行為複製（Behavioral Cloning）**與 **PPO 演算法**，旨在突破傳統演算法的限制，尋找 3x3 到 12x12 盤面的極限最短路徑。

**puzzle_solver** is a high-performance N-Puzzle (Sliding Puzzle) automated solver powered by Deep Reinforcement Learning. By combining **Behavioral Cloning (BC)** with the **Proximal Policy Optimization (PPO)** algorithm, this project aims to surpass traditional algorithmic constraints and discover optimal paths for puzzle sizes ranging from 3x3 up to 12x12.

---

## 核心特性 / Key Features

- 🧠 **混合學習架構 (Hybrid Learning)**: 利用 BC 模型繼承專家經驗進行冷啟動，再透過 PPO 強化學習進行步數壓縮與路徑優化。
- 🧪 **穩定微調技術 (Stable Fine-tuning)**: 實作了 **Entropy Floor 保護**與 **Critic Warm-up** 機制，有效防止強化學習過程中的策略崩塌。
- 🌐 **網頁自動化 (Web Automation)**: 整合 Selenium 技術，讓 AI 能夠即時接管網頁版數字華容道遊戲並進行解題演示。
- 📈 **課程學習 (Curriculum Learning)**: 支援從簡單到困難（3x3 到 12x12）的漸進式訓練流程。
- 🛠️ **自定義環境 (Custom Env)**: 基於 OpenAI Gym 規格開發的專屬拼圖環境，支援曼哈頓距離追蹤與密集獎勵設計。

---

## 快速開始 / Quick Start

### 1. 安裝環境 / Setup
```bash
pip install torch numpy selenium tqdm opencv-python
```

### 2. 資料生成與模仿學習 / Data & Imitation Learning
```bash
# 生成專家經驗資料集 (dataset.pt)
python generate_data.py --games 1000 --sizes 3 4 5

# 進行行為複製 (BC) 訓練
python train.py --data_path dataset.pt --epochs 100
```

### 3. 強化學習微調 / RL Fine-tuning
```bash
# 讀取 BC 權重並針對 3x3 盤面進行 PPO 微調
python train_rl.py --bc_model best_model.pth --focus_size 3 --total_updates 3000
```

### 3. 展示與執行 / Execution
```bash
# 使用訓練好的 RL 模型在網頁上執行展示
python play_web_rl.py --size 3 --model best_rl_model_3x3.pth
```

### 留名青史
```python
# 過關後自動登錄排行榜
env.get_state(name=f"Psyduck_{n}x{n}")
```

---

## 技術細節 / Technical Details

- **Backbone**: Deep Residual Convolutional Neural Network (CNN)
- **RL Algorithm**: PPO with GAE (Generalized Advantage Estimation)
- **Exploration**: Adaptive Entropy Bonus with dynamic floor protection
- **Automation**: Selenium with real-time screen parsing

---

## 作者 / Author
**Jimmy** - [GitHub Profile](https://github.com/jimmy223316)
