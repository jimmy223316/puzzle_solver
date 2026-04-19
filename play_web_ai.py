import time
import torch
import argparse

# 引入你已經寫好的各個模組
from manual_solver import WebPuzzleEnv  
from model import PuzzleSolverCNN
from generate_data import encode_state
from env import ACTION_DELTAS



def play_web_with_ai(n: int, model_path: str):
    # 1. 初始化硬體與載入模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  使用裝置: {device}")
    
    print(f"📂 載入神經網路大腦: {model_path}...")
    model = PuzzleSolverCNN().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"✅ 大腦載入成功！(來自 Epoch {checkpoint.get('epoch', '?')})")

    # 2. 開啟瀏覽器進入遊戲
    print(f"\n🌐 正在開啟 {n}x{n} 網頁華容道...")
    env = WebPuzzleEnv(n=n)
    time.sleep(1) # 等待開局動畫
    
    # 抓取初始盤面
    state = env.get_state()
    goal_state = list(range(1, n*n)) + [0]
    
    step_count = 0
    max_steps = n * n * 50 # 設定防呆上限，避免 AI 陷入死迴圈
    
    print("\n🚀 AI 接管控制權！開始自動遊玩...")
    

    from collections import deque
    recent_states = deque(maxlen=20) # 記住最近 20 步的盤面
    recent_states.append(tuple(state))

    # 3. AI 遊玩主迴圈
    while state != goal_state and step_count < max_steps:
        
        # --- A. AI 觀察盤面並思考 ---
        # 1. 將 1D 陣列編碼為神經網路看得懂的 (3, 12, 12) 張量
        encoded_state = encode_state(state, n).unsqueeze(0).to(device)
        
        # 2. 神經網路前向傳播，給出 4 個方向的機率分佈 (Logits)
        with torch.no_grad():
            logits = model(encoded_state).squeeze(0)
            
        # 3. 過濾掉撞牆的非法動作
        zero_idx = state.index(0)
        r, c = zero_idx // n, zero_idx % n
        legal_mask = torch.full((4,), float("-inf"), device=device)
        
        for action, (dr, dc) in ACTION_DELTAS.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n:
                target_idx = nr * n + nc

                simulated_state = list(state)
                simulated_state[zero_idx], simulated_state[target_idx] = simulated_state[target_idx], simulated_state[zero_idx]
                
                # 🚧 【防鬼打牆核心邏輯】 🚧
                # 檢查這個預測的盤面，是不是在最近 20 步內出現過？
                if tuple(simulated_state) in recent_states:
                    # 如果出現過，給予極大的懲罰，迫使 AI 放棄這個動作 (除非無路可走)
                    legal_mask[action] = -100.0 
                else:
                    # 如果是全新盤面，正常開放
                    legal_mask[action] = 0.0
                
        # 4. 做出最終決定
        masked_logits = logits + legal_mask
        best_action = masked_logits.argmax().item()
        
        # --- B. 將 AI 的想法化為實際點擊 ---
        # 計算要點擊哪個數字 (找出目標座標)
        dr, dc = ACTION_DELTAS[best_action]
        target_r, target_c = r + dr, c + dc
        target_idx = target_r * n + target_c
        
        number_to_click = state[target_idx]
        
        # 執行點擊
        env.click_tile(number_to_click)
        
        # --- C. 內部狀態極速更新 ---
        # 為了保持極速，我們直接在 Python 內部交換陣列數字，不依賴網頁爬取
        state[zero_idx], state[target_idx] = state[target_idx], state[zero_idx]
        
        step_count += 1
        
        # 防錯機制：每走 50 步，重新從網頁爬取一次真實盤面，防止 Python 內部陣列與網頁動畫脫鉤
        if step_count % 500 == 0:
            print(f"  🤖 目前已走 {step_count} 步，重新校準網頁狀態...")
            time.sleep(0.001) # 等待動畫完畢再抓取
            state = env.get_state()

    # 4. 結算
    if state == goal_state:
        print(f"\n🎉 震撼！AI 成功過關！總共花費了 {step_count} 步！")
        time.sleep(0.5)
        # 觸發破紀錄輸入框
        env.get_state(name=f"可達鴨_{n}x{n}")
    else:
        print(f"\n❌ AI 耗盡了體力 (超過 {max_steps} 步)。")
        print("可能原因：遇到了沒見過的複雜盤面，導致誤差累積陷入死迴圈。")
        
    input("按下 Enter 關閉瀏覽器...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="讓 AI 直接在網頁上玩華容道")
    parser.add_argument("--size", type=int, default=9, help="盤面尺寸 (預設 4x4)")
    parser.add_argument("--model", type=str, default="best_model.pth", help="訓練好的模型路徑")
    args = parser.parse_args()
    
    play_web_with_ai(n=args.size, model_path=args.model)