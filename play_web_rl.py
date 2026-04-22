import time
import torch
import argparse
from collections import deque

# 引入你的環境與工具
from manual_solver import WebPuzzleEnv  
from generate_data import encode_state
from env_rl import ACTION_DELTAS

# 引入全新的 RL 大腦
from model_rl import ActorCriticCNN

def play_web_with_rl_ai(n: int, model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  使用裝置: {device}")
    
    print(f"📂 載入強化學習大腦: {model_path}...")
    model = ActorCriticCNN().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"✅ 大腦載入成功！(來自 RL Update {checkpoint.get('update', '?')})")

    print(f"\n🌐 正在開啟 {n}x{n} 網頁華容道...")
    env = WebPuzzleEnv(n=n)
    time.sleep(1)
    
    state = env.get_state()
    goal_state = list(range(1, n*n)) + [0]
    
    step_count = 0
    max_steps = n * n * 50 
    
    print("\n🚀 RL 超級 AI 接管控制權！開始展示極限速通...")
    
    recent_states = deque(maxlen=20) 
    recent_states.append(tuple(state))

    while state != goal_state and step_count < max_steps:
        # 1. 將盤面轉為 Tensor
        encoded_state = encode_state(state, n).unsqueeze(0).to(device)
        
        # 2. 前向傳播 (相容 Actor-Critic 架構)
        with torch.no_grad():
            output = model(encoded_state)
            # PPO 模型通常會回傳 (logits, value)，我們只需要 logits 來決定動作
            if isinstance(output, tuple):
                logits = output[0].squeeze(0)
            else:
                logits = output.squeeze(0)
            
        # 3. 合法動作過濾與防鬼打牆
        zero_idx = state.index(0)
        r, c = zero_idx // n, zero_idx % n
        legal_mask = torch.full((4,), float("-inf"), device=device)
        
        for action, (dr, dc) in ACTION_DELTAS.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n:
                target_idx = nr * n + nc
                simulated_state = list(state)
                simulated_state[zero_idx], simulated_state[target_idx] = simulated_state[target_idx], simulated_state[zero_idx]
                
                if tuple(simulated_state) in recent_states:
                    legal_mask[action] = -100.0 
                else:
                    legal_mask[action] = 0.0
                
        # 4. 取出最佳動作
        masked_logits = logits + legal_mask
        best_action = masked_logits.argmax().item()
        
        dr, dc = ACTION_DELTAS[best_action]
        target_r, target_c = r + dr, c + dc
        target_idx = target_r * n + target_c
        number_to_click = state[target_idx]
        
        # 5. 點擊與更新狀態
        env.click_tile(number_to_click)
        state[zero_idx], state[target_idx] = state[target_idx], state[zero_idx]
        recent_states.append(tuple(state))
        
        step_count += 1
        
        if step_count % 500 == 0:
            print(f"  🤖 目前已走 {step_count} 步，重新校準網頁狀態...")
            time.sleep(0.001)
            state = env.get_state()

    if state == goal_state:
        print(f"\n🎉 震撼！RL AI 成功過關！總共花費了 {step_count} 步！")
        time.sleep(0.5)
        # 留名青史
        env.get_state(name=f"Psyduck_{n}x{n}")
    else:
        print(f"\n❌ 耗盡體力 (超過 {max_steps} 步)。")
        
    input("按下 Enter 關閉瀏覽器...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL AI 網頁展示")
    parser.add_argument("--size", type=int, default=4, help="盤面尺寸")
    parser.add_argument("--model", type=str, default="best_rl_model_30s.pth", help="RL 模型路徑")
    args = parser.parse_args()
    
    play_web_with_rl_ai(n=args.size, model_path=args.model)