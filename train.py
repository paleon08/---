import numpy as np
from env.marine_env import MarineEnv
# 향후 model/DDPG.py에 네트워크들을 하나로 묶는 DDPGAgent 클래스를 만들고 임포트해야 해!
# from model.DDPG import DDPGAgent 

def main():
    print("🌊 자율주행 선박 강화학습 훈련을 시작합니다!")
    
    env = MarineEnv()
    
    # State 6차원, Action 1차원(타각 제어)
    # agent = DDPGAgent(state_dim=6, action_dim=1) 
    
    episodes = 500
    
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0
        
        while not done:
            # 1. AI가 현재 상태를 보고 행동(타각) 결정
            # action = agent.get_action(state)
            action = np.array([0.0]) # 일단 직진만 하는 더미 액션(테스트용)
            
            # 2. 시뮬레이터 환경에 행동 투입 및 결과 확인
            next_state, reward, done, _ = env.step(action)
            
            # 3. AI 학습 (경험 메모리 저장 및 가중치 업데이트)
            # agent.train(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            step += 1
            
        print(f"[Episode {ep+1}/{episodes}] Total Reward: {total_reward:.2f} | Steps: {step}")

if __name__ == "__main__":
    main()