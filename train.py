import numpy as np
import time
from collections import deque
from env.marine_env import MarineEnv
from model.DDPG import DDPGAgent

def train():
    env = MarineEnv()
    agent = DDPGAgent(state_dim=6, action_dim=1)

    # -------------------------------------------------------------
    # 🎯 1. 학습 목표 및 종료 조건 설정
    # -------------------------------------------------------------
    MAX_EPISODES = 500               # 최대 수행 에피소드 수
    TARGET_AVG_REWARD = 180.0        # 수렴 판단 목표 평균 보상 (환경 설정에 따라 조정)
    TARGET_TRACK_ERROR = 1.0         # 목표 평균 항로 오차 (m)
    CONSECUTIVE_CHECK_EPISODES = 30  # 이동 평균을 계산할 에피소드 구간
    
    reward_window = deque(maxlen=CONSECUTIVE_CHECK_EPISODES)
    track_error_window = deque(maxlen=CONSECUTIVE_CHECK_EPISODES)
    best_avg_reward = -float('inf')

    print("🚀 [System] 자율주행배 DDPG 강화학습을 시작합니다.")
    print(f"🎯 [Goal] 최근 {CONSECUTIVE_CHECK_EPISODES}회 평균 오차 <= {TARGET_TRACK_ERROR}m 달성 시 자동 종료\n")

    for episode in range(1, MAX_EPISODES + 1):
        state = env.reset()
        episode_reward = 0.0
        episode_track_errors = []
        done = False

        while not done:
            action = agent.get_action(state, explore=True)
            next_state, reward, done, info = env.step(action)

            # 경험 저장 및 학습
            agent.replay_buffer.add(state, action, reward, next_state, done)
            agent.train()

            # 오차 및 보상 누적
            episode_track_errors.append(abs(state[0]))  # state[0] = e_track
            episode_reward += reward
            state = next_state

        # 에피소드 결과 기록
        avg_ep_track_error = np.mean(episode_track_errors)
        reward_window.append(episode_reward)
        track_error_window.append(avg_ep_track_error)

        moving_avg_reward = np.mean(reward_window)
        moving_avg_track_error = np.mean(track_error_window)

        print(f"Episode {episode:3d} | Reward: {episode_reward:7.2f} | "
              f"e_track: {avg_ep_track_error:5.2f}m | "
              f"Recent {len(reward_window)}Ep Avg Reward: {moving_avg_reward:7.2f} | "
              f"Avg e_track: {moving_avg_track_error:5.2f}m")

        # -------------------------------------------------------------
        # 💾 2. 최고 성능 가중치 자동 저장 (Best Checkpoint)
        # -------------------------------------------------------------
        if moving_avg_reward > best_avg_reward and episode >= CONSECUTIVE_CHECK_EPISODES:
            best_avg_reward = moving_avg_reward
            agent.save_models(save_dir="checkpoints/best")
            print(f"  👉 [Best Model] 최고 성능 경신! 가중치 저장 완료 (Avg Reward: {best_avg_reward:.2f})")

        # -------------------------------------------------------------
        # 🛑 3. 수렴 조건 만족 시 조기 종료 (Early Stopping)
        # -------------------------------------------------------------
        if (len(reward_window) == CONSECUTIVE_CHECK_EPISODES and 
            moving_avg_track_error <= TARGET_TRACK_ERROR and 
            moving_avg_reward >= TARGET_AVG_REWARD):
            
            print("\n" + "=" * 60)
            print(f"🎉 [System] 학습 수렴 조건 달성! (Episode {episode})")
            print(f"   - 최근 {CONSECUTIVE_CHECK_EPISODES}회 평균 항로 오차: {moving_avg_track_error:.2f}m")
            print(f"   - 최근 {CONSECUTIVE_CHECK_EPISODES}회 평균 보상: {moving_avg_reward:.2f}")
            print("=" * 60)
            agent.save_models(save_dir="checkpoints/final")
            break

    # 최대 에피소드 도달 시 최종 저장
    agent.save_models(save_dir="checkpoints/latest")

if __name__ == "__main__":
    train()