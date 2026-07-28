import time
from env.marine_env import MarineEnv
from model.DDPG import DDPGAgent

def train():
    env = MarineEnv()
    agent = DDPGAgent(state_dim=6, action_dim=1)

    # 1. 기존 학습된 모델 가중치가 있으면 불러오기 (연속 업데이트 모드)
    resume_training = True
    if resume_training:
        agent.load_models(save_dir="checkpoints")

    max_episodes = 1000
    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # 실시간 추론 연산 지연(ms) 측정
            start_time = time.time()
            action = agent.get_action(state)
            inference_time_ms = (time.time() - start_time) * 1000

            # 환경 진행
            next_state, reward, done, info = env.step(action)
            
            # 경험 저장 및 신경망 가중치 업데이트
            agent.replay_buffer.add(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            episode_reward += reward

        print(f"Episode: {episode+1} | Reward: {episode_reward:.2f} | Latency: {inference_time_ms:.2f}ms")

        # 2. 100 에피소드마다 주기적 모델 가중치 저장
        if (episode + 1) % 100 == 0:
            agent.save_models(save_dir="checkpoints")

    # 훈련 최종 종료 후 저장
    agent.save_models(save_dir="checkpoints")

if __name__ == "__main__":
    train()