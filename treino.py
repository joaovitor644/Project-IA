import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
import ale_py
import torch

# Verificar se GPU está disponível
device = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE =", device)

gym.register_envs(ale_py)

# Criar o ambiente Atari
env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
env = AtariWrapper(env)

model = DQN("CnnPolicy", env, verbose=1, buffer_size=100000, learning_rate=1e-4,
            batch_size=64, device=device)

# Treinar o modelo
model.learn(total_timesteps=2000000)

# Salvar o modelo
model.save("dqn_pong_2M")
