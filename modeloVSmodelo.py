import ale_py.env
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
import torch
import ale_py
import copy
import random


gym.register_envs(ale_py)

# Configuração
N_COMPARACOES = 100  # Número de partidas que cada agente ira jogar.
PONTUACAO_MAX = 7  # Define o tamanho dos episodios.

# Criar ambiente
def create_env(render_mode=None):
    env = gym.make("ALE/Pong-v5", render_mode=render_mode)
    return AtariWrapper(env)

# Carregar os modelos treinados
# Use nomes diferentes para os modelos a serem comparados.
dqn_agent_puro = DQN.load("data/dqn_pong_2M")          # Agente DQN puro
dqn_agent_mcts = DQN.load("data/dqn_pong_1M")       # Agente DQN + MCTS

device = "cuda" if torch.cuda.is_available() else "cpu"

# Função para pegar ação do DQN (retorna ação)
def get_dqn_action(model, obs):
    # Garante que a observação esteja no formato [batch_size, channels, height, width]
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).permute(0, 3, 1, 2)
    with torch.no_grad():
        q_values = model.policy.q_net(obs_tensor)
    return int(torch.argmax(q_values).item())

# Função para retornar os Q-values (retorna um array de Q-values)
def get_q_values(model, obs):
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).permute(0, 3, 1, 2)
    with torch.no_grad():
        q_values = model.policy.q_net(obs_tensor)
    return q_values.cpu().numpy().flatten()

# Implementação simples do MCTS
class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = copy.deepcopy(state)
        self.parent = parent
        self.action = action
        self.children: list[MCTSNode] = []
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self, action_space):
        return len(self.children) == action_space

    def best_child(self, c_param=1.4):
        choices_weights = [
            (child.value / (child.visits + 1e-4)) + c_param * np.sqrt(np.log(self.visits + 1) / (child.visits + 1e-4))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

def mcts(env: AtariWrapper, agent, simulations=1):
    temp_env = create_env()
    action_space = env.action_space.n
    state = copy.deepcopy(env.unwrapped.clone_state())
    root = MCTSNode(state=state)

    for _ in range(simulations):
        node = root
        state = copy.deepcopy(root.state)

        # Seleção
        while node.is_fully_expanded(temp_env.action_space.n):
            node = node.best_child()

        # Expansão
        if not node.is_fully_expanded(temp_env.action_space.n):
            action = random.choice([a for a in range(temp_env.action_space.n) if a not in [child.action for child in node.children]])
            temp_env.reset(seed=None, options={"state": state})
            obs, reward, done, _, _ = temp_env.step(action)
            new_state = obs
            child_node = MCTSNode(new_state, node, action)
            node.children.append(child_node)
            node = child_node

        # Simulaçao
        total_reward = 0
        rollout_length = 10
        for _ in range(rollout_length):
            q_values = get_q_values(agent, obs)
            best_q = np.max(q_values)
            action = q_values.argmax().item()
            # action = agent.select_action(new_state)
            # obs, reward, done, _, _ = temp_env.step(action)
            total_reward += best_q
            if done:
                break
        
        # Backpropagation
        while node:
            node.visits += 1
            node.value += total_reward
            node = node.parent

    return root.best_child(c_param=0)


# Função para simular partidas entre os agentes e a maquina. um agente por episodio
def play_agents():
    wins_mcts = 0
    loss_mcts = 0
    score_total_mcts = 0
    wins_puro = 0
    loss_puro = 0
    score_total_puro = 0

    env = create_env(render_mode=None) # pode setar o render_mode para "human" se quiser que renderize uma janela mostrando o jogo.
    epNow = 0 # 0 = sem mcts, 1 = com mcts

    for episode in range(N_COMPARACOES * 2):
        print(f"EPISODIO {episode+1} {"(SEM MCTS)" if epNow == 0 else "(COM MCTS)"}")
        obs, _ = env.reset()
        done = False
        step_count = 0
        score_mcts = 0
        score_puro = 0
        score_adversario = 0

        # joga com qualquer um dos agentes até algum dos jogadores ganhar a partida
        while not done:
            if epNow != 0: 
                action = mcts(env, dqn_agent_mcts).action 
                acting_agent = "MCTS"
            else:
                action = get_dqn_action(dqn_agent_puro, obs)
                acting_agent = "Puro"
            obs, reward, done, truncated, _ = env.step(action)
            
            if(env.render_mode == "human"):
                env.render()

            # Interpretação simplificada: se o agente que agiu obteve recompensa positiva, ele marca ponto, caso contrario, o adversario marca ponto.
            if reward > 0:
                if acting_agent == "MCTS":
                    score_mcts += 1
                else:
                    score_puro += 1
            elif reward < 0:
                score_adversario += 1   

            step_count += 1

            if score_mcts >= PONTUACAO_MAX or score_puro >= PONTUACAO_MAX or score_adversario >= PONTUACAO_MAX:
                done = True

        if score_mcts >= PONTUACAO_MAX:
            wins_mcts += 1
        elif score_puro >= PONTUACAO_MAX:
            wins_puro += 1
        elif epNow == 1 and score_adversario > PONTUACAO_MAX:
            loss_mcts += 1
        elif epNow == 0 and score_adversario > PONTUACAO_MAX:
            loss_puro += 1

        score_total_mcts += score_mcts
        score_total_puro += score_puro

        print(f"Episódio {episode+1}: Score {'MCTS' if epNow == 1 else 'Puro'} = {score_mcts if epNow == 1 else score_puro}")
        env.close()

        if epNow == 1:
            epNow = 0
        else:
            epNow = 1

    win_rate_mcts = wins_mcts / (N_COMPARACOES)
    win_rate_puro = wins_puro / (N_COMPARACOES)

    print(f"\nDQN (MCTS) -> Taxa de Vitórias: {win_rate_mcts * 100:.2f}%")
    print(f"DQN Puro    -> Taxa de Vitórias: {win_rate_puro * 100:.2f}%")

    print(f"\nDQN (MCTS) -> Pontuação total: {score_total_mcts}")
    print(f"DQN Puro    -> Pontuação total: {score_total_puro}")

# Rodar a avaliação
play_agents()
