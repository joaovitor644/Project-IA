import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
import torch
import ale_py
import copy  # para fallback no clone do ambiente

# Configurações
N_EPISODES = 1000          # Número de partidas para avaliar
WINNING_SCORE = 7        # Pontos para encerrar o episódio
MCTS_ITERATIONS = 1     # Número de iterações para o MCTS

# Cria o ambiente sem renderização
def create_env():
    env = gym.make("ALE/Pong-v5", render_mode=None)  # Sem janela gráfica
    return AtariWrapper(env)

# Função para clonar o ambiente a partir do estado atual
def clone_env(env):
    """
    Tenta clonar o estado do ambiente.
    Presume que o ambiente suporta 'clone_full_state' e 'restore_full_state'.
    Caso contrário, tenta usar copy.deepcopy (pode não funcionar para todos os casos).
    """
    env_clone = create_env()
    try:
        state = env.unwrapped.clone_full_state()
        env_clone.unwrapped.restore_full_state(state)
    except AttributeError:
        env_clone = copy.deepcopy(env)
    return env_clone

# Carregar os modelos treinados
dqn_agent_puro = DQN.load("dqn_pong_2M")    # Agente DQN puro (2M passos)
dqn_agent_mcts = DQN.load("dqn_pong_1M")      # Agente DQN + MCTS (1M passos)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Função para pegar ação do DQN (retorna ação)
def get_dqn_action(model, obs):
    # Garante que a observação esteja no formato [batch_size, canais, altura, largura]
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
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.q_value = 0

    def expand(self, actions):
        for action in actions:
            if action not in self.children:
                self.children[action] = MCTSNode(state=None, parent=self)

    def select(self, c=1.4):
        # Seleciona a ação que maximiza o balanço entre exploração e exploração (fórmula UCB)
        return max(
            self.children.items(),
            key=lambda item: item[1].q_value / (1 + item[1].visits) + 
                           c * np.sqrt(np.log(self.visits + 1) / (1 + item[1].visits))
        )[0]

    def update(self, reward):
        self.visits += 1
        self.q_value += reward

# Função para rodar MCTS + DQN a partir do estado atual
def mcts(env, obs, model, iterations=MCTS_ITERATIONS):
    root = MCTSNode(state=obs)
    actions = list(range(env.action_space.n))
    # Expande o nó raiz para garantir que ele tenha filhos
    root.expand(actions)
    
    for _ in range(iterations):
        node = root
        # Clona o ambiente a partir do estado atual
        temp_env = clone_env(env)
        temp_obs = obs  # inicia com o estado atual
        done = False

        # Simula a trajetória a partir do estado atual
        while node.children and not done:
            action = node.select()
            temp_obs, reward, done, truncated, _ = temp_env.step(action)
            # Se a ação ainda não foi expandida a partir deste nó, expande-a
            if action not in node.children:
                node.expand(actions)
            node = node.children[action]

        # Avalia o estado terminal ou não terminal utilizando os Q-values
        if not done:
            q_values = get_q_values(model, temp_obs)
            best_q = np.max(q_values)
        else:
            best_q = 0

        # Retropropaga o valor obtido na simulação
        current = node
        while current is not None:
            current.update(best_q)
            current = current.parent

    # Seleciona a melhor ação a partir da raiz (sem fator de exploração)
    best_action = root.select(c=0)
    return best_action

# Função para simular uma partida entre os dois agentes alternando as ações
def play_agents():
    wins_mcts = 0
    wins_puro = 0
    placar = [0,0]

    for episode in range(N_EPISODES):
        print(f"EPISÓDIO {episode+1}")
        env = create_env()
        obs, _ = env.reset()
        done = False
        score_mcts = 0
        score_puro = 0
        step_count = 0

        # Alterna o controle entre os agentes a cada passo:
        # Se step_count for par, o agente MCTS (DQN + MCTS) age;
        # Se ímpar, o agente DQN puro age.
        while not done:
            if step_count % 2 == 0:
                action = mcts(env, obs, dqn_agent_mcts)
                acting_agent = "MCTS"
            else:
                action = get_dqn_action(dqn_agent_puro, obs)
                acting_agent = "Puro"
            obs, reward, done, truncated, _ = env.step(action)

            # Interpretação simplificada: se o agente que agiu obteve recompensa positiva, ele marca ponto;
            # se negativa, o outro agente marca.
            if reward > 0:
                if acting_agent == "MCTS":
                    score_mcts += 1
                else:
                    score_puro += 1
            elif reward < 0:
                if acting_agent == "MCTS":
                    score_puro += 1
                else:
                    score_mcts += 1

            # Encerra o episódio se algum agente atingir o placar definido
            if score_mcts >= WINNING_SCORE or score_puro >= WINNING_SCORE:
                done = True

            step_count += 1

        if score_mcts > score_puro:
            wins_mcts += 1
        else:
            wins_puro += 1

        print(f"Episódio {episode+1}: Score MCTS = {score_mcts}, Score Puro = {score_puro}")
        if score_mcts > score_puro:
            placar[0] += 1
        else:
            placar[1] += 1
        print(f"PLACAR: DQN_MCTS {placar[0]}x{placar[1]} DQN")

    win_rate_mcts = wins_mcts / N_EPISODES
    win_rate_puro = wins_puro / N_EPISODES

    print(f"\nDQN (MCTS) -> Taxa de Vitórias: {win_rate_mcts * 100:.2f}%")
    print(f"DQN Puro    -> Taxa de Vitórias: {win_rate_puro * 100:.2f}%")
    print(f"Placar final: DQN {int(win_rate_puro*N_EPISODES)} x {int(win_rate_mcts*N_EPISODES)} DQN_MCTS")

# Rodar a avaliação
play_agents()
