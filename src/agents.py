import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from src.game import Game


class Agent:
    def __init__(self):
        self.history = []
        self.reward_history = []

    def __get_params__(self):
        return self.history, self.reward_history

    def choose_action(self):
        pass

    def update(self, action):
        pass

    def reset(self):
        self.history = []
        self.reward_history = []


class RandomAgent(Agent):
    def choose_action(self):
        return random.choice([0, 1])


class AlwaysCooperateAgent(Agent):
    def choose_action(self):
        return 0  # Always cooperates


class AlwaysDefectAgent(Agent):
    def choose_action(self):
        return 1  # Always defects


class TitForTatAgent(Agent):
    def __init__(self):
        super().__init__()
        self.last_opponent_action = 0  # Start by cooperating

    def choose_action(self):
        return self.last_opponent_action  # Copy the opponent's last action

    def update(self, opponent_action):
        self.last_opponent_action = opponent_action  # Store opponent's action

    def reset(self):
        self.last_opponent_action = 0
        self.history = []
        self.reward_history = []


class SpitefulAgent(Agent):
    def __init__(self):
        super().__init__()
        self.opponent_defected = False  # Start by cooperating

    def choose_action(self):
        if self.opponent_defected:
            return 1  # Defects if opponent defected once
        return 0  # Cooperates

    def update(self, opponent_action):
        if opponent_action == 1:  # Last opponent defected
            self.opponent_defected = True

    def reset(self):
        self.opponent_defected = False
        self.history = []
        self.reward_history = []


class PavlovAgent(Agent):
    def __init__(self):
        super().__init__()
        self.last_opponent_action = 0

    def choose_action(self):
        if self.history:
            return int(not self.history[-1] == self.last_opponent_action)
        return 0

    def update(self, opponent_action):
        self.last_opponent_action = opponent_action


class TitForTwoTatsAgent(Agent):
    def __init__(self):
        super().__init__()
        self.last_opponent_actions = [0, 0]

    def choose_action(self):
        if self.last_opponent_actions == [1, 1]:  # Last opponent defects twice
            return 1
        return 0

    def update(self, opponent_action):
        self.last_opponent_actions.append(opponent_action)
        self.last_opponent_actions.pop(0)

    def reset(self):
        self.last_opponent_actions = [0, 0]
        self.history = []
        self.reward_history = []


class TwoTitsForTatAgent(Agent):
    def __init__(self):
        super().__init__()
        self.last_opponent_actions = [0, 0]

    def choose_action(self):
        if 1 in self.last_opponent_actions:
            return 1
        return 0

    def update(self, opponent_action):
        self.last_opponent_actions.append(opponent_action)
        self.last_opponent_actions.pop(0)

    def reset(self):
        self.last_opponent_actions = [0, 0]
        self.history = []
        self.reward_history = []


class ProvocativeAgent(Agent):
    def choose_action(self):
        try:
            if self.history[-1] == 0 and self.history[-2] == 0:
                return 1
        except:
            return 0
        return 0


class TitForTatOppositeAgent(Agent):
    def __init__(self):
        super().__init__()
        self.last_opponent_action = 0  # Start by cooperating

    def choose_action(self):
        return int(not self.last_opponent_action)  # Copy the opponent's last action

    def update(self, opponent_action):
        self.last_opponent_action = opponent_action  # Store opponent's action

    def reset(self):
        self.last_opponent_action = 0
        self.history = []
        self.reward_history = []


class AdaptiveAgent(Agent):
    def __init__(self, memory_size=10):
        super().__init__()
        self.opponent_actions = deque(maxlen=memory_size)

    def choose_action(self):
        if not self.opponent_actions:  # Default to cooperation if no history
            return 0
        avg_action = sum(self.opponent_actions) / len(self.opponent_actions)
        return 1 if avg_action > 0.5 else 0  # Defect if opponent defects more than 50%

    def update(self, opponent_action):
        self.opponent_actions.append(opponent_action)

    def reset(self):
        self.opponent_actions.clear()
        self.history = []
        self.reward_history = []


class GenerousTitForTatAgent(Agent):
    def __init__(self, forgiveness_prob=0.1):
        super().__init__()
        self.forgiveness_prob = forgiveness_prob
        self.last_opponent_action = 0

    def choose_action(self):
        if self.last_opponent_action == 1 and random.random() < self.forgiveness_prob:
            return 0  # Forgive with a certain probability
        return self.last_opponent_action

    def update(self, opponent_action):
        self.last_opponent_action = opponent_action

    def reset(self):
        self.last_opponent_action = 0
        self.history = []
        self.reward_history = []


class WinStayLoseShiftAgent(Agent):
    def choose_action(self):
        if self.reward_history:  # Default to cooperation at the start
            if self.reward_history[-1] > 0:  # Stay if last reward was positive
                return self.history[-1]
            else:  # Shift otherwise
                return 1 - self.history[-1]
        return 0


class SuspiciousTitForTatAgent(Agent):
    def __init__(self):
        super().__init__()
        self.last_opponent_action = 1  # Start by defecting

    def choose_action(self):
        return self.last_opponent_action

    def update(self, opponent_action):
        self.last_opponent_action = opponent_action

    def reset(self):
        self.last_opponent_action = 1
        self.history = []
        self.reward_history = []


class GradualAgent(Agent):
    def __init__(self):
        super().__init__()
        self.retaliation_count = 0
        self.forgiveness = False

    def choose_action(self):
        if self.retaliation_count > 0:
            self.retaliation_count -= 1
            return 1  # Retaliate
        elif self.forgiveness:
            self.forgiveness = False
            return 0  # Forgive
        return 0  # Cooperate by default

    def update(self, opponent_action):
        if opponent_action == 1:
            self.retaliation_count += (
                1  # Add one round of retaliation for each defection
            )
            self.forgiveness = True  # Forgive after retaliating

    def reset(self):
        self.retaliation_count = 0
        self.forgiveness = False
        self.history = []
        self.reward_history = []


class SoftMajorityAgent(Agent):
    def __init__(self):
        super().__init__()
        self.opponent_cooperation = 0
        self.opponent_defection = 0

    def choose_action(self):
        return 0 if self.opponent_cooperation >= self.opponent_defection else 1

    def update(self, opponent_action):
        if opponent_action == 0:
            self.opponent_cooperation += 1
        else:
            self.opponent_defection += 1

    def reset(self):
        self.opponent_cooperation = 0
        self.opponent_defection = 0
        self.history = []
        self.reward_history = []


class QLearningAgent(Agent):
    def __init__(self, q_table=None, alpha=0.1, gamma=0.5, epsilon=0.1):
        super().__init__()
        if q_table is None:
            self.q_table = np.zeros(
                (2, 2)
            )  # Q-values for (Agent_Action, Opponent_Action)
        else:
            self.q_table = np.array(q_table)  # q_table should look like [[], []]
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.last_opponent_action = None

    def choose_action(self):
        # If no previous action by opponent, randomly choose an action
        if self.last_opponent_action is None or random.uniform(0, 1) < self.epsilon:
            action = random.choice([0, 1])  # Explore: random action
        else:
            # Exploit: Choose action with the highest Q-value given opponent's last action
            action = np.argmax(self.q_table[self.last_opponent_action])
        return action

    def update_q_values(self, opponent_action, reward):
        # Update Q-value using the Q-learning formula
        best_next_action = np.max(
            self.q_table[opponent_action]
        )  # Best Q-value for opponent's action
        self.q_table[self.history[-1], opponent_action] += self.alpha * (
            reward
            + self.gamma * best_next_action
            - self.q_table[self.history[-1], opponent_action]
        )

    def reset(self):
        self.last_action = None
        self.history = []
        self.reward_history = []


class DeepQNetwork(nn.Module):
    def __init__(self, input_size):
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DeepQLearningAgent(Agent):
    def __init__(
        self,
        state_size=10,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        learning_rate=0.001,
        load_model=True,
    ):
        super().__init__()
        self.state_size = state_size
        self.memory = deque(maxlen=2000)  # Replay memory
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min  # Minimum exploration rate
        self.epsilon_decay = epsilon_decay  # Exploration decay rate

        # Neural network and optimizer
        self.model = DeepQNetwork(state_size)
        if torch.cuda.is_available:
            self.model.to("cuda")
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()

        self.temperature = 1.0
        if load_model:
            self.model.load_state_dict(torch.load("deep_q_agent.pt"))
            self.model.eval()
            self.temperature = 0.0
        

        # Initialize with default state
        self.prev_actions = [0] * state_size

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, temperature=1.0):
        state = torch.FloatTensor(state)

        with torch.no_grad():
            q_values = self.model(state)  # Predicted Q-values for actions
        if temperature > 0.0:
            # Boltzmann sampling for exploration
            probabilities = torch.softmax(q_values / temperature, dim=0)  # Compute probabilities
            action = torch.multinomial(probabilities, 1).item()  # Sample action
        else:
            # Greedy selection for exploitation
            action = torch.argmax(q_values).item()

        return action

        

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state)  # Add batch dimension
            next_state = torch.FloatTensor(next_state)  # Add batch dimension
            reward = torch.FloatTensor([reward])
            # print(state.shape, next_state.shape)
            target = reward
            if not done:
                target += self.gamma * torch.max(self.model(next_state).detach())

            # print(self.model(state).shape, self.model(state).squeeze(0).shape, self.model(state).squeeze(0)[action].unsqueeze(0), target.shape)
            # print(action)
            current_q = self.model(state).squeeze(0)[action].unsqueeze(0)
            # current_q = self.model(state)[action]

            # print(current_q, target)
            loss = self.loss_fn(current_q, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.temperature = max(0.1, self.temperature * 0.995) 
        return loss

    def update(self, opponent_action):
        if len(self.history) >= 2:
            self.prev_actions = self.prev_actions[2:] + [
                self.history[-1],
                opponent_action,
            ]
        else:
            self.prev_actions = self.prev_actions[2:] + [
                self.prev_actions[-2],
                opponent_action,
            ]

    def choose_action(self):
        state = np.array(self.prev_actions)
        return self.act(state, temperature=self.temperature)


ALL_AGENTS = [
    RandomAgent(),
    QLearningAgent(),
    TitForTatAgent(),
    AlwaysCooperateAgent(),
    AlwaysDefectAgent(),
    PavlovAgent(),
    TitForTatOppositeAgent(),
    TwoTitsForTatAgent(),    
    DeepQLearningAgent(load_model=False),
    SpitefulAgent(),
    ProvocativeAgent(),
    TitForTwoTatsAgent(),
    GradualAgent(),
    SuspiciousTitForTatAgent(),
    WinStayLoseShiftAgent(),
    GenerousTitForTatAgent(),
    AdaptiveAgent(),
    SoftMajorityAgent(),
]

BASIC_AGENTS = [agent for agent in ALL_AGENTS if "QLearning" not in agent.__class__.__name__]

if __name__ == "__main__":
    # q_agent = DeepQLearningAgent()
    # game = Game(SpitefulAgent(), AlwaysCooperateAgent())
    # a, b = game.play_iterated_game(100)
    # print(a, b)
    print(len(ALL_AGENTS))
    print(len(BASIC_AGENTS))
    

    # print(q_agent.train_multiple_agents())
