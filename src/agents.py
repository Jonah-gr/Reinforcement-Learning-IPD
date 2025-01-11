import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from src.game import Game


class Agent:
    def __init__(self):
        """
        Initialize a new Agent instance.

        Attributes
        ----------
        history : list
            History of actions taken.
        reward_history : list
            History of rewards received.
        """
        self.history = []
        self.reward_history = []

    def __get_params__(self):
        return self.history, self.reward_history

    def choose_action(self):
        pass

    def update(self, action):
        pass

    def reset(self):
        """
        Reset the agent's state for a new episode.

        This function clears the agent's action and reward histories,
        preparing the agent for a new iteration or episode.
        """
        self.history = []
        self.reward_history = []


class RandomAgent(Agent):
    def choose_action(self):
        """
        Choose a random action.

        Returns
        -------
        action : int
            Randomly chosen action, either 0 (cooperate) or 1 (defect).
        """
        return random.choice([0, 1])


class AlwaysCooperateAgent(Agent):
    def choose_action(self):
        """
        Choose an action based on the agent's policy.

        For AlwaysCooperateAgent, this always returns 0, meaning the agent
        will always cooperate.

        Returns
        -------
        action : int
            The chosen action, either 0 (cooperate) or 1 (defect).
        """
        return 0  # Always cooperates


class AlwaysDefectAgent(Agent):
    def choose_action(self):
        """
        Choose an action based on the agent's policy.

        For AlwaysDefectAgent, this always returns 1, meaning the agent
        will always defect.

        Returns
        -------
        action : int
            The chosen action, either 0 (cooperate) or 1 (defect).
        """
        return 1  # Always defects


class TitForTatAgent(Agent):
    def __init__(self):
        super().__init__()
        self.last_opponent_action = 0  # Start by cooperating

    def choose_action(self):
        """
        Determine the next action based on the opponent's last action.

        This function implements the Tit for Tat strategy, where the agent
        simply mimics the opponent's last action. If the opponent cooperated
        in the last round, the agent will cooperate; if the opponent defected,
        the agent will defect.

        Returns
        -------
        int
            The chosen action, either 0 (cooperate) or 1 (defect).
        """
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
        """
        Choose an action based on the agent's policy.

        For PavlovAgent, this function uses the Pavlov strategy, where the
        agent cooperates if the last actions were the same, and defects if
        the last actions were different.

        Returns
        -------
        action : int
            The chosen action, either 0 (cooperate) or 1 (defect).
        """
        if self.history:
            return int(not (self.history[-1] == self.last_opponent_action))
        return 0

    def update(self, opponent_action):
        self.last_opponent_action = opponent_action


class SuspiciousPavlovAgent(Agent):
    def __init__(self):
        super().__init__()
        self.last_opponent_action = 1

    def choose_action(self):
        if self.history:
            return int(not (self.history[-1] == self.last_opponent_action))
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


class SuspiciousTitForTwoTatsAgent(Agent):
    def __init__(self):
        super().__init__()
        self.last_opponent_actions = [1, 1]

    def choose_action(self):
        if self.last_opponent_actions == [1, 1]:  # Last opponent defects twice
            return 1
        return 0

    def update(self, opponent_action):
        self.last_opponent_actions.append(opponent_action)
        self.last_opponent_actions.pop(0)

    def reset(self):
        self.last_opponent_actions = [1, 1]
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


class SuspiciousTwoTitsForTatAgent(Agent):
    def __init__(self):
        super().__init__()
        self.last_opponent_actions = [1, 1]

    def choose_action(self):
        if 1 in self.last_opponent_actions:
            return 1
        return 0

    def update(self, opponent_action):
        self.last_opponent_actions.append(opponent_action)
        self.last_opponent_actions.pop(0)

    def reset(self):
        self.last_opponent_actions = [1, 1]
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
        """
        Choose an action based on the agent's policy.

        This function implements an adaptive strategy, where the agent
        cooperates if the opponent cooperates more than 50% of the time and
        defects if the opponent defects more than 50% of the time.

        Returns
        -------
        int
            The chosen action, either 0 (cooperate) or 1 (defect).
        """
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


class SuspiciousAdaptiveAgent(Agent):
    def __init__(self, memory_size=10):
        super().__init__()
        self.opponent_actions = deque(maxlen=memory_size)

    def choose_action(self):
        if not self.opponent_actions:  # Default to cooperation if no history
            return 1
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


class SuspiciousGenerousTitForTatAgent(Agent):
    def __init__(self, forgiveness_prob=0.1):
        super().__init__()
        self.forgiveness_prob = forgiveness_prob
        self.last_opponent_action = 1

    def choose_action(self):
        if self.last_opponent_action == 1 and random.random() < self.forgiveness_prob:
            return 0  # Forgive with a certain probability
        return self.last_opponent_action

    def update(self, opponent_action):
        self.last_opponent_action = opponent_action

    def reset(self):
        self.last_opponent_action = 1
        self.history = []
        self.reward_history = []


class WinStayLoseShiftAgent(Agent):
    def choose_action(self):
        """
        Choose an action based on the agent's policy.

        If the reward history is not empty, the agent will stay with the last
        action if the last reward was positive, and shift to the other action
        otherwise.

        If the reward history is empty, the agent will cooperate.

        Returns
        -------
        action : int
            The chosen action, either 0 (cooperate) or 1 (defect).
        """
        if self.reward_history:  # Default to cooperation at the start
            if self.reward_history[-1] > 1:  # Stay if last reward was positive
                return self.history[-1]
            else:  # Shift otherwise
                return 1 - self.history[-1]
        return 0


class SuspiciousWinStayLoseShiftAgent(Agent):
    def choose_action(self):
        if self.reward_history:  # Default to cooperation at the start
            if self.reward_history[-1] > 1:  # Stay if last reward was positive
                return self.history[-1]
            else:  # Shift otherwise
                return 1 - self.history[-1]
        return 1


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
        """
        Choose an action based on the agent's policy.

        This function implements the Gradual strategy, where the agent
        gradually increases its retaliation count when the opponent defects
        and forgives after retaliating.

        Returns
        -------
        action : int
            The chosen action, either 0 (cooperate) or 1 (defect).
        """
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


class SuspiciousGradualAgent(Agent):
    def __init__(self):
        super().__init__()
        self.retaliation_count = 1
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
        self.retaliation_count = 1
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


class SuspiciousSoftMajorityAgent(Agent):
    def __init__(self):
        super().__init__()
        self.opponent_cooperation = 0
        self.opponent_defection = 1

    def choose_action(self):
        return 0 if self.opponent_cooperation >= self.opponent_defection else 1

    def update(self, opponent_action):
        if opponent_action == 0:
            self.opponent_cooperation += 1
        else:
            self.opponent_defection += 1

    def reset(self):
        self.opponent_cooperation = 0
        self.opponent_defection = 1
        self.history = []
        self.reward_history = []


class QLearningAgent(Agent):
    def __init__(self, q_table=None, alpha=0.01, gamma=0.5, epsilon=1.0):
        super().__init__()
        if q_table is None:
            self.q_table = np.random.uniform(
                -0.1, 0.1, (2, 2)
            )  # Q-values for (Agent_Action, Opponent_Action)
        else:
            self.q_table = np.array(q_table)
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.last_opponent_action = None

    def choose_action(self):
        if self.last_opponent_action is None or np.random.random() < self.epsilon:
            action = random.choice([0, 1])  # Explore: random action
        else:
            # Exploit: Choose action with the highest Q-value given opponent's last action
            action = np.argmax(self.q_table[self.last_opponent_action])
        return action

    def update_q_values(self, opponent_action, reward):
        if reward == 3:
            reward = 0.2
        elif reward == 0:
            reward = -1
        elif reward == 5:
            reward = 2
        else:
            reward = -0.5

        # Update Q-value using the Q-learning formula
        best_next_action = np.max(self.q_table[opponent_action])
        self.q_table[self.history[-1], opponent_action] += self.alpha * (
            reward
            + self.gamma * best_next_action
            - self.q_table[self.history[-1], opponent_action]
        )

        if self.epsilon > 0.001:
            self.epsilon *= 0.995

    def update(self, action):
        self.last_opponent_action = action

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

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = torch.relu((self.fc1(x)))
        x = torch.relu((self.fc2(x)))
        x = self.fc3(x)
        return x


class DeepQLearningAgent(Agent):
    def __init__(
        self,
        state_size=10,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.001,
        epsilon_decay=0.999,
        learning_rate=0.001,
        batch_size=1,
        load_model=True,
        path="deep_q_agent.pt",
        device="cpu",
    ):
        super().__init__()
        self.state_size = state_size
        self.memory = deque(maxlen=20000)  # Replay memory
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min  # Minimum exploration rate
        self.epsilon_decay = epsilon_decay  # Exploration decay rate
        self.batch_size = batch_size
        self.path = path
        self.device = device

        # Neural network and optimizer
        self.model = DeepQNetwork(state_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        if load_model:
            self.model.load_state_dict(torch.load(self.path))
            self.epsilon = 0.0

        # Initialize with default state
        self.prev_actions = [0] * state_size

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, self.get_reward(reward), next_state, done))

    def get_reward(self, reward):
        if reward == 3:
            if self.history and self.history[-1] == 1:
                return 2.5
            return 2
        elif reward == 0:
            return -2
        elif reward == 5:
            return 1
        return -1

    def act(self, state):
        if np.random.random() < self.epsilon:
            return random.choice([0, 1])  # Random action for exploration

        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)  # Predicted Q-values
        action = torch.argmax(q_values).item()
        return action

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            reward = torch.FloatTensor([reward]).to(self.device)
            target = reward
            if not done:
                target += self.gamma * torch.max(self.model(next_state).detach())

            current_q = self.model(state).squeeze(0)[action].unsqueeze(0)

            loss = self.loss_fn(current_q, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
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
        return self.act(state)


ALL_AGENTS = {
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
    SuspiciousAdaptiveAgent(),
    SuspiciousGenerousTitForTatAgent(),
    SuspiciousGradualAgent(),
    SuspiciousPavlovAgent(),
    SuspiciousSoftMajorityAgent(),
    SuspiciousTitForTwoTatsAgent(),
    SuspiciousTwoTitsForTatAgent(),
    SuspiciousWinStayLoseShiftAgent(),
}

BASIC_AGENTS = [
    agent for agent in ALL_AGENTS if "QLearning" not in agent.__class__.__name__
]

HIGHEST_REWARDS = {
    "RandomAgent": 300,
    "TitForTatAgent": 300,
    "AlwaysCooperateAgent": 500,
    "AlwaysDefectAgent": 100,
    "PavlovAgent": 500,
    "TitForTatOppositeAgent": 500,
    "TwoTitsForTatAgent": 300,
    "SpitefulAgent": 300,
    "ProvocativeAgent": 335,
    "TitForTwoTatsAgent": 400,
    "GradualAgent": 300,
    "SuspiciousTitForTatAgent": 298,
    "WinStayLoseShiftAgent": 300,
    "GenerousTitForTatAgent": 300,
    "AdaptiveAgent": 400,
    "SoftMajorityAgent": 300,
    "SuspiciousAdaptiveAgent": 398,
    "SuspiciousGenerousTitForTatAgent": 298,
    "SuspiciousGradualAgent": 298,
    "SuspiciousPavlovAgent": 498,
    "SuspiciousSoftMajorityAgent": 498,
    "SuspiciousTitForTwoTatsAgent": 398,
    "SuspiciousTwoTitsForTatAgent": 298,
    "SuspiciousWinStayLoseShiftAgent": 298,
}


class RandomStrategies(Agent):
    def __init__(self):
        super().__init__()
        self.agent = random.choice(BASIC_AGENTS)

    def choose_action(self):
        return self.agent.choose_action()

    def update(self, action):
        self.agent.update(action)

    def reset(self):
        self.agent.reset()
        self.agent = random.choice(BASIC_AGENTS)


if __name__ == "__main__":
    # for agent in BASIC_AGENTS:
    q_agent = DeepQLearningAgent(state_size=5, path="deep_q_100.pt")
    game = Game(q_agent, RandomAgent(), verbose=2)
    a, b = game.play_iterated_game(10)
    # print(len(ALL_AGENTS))
    # print(len(BASIC_AGENTS))

    # print(q_agent.train_multiple_agents())
