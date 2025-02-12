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


class User(Agent):
    def choose_action(self):
        """
        Choose an action based on user input.

        Returns
        -------
        action : int
            The chosen action, either 0 (cooperate) or 1 (defect).
        """
        action = int(input("Enter your action (0 or 1): "))
        return action


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
        """
        Choose an action based on the agent's policy.

        This function implements the Spiteful strategy, where the agent
        cooperates until the opponent defects, and then defects from then on.

        Returns
        -------
        int
            The chosen action, either 0 (cooperate) or 1 (defect).
        """

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
        """
        Choose an action based on the agent's policy.

        For SuspiciousPavlovAgent, this function uses the Pavlov strategy, where the
        agent cooperates if the last actions were the same, and defects if
        the last actions were different. The agent starts by cooperating.

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


class TitForTwoTatsAgent(Agent):
    def __init__(self):
        super().__init__()
        self.last_opponent_actions = [0, 0]

    def choose_action(self):
        """
        Decide the next action based on the opponent's last two actions.

        Implements the Tit for Two Tats strategy, where the agent defects
        only if the opponent has defected in the last two consecutive rounds.
        Otherwise, the agent cooperates.

        Returns
        -------
        int
            The chosen action, either 0 (cooperate) or 1 (defect).
        """
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
        """
        Decide the next action based on the opponent's last two actions.

        Implements the Suspicious Tit for Two Tats strategy, where the agent
        defects only if the opponent has defected in the last two consecutive
        rounds. Otherwise, the agent cooperates. The agent starts by defecting.

        Returns
        -------
        int
            The chosen action, either 0 (cooperate) or 1 (defect).
        """
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
        """
        Decide the next action based on the opponent's last two actions.

        Implements the Two Tits for Tat strategy, where the agent
        defects only if the opponent has defected in at least one of
        the last two consecutive rounds. Otherwise, the agent cooperates.

        Returns
        -------
        int
            The chosen action, either 0 (cooperate) or 1 (defect).
        """
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
        """
        Decide the next action based on the opponent's last two actions.

        Implements the Suspicious Two Tits for Tat strategy, where the agent
        defects only if the opponent has defected in at least one of
        the last two consecutive rounds. Otherwise, the agent cooperates.
        The agent starts by defecting.

        Returns
        -------
        int
            The chosen action, either 0 (cooperate) or 1 (defect).
        """
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
        """
        Decide the next action based on the agent's own previous actions.

        The Provocative Agent defects if the agent has cooperated in the last two
        rounds, otherwise the agent cooperates.

        Returns
        -------
        int
            The chosen action, either 0 (cooperate) or 1 (defect).
        """
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
        """
        Decide the next action based on the opponent's last action.

        This function implements the Tit for Tat Opposite strategy, where the agent
        simply copies the opponent's last action. If the opponent cooperated in the
        last round, the agent will defect; if the opponent defected, the agent will
        cooperate.

        Returns
        -------
        int
            The chosen action, either 0 (cooperate) or 1 (defect).
        """
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
        """
        Choose an action based on the agent's policy.

        This function implements a suspicious adaptive strategy, where the agent
        defects if the opponent cooperates more than 50% of the time and
        cooperates if the opponent defects more than 50% of the time. The agent
        starts by defecting.

        Returns
        -------
        int
            The chosen action, either 0 (cooperate) or 1 (defect).
        """
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
        """
        Choose an action based on the agent's policy.

        This function implements a generous Tit-For-Tat strategy, where the
        agent cooperates if the opponent cooperated, defects if the opponent
        defected, and forgives (cooperates) with a certain probability if the
        opponent defected.

        Returns
        -------
        int
            The chosen action, either 0 (cooperate) or 1 (defect).
        """
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
        """
        Choose an action based on the agent's policy.

        This function implements a suspicious generous Tit-For-Tat strategy, where the
        agent cooperates if the opponent cooperated, defects if the opponent defected,
        and forgives (cooperates) with a certain probability if the opponent defected.
        The agent starts by defecting.

        Returns
        -------
        int
            The chosen action, either 0 (cooperate) or 1 (defect).
        """
        if self.last_opponent_action == 1 and random.random() < self.forgiveness_prob:
            if self.history:
                return 0  # Forgive with a certain probability
            else:
                return 1
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
        """
        Choose an action based on the agent's policy.

        For SuspiciousWinStayLoseShiftAgent, this function implements a strategy
        where the agent stays with the last action if the last reward was positive,
        and shifts to the other action otherwise. The agent starts by defecting.

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
        return 1


class SuspiciousTitForTatAgent(Agent):
    def __init__(self):
        super().__init__()
        self.last_opponent_action = 1  # Start by defecting

    def choose_action(self):
        """
        Choose an action based on the agent's policy.

        For SuspiciousTitForTatAgent, this function implements a strategy
        where the agent simply copies the opponent's last action, starting
        by defecting.

        Returns
        -------
        int
            The chosen action, either 0 (cooperate) or 1 (defect).
        """
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
            self.retaliation_count += 1  # Add one round of retaliation for each defection
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
        """
        Choose an action based on the agent's policy.

        This function implements the SuspiciousGradual strategy, where the agent
        starts by defecting and then gradually increases its retaliation count
        when the opponent defects and forgives after retaliating.

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
            self.retaliation_count += 1  # Add one round of retaliation for each defection
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
        """
        Decide the next action based on the opponent's cooperation and defection count.

        Implements a strategy where the agent cooperates if the opponent has cooperated
        at least as many times as they have defected. Otherwise, the agent defects.

        Returns
        -------
        int
            The chosen action, either 0 (cooperate) or 1 (defect).
        """
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
        """
        Decide the next action based on the opponent's cooperation and defection count.

        Implements a strategy where the agent cooperates if the opponent has cooperated
        at least as many times as they have defected. Otherwise, the agent defects.
        The agent starts by defecting.

        Returns
        -------
        int
            The chosen action, either 0 (cooperate) or 1 (defect).
        """
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
    def __init__(self, q_table=None, alpha=0.001, gamma=0.95, epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.00001):
        super().__init__()
        if q_table is None:
            self.q_table = np.random.uniform(-0.1, 0.1, (2, 2))  # Q-values for (Agent_Action, Opponent_Action)
        else:
            self.q_table = np.array(q_table)
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay  # Exploration decay rate
        self.epsilon_min = epsilon_min
        self.last_opponent_action = 0

    def choose_action(self):
        """
        Choose an action based on the agent's policy.

        The agent explores with probability epsilon, choosing a random action, and
        exploits with probability (1 - epsilon), choosing the action that maximizes
        the Q-value given the opponent's last action.

        Returns
        -------
        int
            The chosen action, either 0 (cooperate) or 1 (defect).
        """
        if np.random.random() < self.epsilon:
            action = random.choice([0, 1])  # Explore: random action
        else:
            # Exploit: Choose action with the highest Q-value given opponent's last action
            action = np.argmax(self.q_table[self.last_opponent_action])
        return action

    def update_q_values(self, opponent_action, reward):
        # Update Q-value using the Q-learning formula
        best_next_action = np.max(self.q_table[opponent_action])
        self.q_table[self.last_opponent_action, self.history[-1]] += self.alpha * (
            reward + self.gamma * best_next_action - self.q_table[self.last_opponent_action, self.history[-1]]
        )

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update(self, action):
        self.last_opponent_action = action

    def reset(self):
        self.last_opponent_action = 0
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
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.00001,
        epsilon_decay=0.9995,
        learning_rate=0.001,
        tau=0.005,
        batch_size=5,
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
        self.tau = tau
        self.batch_size = batch_size
        self.path = path
        self.device = device

        # Neural network and optimizer
        self.model = DeepQNetwork(state_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)  ### Adam
        self.loss_fn = nn.MSELoss()

        if load_model:
            self.model.load_state_dict(torch.load(self.path))
            self.epsilon = 0.0

        self.target_model = DeepQNetwork(state_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())  # Copy initial weights
        self.target_model.eval()

        # Initialize with default state
        self.prev_actions = [0] * state_size

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Choose an action based on the agent's policy.

        The agent explores with probability epsilon, choosing a random action, and
        exploits with probability (1 - epsilon), choosing the action that maximizes
        the Q-value given the current state.

        Args:
            state (list): The current state of the game.

        Returns:
            int: The chosen action, either 0 (cooperate) or 1 (defect).
        """
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
                target += self.gamma * torch.max(self.target_model(next_state).detach())

            current_q = self.model(state).squeeze(0)[action].unsqueeze(0)

            loss = self.loss_fn(current_q, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.update_target_model()
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

    def update_target_model(self):
        target_net_state_dict = self.target_model.state_dict()
        for key in self.model.state_dict():
            target_net_state_dict[key] = self.model.state_dict()[key] * self.tau + target_net_state_dict[key] * (
                1 - self.tau
            )

        self.target_model.load_state_dict(self.model.state_dict())

    def choose_action(self):
        """
        Choose an action based on the agent's policy.

        The agent's policy is based on its memory of the last self.state_size actions, which
        are stored in the `prev_actions` list. The agent uses this list to
        create a state tensor, which is then passed to the `act` method to
        determine the next action.

        Returns
        -------
        int
            The chosen action, either 0 (cooperate) or 1 (defect).
        """
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

BASIC_AGENTS = [agent for agent in ALL_AGENTS if "QLearning" not in agent.__class__.__name__]

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
        self.history = []
        self.reward_history = []


if __name__ == "__main__":
    # for agent in BASIC_AGENTS:
    # q_agent = DeepQLearningAgent(state_size=5, path="deep_q_100.pt")
    # game = Game(q_agent, RandomAgent(), verbose=2)
    # a, b = game.play_iterated_game(10)
    print(len(ALL_AGENTS))
    print(len(BASIC_AGENTS))

    # print(q_agent.train_multiple_agents())
