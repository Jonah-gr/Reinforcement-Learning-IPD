class Game:
    def __init__(self, agent_a, agent_b):
        self.payoff_matrix = {
            (0, 0): (3, 3),  # Both cooperate
            (0, 1): (0, 5),  # Player A cooperates, Player B defects
            (1, 0): (5, 0),  # Player A defects, Player B cooperates
            (1, 1): (1, 1),  # Both defect
        }
        self.agent_a = agent_a
        self.agent_b = agent_b

    def play_round(self):
        action_a = self.agent_a.choose_action()
        action_b = self.agent_b.choose_action()
        reward_a, reward_b = self.payoff_matrix[(action_a, action_b)]
        self.agent_a.history.append(action_a)
        self.agent_b.history.append(action_b)
        self.agent_a.reward_history.append(reward_a)
        self.agent_b.reward_history.append(reward_b)
        return action_a, action_b, reward_a, reward_b

    def play_iterated_game(self, num_rounds):
        total_reward_a = 0
        total_reward_b = 0
        for _ in range(num_rounds):
            action_a, action_b, reward_a, reward_b = self.play_round()
            self.agent_a.update(action_b)
            self.agent_b.update(action_a)
            total_reward_a += reward_a
            total_reward_b += reward_b
        return total_reward_a, total_reward_b
