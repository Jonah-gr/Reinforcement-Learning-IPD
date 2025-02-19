from src.agents import *
import pandas as pd
from tqdm import tqdm


class Tournament:
    def __init__(self, agents, num_games, num_rounds, inlcude_params=False):
        self.agents = agents + BASIC_AGENTS
        self.num_games = num_games
        self.num_rounds = num_rounds
        self.include_params = inlcude_params

        if inlcude_params:
            columns = [
                "agent_a",
                "agent_a_def",
                "agent_b",
                "agent_b_def",
                "total_reward_a",
                "total_reward_b",
            ]
        else:
            columns = [
                "agent_a",
                "agent_b",
                "total_reward_a",
                "total_reward_b",
            ]
        self.results = pd.DataFrame(columns=columns)

    def play_tournament(self):
        """
        Play a tournament with all agents in self.agents against each other agent in
        BASIC_AGENTS.

        For each pair of agents, play self.num_games games and record the total reward
        for each agent in the self.results DataFrame. If self.include_params is True,
        also record the parameters of each agent in the DataFrame.

        Returns
        -------
        None
        """
        index = 0
        for agent_a in tqdm(self.agents):
            for agent_b in BASIC_AGENTS:
                total_reward_a = 0
                total_reward_b = 0
                agent_a_def = 0
                agent_b_def = 0
                for _ in range(self.num_games):
                    agent_a.reset()
                    agent_b.reset()

                    game = Game(agent_a, agent_b)
                    reward_a, reward_b = game.play_iterated_game(self.num_rounds)
                    total_reward_a += reward_a
                    total_reward_b += reward_b

                    if self.include_params:
                        agent_a_def += sum(agent_a.__get_params__()[0])
                        agent_b_def += sum(agent_b.__get_params__()[0])

                if self.include_params:
                    self.results.loc[index] = [
                        type(agent_a).__name__,
                        agent_a_def,
                        type(agent_b).__name__,
                        agent_b_def,
                        total_reward_a,
                        total_reward_b,
                    ]
                else:
                    self.results.loc[index] = [
                        type(agent_a).__name__,
                        type(agent_b).__name__,
                        total_reward_a,
                        total_reward_b,
                    ]
                index += 1

    def save_results(self, filename):
        self.results.to_csv(filename, index=False)

    def print_summary(self):
        print(self.results)


if __name__ == "__main__":
    agents = [
        QLearningAgent(q_table=[[0.4, 0.08562763], [2.17148565, 0.24526072]], epsilon=0.0),
        DeepQLearningAgent(state_size=20),
        RandomStrategies(),
    ]
    # for agent in agents:
    tournament = Tournament(agents, num_games=100, num_rounds=100)
    tournament.play_tournament()
    tournament.print_summary()
    tournament.save_results("tournament_results.csv")
