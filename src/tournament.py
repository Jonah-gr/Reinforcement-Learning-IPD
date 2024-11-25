from src.agents import *
import itertools
import pandas as pd
from tqdm import tqdm

class Tournament:
    def __init__(self, agents, num_games, num_rounds):
        self.agents = agents
        self.num_games = num_games
        self.num_rounds = num_rounds
        self.results = pd.DataFrame(columns=["agent_a", "agent_a_state", "agent_b", "agent_b_state", "total_reward_a", "total_reward_b"])

    def play_tournament(self):
        agent_pairs = list(itertools.combinations(self.agents, 2))
        index = 0
        # with open("results.csv", "w", newline="") as csvfile:
            # writer = csv.writer(csvfile)
        for agent_a, agent_b in tqdm(agent_pairs):
            for _ in range(self.num_games):
                agent_a.reset()
                agent_b.reset()
                
                # Create a game and play
                game = Game(agent_a, agent_b)
                total_reward_a, total_reward_b = game.play_iterated_game(self.num_rounds)

                # Save results
                self.results.loc[index] = [type(agent_a).__name__, agent_a.__get_params__(), type(agent_b).__name__, agent_b.__get_params__(), total_reward_a, total_reward_b]
                # writer.writerow(self.results.iloc[-1])
                index += 1

    def save_results(self, filename):
        self.results.to_csv(filename, index=True)

    def print_summary(self):
        print(self.results)


# Example usage:
if __name__ == "__main__":
    agents = [
        RandomAgent(),
        AlwaysCooperateAgent(),
        AlwaysDefectAgent(),
        TitForTatAgent(),
        SpitefulAgent(),
        PavlovAgent(),
        TitForTwoTatsAgent(),
        TwoTitsForTatAgent(),
        TitForTatOppositeAgent(),
        ProvocativeAgent(),
        QLearningAgent(q_table=[[8.80625374, 7.12855422], [10.78324676, 8.13047989]], epsilon=1),
        DeepQLearningAgent(state_size=5),
    ]

    tournament = Tournament(agents, num_games=100, num_rounds=100)
    tournament.play_tournament()
    tournament.print_summary()
    tournament.save_results("C://Users/jonah/OneDrive - Hochschule Düsseldorf/tournament_results.csv")


