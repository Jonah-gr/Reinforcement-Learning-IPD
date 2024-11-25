from matplotlib import pyplot as plt
from tqdm import tqdm
from src.agents import *


def train_q_agent(episodes=1000, num_rounds=100):
    opponents = [
        AlwaysCooperateAgent(),
        AlwaysDefectAgent(),
        TitForTatAgent(),
        SpitefulAgent(),
        RandomAgent(),
        PavlovAgent(),
        TitForTwoTatsAgent(),
        TwoTitsForTatAgent(),
        TitForTatOppositeAgent(),
        ProvocativeAgent(),
    ]
    result = 0
    for episode in tqdm(range(episodes)):
        alpha, gamma, epsilon = (
            np.random.random(),
            np.random.random(),
            np.random.random(),
        )
        q_agent = QLearningAgent(alpha=alpha, gamma=gamma, epsilon=epsilon)
        q_agent.reset()
        opponent = random.choice(opponents)

        game = Game(q_agent, opponent)

        for _ in range(num_rounds):
            # print(opponent)
            action_a, action_b, reward_a, reward_b = game.play_round()
            game.agent_a.update_q_values(action_b, reward_a)
            game.agent_b.update(action_a)

            result += reward_a

    return game.agent_a.q_table, game.agent_a.epsilon, result / episodes


def q_plot():
    (
        best_params,
        best_score,
        best_q_table,
        best_q_agent_history,
        best_opponent_history,
        best_opponents_list,
        best_q_actions,
    ) = train_q_agent()
    print(f"Best params: {best_params}")
    print(f"Best score: {best_score}")
    print(f"Best q_table: {best_q_table}")
    print(
        f"Best q_actions: Cooperate: {best_q_actions.count(0)}, Defect: {best_q_actions.count(1)}"
    )

    plt.plot(best_q_agent_history, label="q_agent")
    x = 0
    for opponent in best_opponents_list:
        x += 100
        plt.plot(
            range(x - 100, x),
            best_opponent_history[x - 100 : x],
            label=f"{opponent.__class__.__name__}",
        )
    plt.legend()
    plt.show()


def train_deep_q_agent(
    state_size=5,
    action_size=2,
    num_rounds=100,
    batch_size=32,
    episodes=1000,
    save_path="deep_q_agent.pt",
):
    # Initialize the DQN agent
    deep_q_agent = DeepQLearningAgent(state_size, action_size, load_model=False)

    # List of opponent agents
    opponents = [
        AlwaysCooperateAgent(),
        AlwaysDefectAgent(),
        TitForTatAgent(),
        SpitefulAgent(),
        RandomAgent(),
        PavlovAgent(),
        TitForTwoTatsAgent(),
        QLearningAgent(q_table=[[5.32301959, 3.19583982], [7.32301959, 4.19583982]]),
        TitForTatOppositeAgent(),
        TwoTitsForTatAgent(),
        ProvocativeAgent(),
    ]

    all_rewards = []
    highest_posssible_rewards = []

    for episode in tqdm(range(episodes), desc="Training Episodes"):
        deep_q_agent.reset()
        # Choose a random opponent for this episode
        opponent = random.choice(opponents)
        game = Game(deep_q_agent, opponent)

        # Play iterated games and train
        total_reward_a = 0
        all_oponent_actions = []

        # Initialize replay memory
        next_state_a = np.array([0] * state_size)

        bar = tqdm(range(num_rounds))
        for _ in bar:
            action_a = game.agent_a.choose_action()
            action_b = game.agent_b.choose_action()
            reward_a, _ = game.payoff_matrix[(action_a, action_b)]

            # Create states for replay memory
            state_a = np.array([game.agent_a.prev_opponent_action])
            game.agent_a.update(action_b)
            game.agent_b.update(action_a)
            next_state_a = np.array([game.agent_a.prev_opponent_action])
            # print(state_a, next_state_a)

            # Update replay memory
            game.agent_a.remember(state_a, action_a, reward_a, next_state_a, False)
            total_reward_a += reward_a

            # Train the agents with replay memory
            game.agent_a.replay(batch_size)

            # Safe opponent action
            all_oponent_actions.append(action_b)

            bar.set_description(
                f"Episode: {episode}/{episodes}, Score: {total_reward_a}, Opponent: {opponent.__class__.__name__}"
            )

        all_rewards.append(total_reward_a)
        highest_posssible_rewards.append(
            all_oponent_actions.count(0) * 5 + all_oponent_actions.count(1)
        )

    # Save the trained model
    plt.plot(
        range(episodes),
        list(map(lambda x: x[0] / x[1], zip(all_rewards, highest_posssible_rewards))),
    )
    plt.show()
    torch.save(deep_q_agent.model.state_dict(), save_path)
    print(f"Trained model saved to {save_path}")


if __name__ == "__main__":
    train_deep_q_agent(
        state_size=5, action_size=2, num_rounds=100, batch_size=32, episodes=100
    )

    # best_q_table = 0
    # best_epsilon = 0
    # best_result = 0
    # for i in tqdm(range(100)):
    #     q_table, epsilon, result = train_q_agent()
    #     if result > best_result:
    #         best_q_table = q_table
    #         best_epsilon = epsilon
    #         best_result = result
    # print(f"Best q_table: {best_q_table}, Best epsilon: {best_epsilon}, Best result: {best_result}")

    # total_reward_a, total_reward_b = game.play_iterated_game(num_rounds, batch_size)
    # print(f"Total Reward for Agent A: {total_reward_a}")
    # print(f"Total Reward for Agent B: {total_reward_b}")
