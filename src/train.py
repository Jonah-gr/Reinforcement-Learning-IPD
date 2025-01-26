from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
from src.agents import *


def train(
    agents,
    num_rounds=100,
    episodes=1000,
    log_dir="runs/training",
):
    """
    Train a list of agents against randomly selected opponent agents over multiple episodes.

    Args:
        agents (list): A list of agents to be trained.
        num_rounds (int, optional): Number of rounds to be played in each episode. Default is 100.
        episodes (int, optional): Number of episodes for training each agent. Default is 1000.
        log_dir (str, optional): Directory path for storing TensorBoard logs. Default is "runs/training".

    The function initializes a TensorBoard writer for logging training metrics.
    For each agent, it selects random opponents from BASIC_AGENTS and plays a specified number of episodes.
    During each episode, rewards and losses are tracked and logged.
    The function handles training for agents with class names "DeepQLearningAgent" and "QLearningAgent",
    including updating their models and Q-values respectively.
    Trained models for DeepQLearningAgent are saved to disk. TensorBoard writer is closed after completion.
    """
    # TensorBoard writer
    if os.path.exists(log_dir):
        counter = 1
        while os.path.exists(f"{log_dir}{counter}"):
            counter += 1
        writer = SummaryWriter(log_dir=f"{log_dir}{counter}")
    else:
        writer = SummaryWriter(log_dir=log_dir)

    # List of opponent agents
    opponents = random.choices(BASIC_AGENTS, k=episodes)

    for agent in agents:
        all_rewards = []
        highest_posssible_rewards = []

        for episode in tqdm(range(episodes), desc="Training Episodes"):
            # Choose a random opponent for this episode
            opponent = opponents[episode]
            game = Game(agent, opponent)

            total_reward_a = 0

            bar = tqdm(range(num_rounds), leave=False)
            for _ in bar:
                if game.agent_a.__class__.__name__ == "DeepQLearningAgent":
                    state_a = np.array([game.agent_a.prev_actions])
                (
                    action_a,
                    action_b,
                    reward_a,
                    _,
                ) = game.play_round()
                game.agent_a.update(action_b)
                game.agent_b.update(action_a)
                if game.agent_a.__class__.__name__ == "DeepQLearningAgent":
                    next_state_a = np.array([game.agent_a.prev_actions])
                    game.agent_a.remember(state_a, action_a, reward_a, next_state_a, False)
                    loss = game.agent_a.replay()
                    if loss is None:
                        loss = 0

                if game.agent_a.__class__.__name__ == "QLearningAgent":
                    game.agent_a.update_q_values(action_b, reward_a)

                total_reward_a += reward_a

                bar.set_description(
                    f"Episode: {episode}/{episodes}, Score: {total_reward_a}, Agent: {game.agent_a.__class__.__name__}, Opponent: {opponent.__class__.__name__}"
                )
            all_rewards.append(total_reward_a)

            # Log metrics to TensorBoard

            writer.add_scalars(
                f"{game.agent_a.__class__.__name__} Rewards/Episode",
                {
                    f"{game.agent_a.__class__.__name__}": total_reward_a,
                    f"AVG {game.agent_a.__class__.__name__}": sum(all_rewards) / (episode + 1),
                    "Maximum": HIGHEST_REWARDS[opponent.__class__.__name__],
                },
                episode,
            )

            if game.agent_a.__class__.__name__ == "DeepQLearningAgent":
                writer.add_scalar("Loss/Episode", loss, episode)
                writer.add_scalar("Epsilon", game.agent_a.epsilon, episode)
                writer.add_scalar("Temparature", game.agent_a.temperature, episode)

            if (
                game.agent_a.__class__.__name__ == "QLearningAgent"
                or game.agent_a.__class__.__name__ == "DeepQLearningAgent"
            ):
                writer.add_scalar("sum(cooperate,defect)/Episode", sum(game.agent_a.history), episode)

            highest_posssible_rewards.append(HIGHEST_REWARDS[opponent.__class__.__name__])

            game.agent_a.reset()
            game.agent_b.reset()

        if game.agent_a.__class__.__name__ == "QLearningAgent":
            print(game.agent_a.q_table)

    # Save the trained model
    if game.agent_a.__class__.__name__ == "DeepQLearningAgent":
        torch.save(game.agent_a.model.state_dict(), game.agent_a.path)
        print(f"Trained model saved to {game.agent_a.path}")

    # Close TensorBoard writer
    writer.close()


if __name__ == "__main__":
    train(
        agents=[DeepQLearningAgent(), QLearningAgent(), RandomStrategies()],
        num_rounds=100,
        episodes=10000,
    )
