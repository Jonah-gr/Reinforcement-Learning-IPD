from src.train import *
from src.tournament import Tournament
from src.game import Game
from app.app import app, WebUser
import argparse
import ast
import webbrowser


def create_object_from_string(input_string):
    # Parse the string to an AST node
    try:
        expr = ast.parse(input_string, mode="eval")

        if not isinstance(expr.body, ast.Call):
            raise ValueError("Input must be a call expression, e.g., ClassName(arg=value)")

        func_name = expr.body.func.id

        cls = globals().get(func_name)
        if cls is None or not isinstance(cls, type):
            raise ValueError(f"Class '{func_name}' is not defined.")

        # Extract keyword arguments
        kwargs = {}
        for keyword in expr.body.keywords:
            kwargs[keyword.arg] = ast.literal_eval(keyword.value)

        return cls(**kwargs)

    except Exception as e:
        raise ValueError(f"Invalid input: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run an iterated game simulation.")

    # Subcommands for the script
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand for the game
    game_parser = subparsers.add_parser("game", help="Run a game")
    game_parser.add_argument(
        "--agents",
        nargs=2,
        required=True,
        help="Specify the two agents to compete, e.g., QLearningAgent(epsilon=0.5) RandomAgent",
    )
    game_parser.add_argument(
        "--num_rounds",
        type=int,
        default=100,
        help="Specify the number of rounds to play",
    )
    game_parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Set verbosity level (0 = no output, 1 = summary, 2 = detailed)",
    )

    # Subcommand for training
    train_parser = subparsers.add_parser("train", help="Train an agent")
    train_parser.add_argument(
        "--agents",
        nargs="+",
        required=True,
        help="Specify the agent to train, e.g., QLearningAgent(epsilon=0.5)",
    )
    train_parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Specify the number of episodes to train",
    )
    train_parser.add_argument(
        "--num_rounds",
        type=int,
        default=100,
        help="Specify the number of rounds to train",
    )
    train_parser.add_argument(
        "--log_dir",
        type=str,
        default="runs/training",
        help="Specify the directory to save logs",
    )

    # Subcommand for tournament
    tournament_parser = subparsers.add_parser("tournament", help="Run a tournament")
    tournament_parser.add_argument(
        "--agents",
        nargs="+",
        required=True,
        help="Specify the agents to compete, e.g., QLearningAgent(epsilon=0.5) RandomAgent()",
    )
    tournament_parser.add_argument(
        "--num_games",
        type=int,
        default=100,
        help="Specify the number of games to play against each agent",
    )
    tournament_parser.add_argument(
        "--num_rounds",
        type=int,
        default=100,
        help="Specify the number of rounds to play",
    )
    tournament_parser.add_argument(
        "--include_params",
        type=bool,
        default=False,
        help="Include agent parameters in the output",
    )
    tournament_parser.add_argument(
        "--save_dir",
        type=str,
        default="runs/tournament/tournament_results.csv",
        help="Specify the directory to save results",
    )

    # Subcommand for app
    app_parser = subparsers.add_parser("app", help="Run the app")
    app_parser.add_argument(
        "--debug",
        type=bool,
        default=False,
        help="Run the app in debug mode",
    )
    app_parser.add_argument(
        "--path",
        type=str,
        default="deep_q_agent.pt",
        help="Path to the saved agent file",
    )

    args = parser.parse_args()

    if args.command != "app":
        agent_classes = {cls.__class__.__name__: cls for cls in ALL_AGENTS}
        try:
            for i in range(len(args.agents)):
                args.agents[i] = create_object_from_string(args.agents[i])
        except Exception as e:
            print(f"Error: Failed to initialize agents. {e}")
            print("Available agents:", ", ".join(agent_classes.keys()))
            return

    if args.command == "game":
        game = Game(args.agents[0], args.agents[1], verbose=args.verbose)
        print(
            f"Starting game between {args.agents[0].__class__.__name__} and {args.agents[1].__class__.__name__} for {args.num_rounds} rounds..."
        )
        game.play_iterated_game(num_rounds=args.num_rounds)

    elif args.command == "train":
        train(
            agents=args.agents,
            num_rounds=args.num_rounds,
            episodes=args.episodes,
            log_dir=args.log_dir,
        )

    elif args.command == "tournament":
        tournament = Tournament(
            agents=args.agents,
            num_games=args.num_games,
            num_rounds=args.num_rounds,
            inlcude_params=args.include_params,
        )
        tournament.play_tournament()
        tournament.print_summary()
        tournament.save_results(args.save_dir)

    elif args.command == "app":
        app.deep_q_agent = DeepQLearningAgent(state_size=20, path=args.path)
        app.web_user = WebUser()
        app.game = Game(app.deep_q_agent, app.web_user)
        webbrowser.open("http://127.0.0.1:5000")
        app.run(args.debug)


if __name__ == "__main__":
    main()
