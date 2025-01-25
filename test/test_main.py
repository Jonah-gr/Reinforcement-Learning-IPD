import pytest
from src.main import create_object_from_string, main
from src.agents import RandomAgent, QLearningAgent
from src.game import Game
from src.tournament import Tournament
import argparse
import sys
from unittest.mock import patch, MagicMock


# Test `create_object_from_string` function
def test_create_object_from_string_valid():
    agent = create_object_from_string("QLearningAgent(epsilon=0.5)")
    assert isinstance(agent, QLearningAgent)
    assert agent.epsilon == 0.5

    agent = create_object_from_string("RandomAgent()")
    assert isinstance(agent, RandomAgent)


def test_create_object_from_string_invalid():
    with pytest.raises(ValueError, match="Class 'NonExistentAgent' is not defined."):
        create_object_from_string("NonExistentAgent()")

    with pytest.raises(ValueError, match="Input must be a call expression"):
        create_object_from_string("QLearningAgent")


# Mock functions for CLI subcommands
@pytest.fixture
def mock_game():
    with patch("src.main.Game") as mock_game_cls:
        yield mock_game_cls


@pytest.fixture
def mock_train():
    with patch("src.main.train") as mock_train_func:
        yield mock_train_func


@pytest.fixture
def mock_tournament():
    with patch("src.main.Tournament") as mock_tournament_cls:
        yield mock_tournament_cls


# Test the "game" subcommand
def test_game_subcommand(mock_game):
    args = [
        "game",
        "--agents",
        "RandomAgent()",
        "QLearningAgent(epsilon=0.5)",
        "--num_rounds",
        "10",
        "--verbose",
        "1",
    ]

    with patch.object(sys, "argv", ["main.py"] + args):
        main()

    mock_game.assert_called_once()
    agent_a, agent_b = mock_game.call_args[0]
    assert isinstance(agent_a, RandomAgent)
    assert isinstance(agent_b, QLearningAgent)
    assert agent_b.epsilon == 0.5
    assert mock_game.call_args[1]["verbose"] == 1


# Test the "train" subcommand
def test_train_subcommand(mock_train):
    args = [
        "train",
        "--agents",
        "QLearningAgent(epsilon=0.5)",
        "--episodes",
        "1000",
        "--num_rounds",
        "100",
        "--log_dir",
        "runs/training",
    ]

    with patch.object(sys, "argv", ["main.py"] + args):
        main()

    mock_train.assert_called_once()
    agent = mock_train.call_args[1]["agents"][0]
    assert isinstance(agent, QLearningAgent)
    assert agent.epsilon == 0.5
    assert mock_train.call_args[1]["episodes"] == 1000
    assert mock_train.call_args[1]["log_dir"] == "runs/training"


# Test the "tournament" subcommand
def test_tournament_subcommand(mock_tournament):
    args = [
        "tournament",
        "--agents",
        "RandomAgent()",
        "QLearningAgent(epsilon=0.5)",
        "--num_games",
        "50",
        "--num_rounds",
        "100",
        "--include_params",
        "True",
        "--save_dir",
        "runs/tournament",
    ]

    with patch.object(sys, "argv", ["main.py"] + args):
        main()

    mock_tournament.assert_called_once()
    agents = mock_tournament.call_args[1]["agents"]
    assert isinstance(agents[0], RandomAgent)
    assert isinstance(agents[1], QLearningAgent)
    assert agents[1].epsilon == 0.5
    assert mock_tournament.call_args[1]["num_games"] == 50
    # assert mock_tournament.call_args[1]["save_dir"] == "runs/tournament"


# Test invalid subcommand
def test_invalid_command():
    args = ["invalid_command"]

    with patch.object(sys, "argv", ["main.py"] + args), pytest.raises(SystemExit):
        main()


# Test invalid agent initialization in game
def test_invalid_agent_in_game():
    args = [
        "game",
        "--agents",
        "NonExistentAgent()",
        "RandomAgent()",
        "--num_rounds",
        "10",
    ]

    with patch.object(sys, "argv", ["main.py"] + args), patch(
        "builtins.print"
    ) as mock_print:
        main()

    mock_print.assert_any_call(
        "Error: Failed to initialize agents. Invalid input: Class 'NonExistentAgent' is not defined."
    )


# Test invalid agent initialization in train
def test_invalid_agent_in_train():
    args = [
        "train",
        "--agents",
        "NonExistentAgent()",
        "--episodes",
        "1000",
    ]

    with patch.object(sys, "argv", ["main.py"] + args), patch(
        "builtins.print"
    ) as mock_print:
        main()

    mock_print.assert_any_call(
        "Error: Failed to initialize agents. Invalid input: Class 'NonExistentAgent' is not defined."
    )


# Test invalid agent initialization in tournament
def test_invalid_agent_in_tournament():
    args = [
        "tournament",
        "--agents",
        "NonExistentAgent()",
        "RandomAgent()",
        "--num_games",
        "50",
    ]

    with patch.object(sys, "argv", ["main.py"] + args), patch(
        "builtins.print"
    ) as mock_print:
        main()

    mock_print.assert_any_call(
        "Error: Failed to initialize agents. Invalid input: Class 'NonExistentAgent' is not defined."
    )
