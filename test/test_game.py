import pytest
from src.game import Game
from src.agents import RandomAgent, AlwaysCooperateAgent, AlwaysDefectAgent


class MockAgent:
    """A mock agent for testing with deterministic actions."""

    def __init__(self, actions):
        self.actions = actions  # Predetermined actions
        self.history = []
        self.reward_history = []
        self.index = 0

    def choose_action(self):
        action = self.actions[self.index % len(self.actions)]
        self.index += 1
        return action

    def update(self, opponent_action):
        pass


def test_play_round_both_cooperate():
    agent_a = MockAgent(actions=[0])  # Always cooperate
    agent_b = MockAgent(actions=[0])  # Always cooperate
    game = Game(agent_a, agent_b, verbose=0)

    action_a, action_b, reward_a, reward_b = game.play_round()

    assert action_a == 0
    assert action_b == 0
    assert reward_a == 3
    assert reward_b == 3
    assert agent_a.history == [0]
    assert agent_b.history == [0]
    assert agent_a.reward_history == [3]
    assert agent_b.reward_history == [3]


def test_play_round_both_defect():
    agent_a = MockAgent(actions=[1])  # Always defect
    agent_b = MockAgent(actions=[1])  # Always defect
    game = Game(agent_a, agent_b, verbose=0)

    action_a, action_b, reward_a, reward_b = game.play_round()

    assert action_a == 1
    assert action_b == 1
    assert reward_a == 1
    assert reward_b == 1
    assert agent_a.history == [1]
    assert agent_b.history == [1]
    assert agent_a.reward_history == [1]
    assert agent_b.reward_history == [1]


def test_play_round_cooperate_vs_defect():
    agent_a = MockAgent(actions=[0])  # Cooperates
    agent_b = MockAgent(actions=[1])  # Defects
    game = Game(agent_a, agent_b, verbose=0)

    action_a, action_b, reward_a, reward_b = game.play_round()

    assert action_a == 0
    assert action_b == 1
    assert reward_a == 0
    assert reward_b == 5
    assert agent_a.history == [0]
    assert agent_b.history == [1]
    assert agent_a.reward_history == [0]
    assert agent_b.reward_history == [5]


def test_play_iterated_game():
    agent_a = MockAgent(actions=[0, 1])  # Alternates: cooperate, defect
    agent_b = MockAgent(actions=[1, 0])  # Alternates: defect, cooperate
    game = Game(agent_a, agent_b, verbose=0)

    total_reward_a, total_reward_b = game.play_iterated_game(num_rounds=4)

    assert agent_a.history == [0, 1, 0, 1]
    assert agent_b.history == [1, 0, 1, 0]
    assert agent_a.reward_history == [0, 5, 0, 5]
    assert agent_b.reward_history == [5, 0, 5, 0]
    assert total_reward_a == 10
    assert total_reward_b == 10


def test_verbose_output(capsys):
    agent_a = MockAgent(actions=[0, 1])  # Alternates: cooperate, defect
    agent_b = MockAgent(actions=[1, 0])  # Alternates: defect, cooperate
    game = Game(agent_a, agent_b, verbose=2)

    game.play_iterated_game(num_rounds=2)
    captured = capsys.readouterr()

    assert "MockAgent: 0, MockAgent: 1 --> ((0, 5))" in captured.out
    assert "MockAgent: 1, MockAgent: 0 --> ((5, 0))" in captured.out


def test_always_cooperate_vs_always_defect():
    agent_a = AlwaysCooperateAgent()
    agent_b = AlwaysDefectAgent()
    game = Game(agent_a, agent_b, verbose=0)

    total_reward_a, total_reward_b = game.play_iterated_game(num_rounds=5)

    assert agent_a.history == [0, 0, 0, 0, 0]  # Always cooperates
    assert agent_b.history == [1, 1, 1, 1, 1]  # Always defects
    assert agent_a.reward_history == [0, 0, 0, 0, 0]  # Cooperator gets no rewards
    assert agent_b.reward_history == [5, 5, 5, 5, 5]  # Defector gets full rewards
    assert total_reward_a == 0
    assert total_reward_b == 25


def test_random_vs_random():
    agent_a = RandomAgent()
    agent_b = RandomAgent()
    game = Game(agent_a, agent_b, verbose=0)

    total_reward_a, total_reward_b = game.play_iterated_game(num_rounds=10)

    assert len(agent_a.history) == 10
    assert len(agent_b.history) == 10
    assert len(agent_a.reward_history) == 10
    assert len(agent_b.reward_history) == 10


def test_edge_case_no_rounds():
    agent_a = RandomAgent()
    agent_b = RandomAgent()
    game = Game(agent_a, agent_b, verbose=0)

    total_reward_a, total_reward_b = game.play_iterated_game(num_rounds=0)

    assert total_reward_a == 0
    assert total_reward_b == 0
    assert agent_a.history == []
    assert agent_b.history == []
    assert agent_a.reward_history == []
    assert agent_b.reward_history == []
