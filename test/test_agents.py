import pytest
import random
from src.agents import *

AGENTS = [agent for agent in BASIC_AGENTS if not isinstance(agent, RandomAgent)]


@pytest.fixture
def seed_random():
    """Fixture to ensure deterministic random behavior during testing."""
    random.seed(42)


@pytest.mark.parametrize("agent", list(AGENTS))
def test_agent_reset(agent):
    """Test that all agents reset properly to their initial state."""
    initial_action = agent.choose_action()
    agent.update(1)  # Update with opponent's defection
    agent.reset()
    assert agent.choose_action() == initial_action


@pytest.mark.parametrize("agent", list(AGENTS))
def test_agent_reacts_to_opponent_defection(agent):
    """Test that all agents react to an opponent's defection in a valid way."""
    agent.reset()
    initial_action = agent.choose_action()
    agent.update(1)  # Opponent defects
    new_action = agent.choose_action()
    assert new_action in {0, 1}
    # Ensure agents behave differently after an opponent's defection if applicable
    if not isinstance(
        agent,
        (
            AlwaysCooperateAgent,
            AlwaysDefectAgent,
            TitForTwoTatsAgent,
            ProvocativeAgent,
            SuspiciousPavlovAgent,
            WinStayLoseShiftAgent,
            PavlovAgent,
            GenerousTitForTatAgent,
        ),
    ):
        assert new_action != initial_action or new_action == 1


@pytest.mark.parametrize("agent", list(AGENTS))
def test_agent_reacts_to_opponent_cooperation(agent):
    """Test that all agents react to an opponent's cooperation in a valid way."""
    agent.reset()
    initial_action = agent.choose_action()
    agent.update(0)  # Opponent cooperates
    new_action = agent.choose_action()
    assert new_action in {0, 1}
    # Check for valid cooperation reactions
    if isinstance(agent, AlwaysDefectAgent):
        assert new_action == 1  # Always defects
    elif not isinstance(agent, AlwaysCooperateAgent):
        assert new_action == 0 or new_action == initial_action


@pytest.mark.parametrize("agent", list(AGENTS))
def test_agent_deterministic_behavior(agent):
    """
    Test deterministic agents for consistent actions.
    """
    if isinstance(agent, (SuspiciousGradualAgent, SuspiciousGenerousTitForTatAgent, GenerousTitForTatAgent)):
        return  # Skip this test for RandomAgent

    agent.reset()
    actions = [agent.choose_action() for _ in range(5)]
    assert len(set(actions)) == 1  # All actions should be the same


@pytest.mark.parametrize("agent", list(AGENTS))
def test_agent_random_behavior(seed_random, agent):
    """
    Test probabilistic agents for varying actions.
    Only applies to RandomAgent and agents with probabilistic forgiveness.
    """
    if isinstance(agent, (GenerousTitForTatAgent, SuspiciousGenerousTitForTatAgent)):
        agent.reset()
        agent.update(1)
        agent.history.append(1)
        actions = [agent.choose_action() for _ in range(1000)]
        assert len(set(actions)) > 1  # Actions should vary


@pytest.mark.parametrize("agent", list(AGENTS))
def test_agent_update_history(agent):
    """Test that agents update their history and reward history correctly."""
    agent.reset()
    opponent_actions = [0, 1, 0, 1]
    rewards = [2, 0, 3, 1]

    for opponent_action, reward in zip(opponent_actions, rewards):
        agent.history.append(agent.choose_action())
        agent.update(opponent_action)
        agent.reward_history.append(reward)

    assert len(agent.history) == len(opponent_actions)
    assert len(agent.reward_history) == len(rewards)
