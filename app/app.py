from flask import Flask, render_template, request, jsonify
from src.agents import DeepQLearningAgent, Agent
from src.game import Game

app = Flask(__name__)


class WebUser(Agent):
    def __init__(self):
        super().__init__()
        self.next_action = None

    def set_next_action(self, action):
        self.next_action = action

    def choose_action(self):
        return self.next_action


# Initialize agents and game
deep_q_agent = DeepQLearningAgent(state_size=20, path="deepq_result.pt")
web_user = WebUser()
game = Game(deep_q_agent, web_user)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start_game", methods=["POST"])
def start_game():
    data = request.json
    num_rounds = data["rounds"]
    # game.agent_a = DeepQLearningAgent(state_size=20, path="deepq_result.pt")
    game.agent_a.reset()
    game.agent_a.prev_actions =  [1, 0] * 5 +[0] * 10 
    game.agent_b.reset()
    game_state = {
        "rounds": num_rounds,
        "current_round": 0,
        "user_score": 0,
        "agent_score": 0,
        "history": [],
    }
    return jsonify({"message": "Game started!", "game_state": game_state})


@app.route("/play_round", methods=["POST"])
def play_round():
    data = request.json
    user_action = data["action"]  # 0 for cooperate, 1 for defect
    web_user.set_next_action(user_action)

    # Play a round
    action_a, action_b, reward_a, reward_b = game.play_round()
    game.agent_a.update(action_b)
    game.agent_b.update(action_a)

    # Update scores
    user_score = sum(web_user.reward_history)
    agent_score = sum(deep_q_agent.reward_history)

    # Append history
    history = [
        {
            "round": len(web_user.history),
            "user_action": action_b,
            "agent_action": action_a,
        }
    ]

    # Check if the game is over
    game_over = len(web_user.history) >= data["rounds"]

    return jsonify(
        {
            "round": len(web_user.history),
            "user_action": action_b,
            "agent_action": action_a,
            "user_score": user_score,
            "agent_score": agent_score,
            "game_over": game_over,
            "history": history,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
