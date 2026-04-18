import random
import pickle
import os
from game import Agent
from util import Counter
from util import manhattanDistance
from feature_extraction import getCompetitionFeatures

TRACE_QUEUE = None

def set_trace_queue(q):
    global TRACE_QUEUE
    TRACE_QUEUE = q

# Key feature names tracked in per-episode logs
_FEATURE_KEYS = [
    "bias", "closestFood", "ghostDist", "danger", "nearGhost",
    "scaredGhostDist", "canEatGhost", "capsuleDist", "mobility", "stopped", "reverse"
]

class ApproxSarsaAgent(Agent):

    def __init__(self, alpha=0.2, gamma=0.9, epsilon=0.5):
        super().__init__()

        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)

        self.weights = Counter()

        # Load existing weights if they exist
        if os.path.exists("data/approx_sarsa_weights.pkl"):
            try:
                with open("data/approx_sarsa_weights.pkl", "rb") as f:
                    saved_weights = pickle.load(f)
                    self.weights.update(saved_weights)
                    print(f"Loaded existing weights for ApproxSarsaAgent: {self.weights}")
            except Exception as e:
                print(f"Error loading weights: {e}")

        self.prevState = None
        self.prevAction = None
        self.episode = 0

        # Step-level feature accumulation for per-episode logging
        self._step_features = Counter()
        self._step_count = 0

        if not os.path.exists("data"):
            os.makedirs("data")

        # Episode score/win/epsilon log
        if not os.path.exists("data/approx_sarsa_scores.csv"):
            with open("data/approx_sarsa_scores.csv", "w") as f:
                f.write("episode,score,win,epsilon\n")

        # Weight evolution log (one row per episode)
        if not os.path.exists("data/approx_sarsa_weights.csv"):
            with open("data/approx_sarsa_weights.csv", "w") as f:
                f.write(
                    "Episode,"
                    "bias,closestFood,ghostDist,danger,nearGhost,"
                    "scaredGhostDist,canEatGhost,capsuleDist,mobility,stopped,reverse\n"
                )

        # Average feature values per episode log
        if not os.path.exists("data/approx_sarsa_features.csv"):
            with open("data/approx_sarsa_features.csv", "w") as f:
                f.write(
                    "episode,"
                    "bias,closestFood,ghostDist,danger,nearGhost,"
                    "scaredGhostDist,canEatGhost,capsuleDist,mobility,stopped,reverse\n"
                )

    def getFeatures(self, state, action):
        return getCompetitionFeatures(state, action)

    def getQValue(self, state, action):
        if action is None:
            return 0.0
        features = self.getFeatures(state, action)
        return self.weights * features

    def chooseAction(self, state):
        legalActions = state.getLegalPacmanActions()
        if not legalActions:
            return None

        if random.random() < self.epsilon:
            return random.choice(legalActions)

        qvals = [(self.getQValue(state, a), a) for a in legalActions]
        maxQ = max(qvals)[0]
        best = [a for q, a in qvals if q == maxQ]
        return random.choice(best)

    def getAction(self, state):
        # SARSA: choose next action A' first, then update using (s, a, r, s', a')
        action = self.chooseAction(state)

        if self.prevState is not None:
            reward = state.getScore() - self.prevState.getScore()

            # On-policy TD target: R + γ·Q(s', a') — uses the ACTUAL chosen action
            tdError = (reward
                       + self.gamma * self.getQValue(state, action)
                       - self.getQValue(self.prevState, self.prevAction))

            features = self.getFeatures(self.prevState, self.prevAction)

            for f in features:
                self.weights[f] += self.alpha * tdError * features[f]

            # Accumulate feature values for episode-level logging
            for f in features:
                self._step_features[f] += features[f]
            self._step_count += 1

            if TRACE_QUEUE is not None:
                try:
                    TRACE_QUEUE.put({
                        "type": "trace",
                        "agent": "ApproxSarsaAgent",
                        "event": "update",
                        "action": self.prevAction,
                        "reward": reward,
                        "td_error": tdError,
                        "features": dict(features),
                        "weights_after": dict(self.weights),
                    })
                except Exception:
                    pass

        if TRACE_QUEUE is not None:
            try:
                legalActions = state.getLegalPacmanActions()
                qvals = {a: self.getQValue(state, a) for a in legalActions}
                TRACE_QUEUE.put({
                    "type": "trace",
                    "agent": "ApproxSarsaAgent",
                    "event": "select_action",
                    "epsilon": self.epsilon,
                    "legal_actions": list(legalActions),
                    "q_values": qvals,
                    "selected": action,
                })
            except Exception:
                pass

        self.prevState = state
        self.prevAction = action

        return action

    def final(self, state):
        """
        Terminal state update.
        SARSA terminal: δ = R + γ·Q(s', a') - Q(s, a)
        Since s' is terminal, Q(s', a') = 0 by definition, so:
            δ = R + 0 - Q(s, a) = R - Q(s, a)
        """
        reward = state.getScore() - self.prevState.getScore()

        tdError = reward - self.getQValue(self.prevState, self.prevAction)

        features = self.getFeatures(self.prevState, self.prevAction)

        for f in features:
            self.weights[f] += self.alpha * tdError * features[f]

        # Accumulate the terminal step features too
        for f in features:
            self._step_features[f] += features[f]
        self._step_count += 1

        self.episode += 1
        score = state.getScore()
        win = 1 if state.isWin() else 0

        # Save weights
        with open("data/approx_sarsa_weights.pkl", "wb") as f:
            pickle.dump(dict(self.weights), f)

        # Log weight evolution
        with open("data/approx_sarsa_weights.csv", "a") as f:
            f.write(
                f"{self.episode},"
                f"{self.weights.get('bias', 0)},"
                f"{self.weights.get('closestFood', 0)},"
                f"{self.weights.get('ghostDist', 0)},"
                f"{self.weights.get('danger', 0)},"
                f"{self.weights.get('nearGhost', 0)},"
                f"{self.weights.get('scaredGhostDist', 0)},"
                f"{self.weights.get('canEatGhost', 0)},"
                f"{self.weights.get('capsuleDist', 0)},"
                f"{self.weights.get('mobility', 0)},"
                f"{self.weights.get('stopped', 0)},"
                f"{self.weights.get('reverse', 0)}\n"
            )

        # Log score / win / epsilon per episode
        with open("data/approx_sarsa_scores.csv", "a") as f:
            f.write(f"{self.episode},{score},{win},{self.epsilon:.4f}\n")

        # Log average feature values per episode
        n = max(self._step_count, 1)
        with open("data/approx_sarsa_features.csv", "a") as f:
            vals = ",".join(
                f"{self._step_features.get(k, 0) / n:.5f}"
                for k in _FEATURE_KEYS
            )
            f.write(f"{self.episode},{vals}\n")

        # Reset step-level accumulators
        self._step_features = Counter()
        self._step_count = 0

        # Reset transition memory (avoid cross-episode contamination)
        self.prevState = None
        self.prevAction = None

        # Epsilon decay: faster 0.997 — only while learning
        if self.alpha > 0:
            self.epsilon = max(0.05, self.epsilon * 0.997)

        print(f"[ApproxSARSA] Ep {self.episode:>4} | score={score:>6.1f} | win={bool(win)} | ε={self.epsilon:.3f}")