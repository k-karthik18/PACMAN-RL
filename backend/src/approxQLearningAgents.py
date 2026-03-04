import random
import pickle
import os
from game import Agent
from util import Counter
from util import manhattanDistance

TRACE_QUEUE = None

def set_trace_queue(q):
    global TRACE_QUEUE
    TRACE_QUEUE = q

class ApproxQLearningAgent(Agent):

    def __init__(self, alpha=0.05, gamma=0.9, epsilon=0.3):
        super().__init__()

        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)

        self.weights = Counter()

        self.prevState = None
        self.prevAction = None

        self.episode = 0

        if not os.path.exists("data"):
            os.makedirs("data")

        # Create CSV file if not exists
        if not os.path.exists("data/approx_weights.csv"):
            with open("data/approx_weights.csv", "w") as f:
                f.write("Episode,bias,closestFood,closestGhost,foodCount,danger\n")

    # ----------------------------
    # FEATURE FUNCTION
    # ----------------------------
    def getFeatures(self, state, action):

        features = Counter()

        nextState = state.generatePacmanSuccessor(action)

        pacPos = nextState.getPacmanPosition()
        foodList = nextState.getFood().asList()
        ghostPositions = nextState.getGhostPositions()

        features["bias"] = 1.0

        # Closest food
        if len(foodList) > 0:
            minFoodDist = min([manhattanDistance(pacPos, food) for food in foodList])
            features["closestFood"] = 1.0 / (minFoodDist + 1.0)

        # Closest ghost
        if len(ghostPositions) > 0:
            minGhostDist = min([manhattanDistance(pacPos, ghost) for ghost in ghostPositions])
            features["closestGhost"] = 1.0 / (minGhostDist + 1.0)

            # Danger feature (very important)
            if minGhostDist <= 1:
                features["danger"] = 1.0
            else:
                features["danger"] = 0.0

        # Remaining food
        features["foodCount"] = nextState.getNumFood() / 100.0

        return features

    # ----------------------------
    def getQValue(self, state, action):
        features = self.getFeatures(state, action)
        return self.weights * features

    # ----------------------------
    def getValue(self, state):
        legalActions = state.getLegalPacmanActions()
        if len(legalActions) == 0:
            return 0.0
        return max([self.getQValue(state, a) for a in legalActions])

    # ----------------------------
    def getPolicy(self, state):
        legalActions = state.getLegalPacmanActions()
        if len(legalActions) == 0:
            return None

        bestValue = self.getValue(state)
        bestActions = [a for a in legalActions
                       if self.getQValue(state, a) == bestValue]

        return random.choice(bestActions)

    # ----------------------------
    def update(self, state, action, nextState, reward):

        features = self.getFeatures(state, action)

        q_sa = self.getQValue(state, action)
        v_next = self.getValue(nextState)

        tdError = reward + self.gamma * v_next - q_sa

        prev_weights = dict(self.weights)

        for f in features:
            self.weights[f] += self.alpha * tdError * features[f]

        if TRACE_QUEUE is not None:
            try:
                TRACE_QUEUE.put({
                    "type": "trace",
                    "agent": "ApproxQLearningAgent",
                    "action": action,
                    "reward": reward,
                    "q_sa": q_sa,
                    "v_next": v_next,
                    "td_error": tdError,
                    "features": dict(features),
                    "weights_before": prev_weights,
                    "weights_after": dict(self.weights),
                })
            except Exception:
                pass

    # ----------------------------
    def getAction(self, state):

        legalActions = state.getLegalPacmanActions()
        if len(legalActions) == 0:
            return None

        if self.prevState is not None:
            reward = state.getScore() - self.prevState.getScore()
            self.update(self.prevState, self.prevAction, state, reward)

        # ε-greedy
        if random.random() < self.epsilon:
            action = random.choice(legalActions)
        else:
            action = self.getPolicy(state)

        if TRACE_QUEUE is not None:
            try:
                qvals = {a: self.getQValue(state, a) for a in legalActions}
                TRACE_QUEUE.put({
                    "type": "trace",
                    "agent": "ApproxQLearningAgent",
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

    # ----------------------------
    def final(self, state):

        reward = state.getScore() - self.prevState.getScore()
        self.update(self.prevState, self.prevAction, state, reward)

        self.episode += 1

        # Save pickle
        with open("data/approx_weights.pkl", "wb") as f:
            pickle.dump(dict(self.weights), f)

        # Append CSV (track evolution)
        with open("data/approx_weights.csv", "a") as f:
            f.write(f"{self.episode},"
                    f"{self.weights['bias']},"
                    f"{self.weights['closestFood']},"
                    f"{self.weights['closestGhost']},"
                    f"{self.weights['foodCount']},"
                    f"{self.weights['danger']}\n")

        # Epsilon decay (very important!)
        self.epsilon = max(0.01, self.epsilon * 0.995)

        print(f"Episode {self.episode} finished. Epsilon now {self.epsilon:.4f}")