import random
import pickle
import os
from game import Agent
from util import Counter
from util import manhattanDistance

class ApproxSarsaAgent(Agent):

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

        if not os.path.exists("data/approx_sarsa_weights.csv"):
            with open("data/approx_sarsa_weights.csv", "w") as f:
                f.write("Episode,bias,closestFood,closestGhost,foodCount,danger\n")

    # ----------------------------
    # FEATURES
    # ----------------------------
    def getFeatures(self, state, action):

        features = Counter()

        nextState = state.generatePacmanSuccessor(action)

        pacPos = nextState.getPacmanPosition()
        foodList = nextState.getFood().asList()
        ghostPositions = nextState.getGhostPositions()

        features["bias"] = 1.0

        if len(foodList) > 0:
            minFoodDist = min([manhattanDistance(pacPos, food) for food in foodList])
            features["closestFood"] = 1.0 / (minFoodDist + 1.0)

        if len(ghostPositions) > 0:
            minGhostDist = min([manhattanDistance(pacPos, ghost) for ghost in ghostPositions])
            features["closestGhost"] = 1.0 / (minGhostDist + 1.0)

            if minGhostDist <= 1:
                features["danger"] = 1.0
            else:
                features["danger"] = 0.0

        features["foodCount"] = nextState.getNumFood() / 100.0

        return features

    # ----------------------------
    def getQValue(self, state, action):
        if action is None:
            return 0.0
        features = self.getFeatures(state, action)
        return self.weights * features

    # ----------------------------
    def chooseAction(self, state):

        legalActions = state.getLegalPacmanActions()
        if len(legalActions) == 0:
            return None

        if random.random() < self.epsilon:
            return random.choice(legalActions)

        qValues = [(self.getQValue(state, a), a) for a in legalActions]
        maxQ = max(qValues)[0]
        bestActions = [a for q, a in qValues if q == maxQ]

        return random.choice(bestActions)

    # ----------------------------
    def getAction(self, state):

        action = self.chooseAction(state)

        if self.prevState is not None:

            reward = state.getScore() - self.prevState.getScore()

            tdError = reward + self.gamma * self.getQValue(state, action) \
                      - self.getQValue(self.prevState, self.prevAction)

            features = self.getFeatures(self.prevState, self.prevAction)

            for f in features:
                self.weights[f] += self.alpha * tdError * features[f]

        self.prevState = state
        self.prevAction = action

        return action

    # ----------------------------
    def final(self, state):

        reward = state.getScore() - self.prevState.getScore()

        tdError = reward - self.getQValue(self.prevState, self.prevAction)

        features = self.getFeatures(self.prevState, self.prevAction)

        for f in features:
            self.weights[f] += self.alpha * tdError * features[f]

        self.episode += 1

        # Save pickle
        with open("data/approx_sarsa_weights.pkl", "wb") as f:
            pickle.dump(dict(self.weights), f)

        # Append CSV
        with open("data/approx_sarsa_weights.csv", "a") as f:
            f.write(f"{self.episode},"
                    f"{self.weights['bias']},"
                    f"{self.weights['closestFood']},"
                    f"{self.weights['closestGhost']},"
                    f"{self.weights['foodCount']},"
                    f"{self.weights['danger']}\n")

        # Epsilon decay
        self.epsilon = max(0.01, self.epsilon * 0.995)

        print(f"Approx SARSA Episode {self.episode} done. Epsilon {self.epsilon:.4f}")