import random
import pickle
import os
from game import Agent
from util import Counter
from util import manhattanDistance
from feature_extraction import getSimpleFeatures

TRACE_QUEUE = None

def set_trace_queue(q):
    global TRACE_QUEUE
    TRACE_QUEUE = q

class ApproxSarsaAgent(Agent):

    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2):
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

        if not os.path.exists("data"):
            os.makedirs("data")
            
        if not os.path.exists("data/approx_sarsa_weights.csv"):
            with open("data/approx_sarsa_weights.csv","w") as f:
                f.write(
                    "Episode,"
                    "bias,closestFood,closestGhost,danger,"
                    "closestScared,eatGhost,closestCapsule,stopped\n"
                )

    def getFeatures(self,state,action):
        return getSimpleFeatures(state, action)

    def getQValue(self,state,action):

        if action is None:
            return 0

        features = self.getFeatures(state,action)

        return self.weights * features

    def chooseAction(self,state):

        legalActions = state.getLegalPacmanActions()

        if not legalActions:
            return None

        if random.random() < self.epsilon:
            return random.choice(legalActions)

        qvals = [(self.getQValue(state,a),a) for a in legalActions]

        maxQ = max(qvals)[0]

        best = [a for q,a in qvals if q == maxQ]

        return random.choice(best)

    def getAction(self,state):

        action = self.chooseAction(state)

        if self.prevState is not None:

            reward = state.getScore() - self.prevState.getScore()

            tdError = reward + self.gamma*self.getQValue(state,action) - \
                      self.getQValue(self.prevState,self.prevAction)

            features = self.getFeatures(self.prevState,self.prevAction)

            for f in features:
                self.weights[f] += self.alpha * tdError * features[f]
            
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

    def final(self,state):

        reward = state.getScore() - self.prevState.getScore()

        tdError = reward - self.getQValue(self.prevState,self.prevAction)

        features = self.getFeatures(self.prevState,self.prevAction)

        for f in features:
            self.weights[f] += self.alpha * tdError * features[f]

        self.episode += 1

        with open("data/approx_sarsa_weights.pkl","wb") as f:
            pickle.dump(dict(self.weights),f)
            
        with open("data/approx_sarsa_weights.csv","a") as f:
            f.write(f"{self.episode},"
                    f"{self.weights.get('bias', 0)},"
                    f"{self.weights.get('closestFood', 0)},"
                    f"{self.weights.get('closestGhost', 0)},"
                    f"{self.weights.get('danger', 0)},"
                    f"{self.weights.get('closestScared', 0)},"
                    f"{self.weights.get('eatGhost', 0)},"
                    f"{self.weights.get('closestCapsule', 0)},"
                    f"{self.weights.get('stopped', 0)}\n")

        # Slower decay helps mediumClassic (needs longer exploration)
        # Only decay while learning. For evaluation runs (alpha=0), keep epsilon as provided.
        if self.alpha > 0:
            self.epsilon = max(0.05, self.epsilon * 0.999)

        print(f"SARSA Episode {self.episode} done | epsilon {self.epsilon:.3f}")