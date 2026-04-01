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

class ApproxQLearningAgent(Agent):

    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2):
        super().__init__()

        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)

        self.weights = Counter()
        
        # Load existing weights if they exist
        if os.path.exists("data/approx_weights.pkl"):
            try:
                with open("data/approx_weights.pkl", "rb") as f:
                    saved_weights = pickle.load(f)
                    self.weights.update(saved_weights)
                    print(f"Loaded existing weights for ApproxQLearningAgent: {self.weights}")
            except Exception as e:
                print(f"Error loading weights: {e}")

        self.prevState = None
        self.prevAction = None
        self.episode = 0

        if not os.path.exists("data"):
            os.makedirs("data")

        if not os.path.exists("data/approx_weights.csv"):
            with open("data/approx_weights.csv","w") as f:
                f.write("Episode,bias,closestFood,nearbyFood,ghostDist,nearGhost,danger,scaredGhostDist,canEatGhost,capsuleDist,mobility,stopped,score\n")

    # ----------------------------
    # FEATURE FUNCTION
    # ----------------------------
    def getFeatures(self,state,action):
        return getCompetitionFeatures(state, action)

    # ----------------------------
    def getQValue(self,state,action):
        features = self.getFeatures(state,action)
        return self.weights * features

    # ----------------------------
    def getValue(self,state):

        legalActions = state.getLegalPacmanActions()
        if not legalActions:
            return 0.0

        return max(self.getQValue(state,a) for a in legalActions)

    # ----------------------------
    def getPolicy(self,state):

        legalActions = state.getLegalPacmanActions()
        if not legalActions:
            return None

        bestValue = self.getValue(state)

        bestActions = [a for a in legalActions
                       if self.getQValue(state,a) == bestValue]

        return random.choice(bestActions)

    # ----------------------------
    def update(self,state,action,nextState,reward):

        features = self.getFeatures(state,action)

        q_sa = self.getQValue(state,action)
        v_next = self.getValue(nextState)

        tdError = reward + self.gamma*v_next - q_sa

        prev_weights = dict(self.weights)
        for f in features:
            self.weights[f] += self.alpha * tdError * features[f]

        if TRACE_QUEUE is not None:
            try:
                TRACE_QUEUE.put({
                    "type": "trace",
                    "agent": "ApproxQLearningAgent",
                    "event": "update",
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
    def getAction(self,state):

        legalActions = state.getLegalPacmanActions()
        if not legalActions:
            return None

        if self.prevState is not None:

            reward = state.getScore() - self.prevState.getScore()

            self.update(self.prevState,self.prevAction,state,reward)

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
    def final(self,state):

        reward = state.getScore() - self.prevState.getScore()

        self.update(self.prevState,self.prevAction,state,reward)

        self.episode += 1

        with open("data/approx_weights.pkl","wb") as f:
            pickle.dump(dict(self.weights),f)

        with open("data/approx_weights.csv","a") as f:
            f.write(f"{self.episode},"
                    f"{self.weights.get('bias', 0)},"
                    f"{self.weights.get('closestFood', 0)},"
                    f"{self.weights.get('nearbyFood', 0)},"
                    f"{self.weights.get('ghostDist', 0)},"
                    f"{self.weights.get('nearGhost', 0)},"
                    f"{self.weights.get('danger', 0)},"
                    f"{self.weights.get('scaredGhostDist', 0)},"
                    f"{self.weights.get('canEatGhost', 0)},"
                    f"{self.weights.get('capsuleDist', 0)},"
                    f"{self.weights.get('mobility', 0)},"
                    f"{self.weights.get('stopped', 0)},"
                    f"{self.weights.get('score', 0)}\n")

        self.epsilon = max(0.01,self.epsilon*0.995)

        print(f"Episode {self.episode} done | epsilon {self.epsilon:.3f}")