import random
import pickle
import os
from game import Agent
from util import Counter

TRACE_QUEUE = None

def set_trace_queue(q):
    global TRACE_QUEUE
    TRACE_QUEUE = q

class QLearningAgent(Agent):

    def __init__(self, alpha=0.1, gamma=0.8, epsilon=0.1, numTraining=0):
        super().__init__()

        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.numTraining = int(numTraining)
        self.qValues = Counter()

        # Load existing Q-values if they exist
        if os.path.exists("data/q_values.pkl"):
            try:
                with open("data/q_values.pkl", "rb") as f:
                    saved_q = pickle.load(f)
                    self.qValues.update(saved_q)
                    print(f"Loaded existing Q-values for QLearningAgent: {len(self.qValues)} states")
            except Exception as e:
                print(f"Error loading Q-values: {e}")

        self.prevState = None
        self.prevAction = None

        # Create folder for storage
        if not os.path.exists("data"):
            os.makedirs("data")

    def getQValue(self, state, action):
        return self.qValues[(state, action)]

    def computeValueFromQValues(self, state):
        legalActions = state.getLegalPacmanActions()
        if not legalActions:
            return 0.0
        return max(self.getQValue(state, action) for action in legalActions)

    def computeActionFromQValues(self, state):
        legalActions = state.getLegalPacmanActions()
        if not legalActions:
            return None
        maxVal = self.computeValueFromQValues(state)
        bestActions = [a for a in legalActions if self.getQValue(state, a) == maxVal]
        return random.choice(bestActions)

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

    def update(self, state, action, nextState, reward):
        q_sa = self.getQValue(state, action)
        v_next = self.getValue(nextState)
        new_q = q_sa + self.alpha * (reward + self.gamma * v_next - q_sa)
        self.qValues[(state, action)] = new_q
        
        if TRACE_QUEUE is not None:
            try:
                TRACE_QUEUE.put({
                    "type": "trace",
                    "agent": "QLearningAgent",
                    "event": "update",
                    "action": action,
                    "reward": reward,
                    "old_value": q_sa,
                    "new_value": new_q,
                    "sample": reward + self.gamma * v_next
                })
            except Exception:
                pass


    def getAction(self, state):

        legalActions = state.getLegalPacmanActions()
        if len(legalActions) == 0:
            return None

        # If not first move, update using previous state
        if self.prevState is not None:
            reward = state.getScore() - self.prevState.getScore()
            self.update(self.prevState, self.prevAction, state, reward)

        # ε-greedy selection
        if random.random() < self.epsilon:
            action = random.choice(legalActions)
        else:
            action = self.getPolicy(state)

        if TRACE_QUEUE is not None:
            try:
                qvals = {a: self.getQValue(state, a) for a in legalActions}
                TRACE_QUEUE.put({
                    "type": "trace",
                    "agent": "QLearningAgent",
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
        Called at end of game
        Save Q-table
        """
        reward = state.getScore() - self.prevState.getScore()
        self.update(self.prevState, self.prevAction, state, reward)

        # Save Q-table
        with open("data/qvalues.pkl", "wb") as f:
            pickle.dump(dict(self.qValues), f)

        print("Q-table saved to data/qvalues.pkl")