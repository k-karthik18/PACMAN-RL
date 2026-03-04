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

    def __init__(self, alpha=0.2, gamma=0.8, epsilon=0.1):
        super().__init__()

        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)

        self.qValues = Counter()

        self.prevState = None
        self.prevAction = None

        # Create folder for storage
        if not os.path.exists("data"):
            os.makedirs("data")

    def getQValue(self, state, action):
        return self.qValues[(state, action)]

    def getValue(self, state):
        legalActions = state.getLegalPacmanActions()
        if len(legalActions) == 0:
            return 0.0
        return max([self.getQValue(state, a) for a in legalActions])

    def getPolicy(self, state):
        legalActions = state.getLegalPacmanActions()
        if len(legalActions) == 0:
            return None

        bestValue = self.getValue(state)
        bestActions = [a for a in legalActions
                       if self.getQValue(state, a) == bestValue]

        return random.choice(bestActions)

    def update(self, state, action, nextState, reward):
        sample = reward + self.gamma * self.getValue(nextState)
        oldValue = self.getQValue(state, action)
        newValue = oldValue + self.alpha * (sample - oldValue)

        self.qValues[(state, action)] = newValue

        if TRACE_QUEUE is not None:
            try:
                TRACE_QUEUE.put({
                    "type": "trace",
                    "agent": "QLearningAgent",
                    "event": "update",
                    "action": action,
                    "reward": reward,
                    "sample": sample,
                    "old_value": oldValue,
                    "new_value": newValue,
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