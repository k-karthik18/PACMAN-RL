import random
from game import Agent

class RandomAgent(Agent):
    
    def __init__(self):
        super().__init__()

    def getAction(self, state):
        """
        state: GameState object
        returns: one legal action
        """
        
        # Get legal actions
        legalActions = state.getLegalPacmanActions()
        
        # Remove STOP to make it more interesting (optional)
        if "Stop" in legalActions:
            legalActions.remove("Stop")

        # Choose random action
        action = random.choice(legalActions)
        
        return action