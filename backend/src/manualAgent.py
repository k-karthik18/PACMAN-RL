# manualAgent.py - Pac-Man agent controlled by keyboard input from the web client

from game import Agent
from game import Directions


class ManualAgent(Agent):
    """Agent that gets actions from a queue (filled by WebSocket when user presses keys)."""

    def __init__(self, action_queue):
        super().__init__()
        self._action_queue = action_queue

    def getAction(self, state):
        legal = state.getLegalPacmanActions()
        if not legal:
            return Directions.STOP
        while True:
            action = self._action_queue.get()
            if action in legal:
                return action
