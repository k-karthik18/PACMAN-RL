# webDisplay.py - Display that pushes game state to a queue for WebSocket streaming

def serialize_state(data):
    """Serialize GameStateData to a JSON-serializable dict. data is state.data (GameStateData)."""
    layout = data.layout
    walls = []
    for x in range(layout.width):
        for y in range(layout.height):
            if layout.walls[x][y]:
                walls.append([x, y])
    food = list(data.food.asList())
    capsules = [list(c) for c in data.capsules]
    agents = []
    for a in data.agentStates:
        if a.configuration is None:
            agents.append({"pos": None, "direction": None, "isPacman": a.isPacman, "scaredTimer": a.scaredTimer})
        else:
            agents.append({
                "pos": list(a.configuration.getPosition()),
                "direction": a.configuration.getDirection(),
                "isPacman": a.isPacman,
                "scaredTimer": a.scaredTimer,
            })
    return {
        "walls": walls,
        "width": layout.width,
        "height": layout.height,
        "food": [list(p) for p in food],
        "capsules": capsules,
        "agents": agents,
        "score": data.score,
        "win": getattr(data, "_win", False),
        "lose": getattr(data, "_lose", False),
    }


class WebDisplay:
    """Display that sends serialized state to a queue for streaming to the frontend."""

    def __init__(self, queue, frame_delay=0.05):
        import time as time_module
        self._queue = queue
        self._frame_delay = frame_delay
        self._time_module = time_module

    def initialize(self, state, isBlue=False):
        self._queue.put({"type": "init", "state": serialize_state(state)})

    def update(self, state):
        self._queue.put({"type": "frame", "state": serialize_state(state)})
        if self._frame_delay > 0:
            self._time_module.sleep(self._frame_delay)

    def checkNullDisplay(self):
        return False

    def pause(self):
        if self._frame_delay > 0:
            self._time_module.sleep(self._frame_delay)

    def draw(self, state):
        pass

    def updateDistributions(self, dist):
        pass

    def finish(self):
        pass  # episode_end is sent by the backend after each game with score
