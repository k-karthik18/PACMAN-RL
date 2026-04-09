import random

from game import Agent
from util import manhattanDistance


class TestAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__()

    def getAction(self, state):
        legal = state.getLegalPacmanActions()
        if not legal:
            return None

        candidates = [a for a in legal if a != "Stop"] or list(legal)

        best_score = None
        best_actions = []

        for action in candidates:
            successor = state.generatePacmanSuccessor(action)
            if successor is None:
                continue

            score = successor.getScore()
            pac_pos = successor.getPacmanPosition()

            ghost_states = successor.getGhostStates()
            active_ghosts = [g for g in ghost_states if getattr(g, "scaredTimer", 0) <= 0]
            scared_ghosts = [g for g in ghost_states if getattr(g, "scaredTimer", 0) > 0]

            if active_ghosts:
                dists = [manhattanDistance(pac_pos, g.getPosition()) for g in active_ghosts]
                min_d = min(dists)
                if min_d <= 1:
                    score -= 10000
                elif min_d == 2:
                    score -= 500
                else:
                    score -= 20.0 / (min_d + 1)

            food_list = successor.getFood().asList()
            if food_list:
                food_d = min(manhattanDistance(pac_pos, f) for f in food_list)
                score += 10.0 / (food_d + 1)

            capsules = successor.getCapsules()
            if capsules:
                cap_d = min(manhattanDistance(pac_pos, c) for c in capsules)
                if active_ghosts and cap_d <= 5:
                    score += 6.0 / (cap_d + 1)

            if scared_ghosts:
                sd = [manhattanDistance(pac_pos, g.getPosition()) for g in scared_ghosts]
                min_sd = min(sd)
                score += 8.0 / (min_sd + 1)

            if action == "Stop":
                score -= 2

            if best_score is None or score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)

        return random.choice(best_actions) if best_actions else random.choice(legal)
