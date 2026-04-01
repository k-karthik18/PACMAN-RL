from pacman import runGames
from layout import getLayout
from ghostAgents import RandomGhost
from approxQLearningAgents import ApproxQLearningAgent
import textDisplay

layout = getLayout("smallClassic")

agent = ApproxQLearningAgent(
    alpha=0.05,
    epsilon=0.3,
    gamma=0.9
)

ghosts = [RandomGhost(1)]

runGames(
    layout=layout,
    pacman=agent,
    ghosts=ghosts,
    display=textDisplay.NullGraphics(),
    numGames=2000,
    record=False,
    numTraining=0,
    catchExceptions=False,
    timeout=30
)