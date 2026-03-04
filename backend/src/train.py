# train.py
from pacman import runGames
from layout import getLayout
from ghostAgents import RandomGhost
import multiAgents
import textDisplay  # <-- Import textDisplay for NullGraphics

# Train for 200 episodes with no graphics
runGames(
    layout=getLayout("mediumClassic"),
    pacman=multiAgents.ApproximateQAgent(alpha=0.01, epsilon=0.1, gamma=0.9),
    ghosts=[RandomGhost(1), RandomGhost(2)],
    display=textDisplay.NullGraphics(),  # ✅ Proper null display
    numGames=1000,
    record=False,
    numTraining=0,
    catchExceptions=False,
    timeout=30
)