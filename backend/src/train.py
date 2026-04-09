"""
train.py

Dedicated training script for high-speed offline training of Pacman AI agents.
This script runs the game without graphics to maximize training speed.
Weights are automatically saved to the 'data' directory.

Usage:
    python train.py -p ApproxQLearningAgent -l mediumClassic -n 1500
    python train.py -p ApproxSarsaAgent -l smallClassic -n 1500
"""

import sys
import os
import time
import pacman
import layout
import ghostAgents
import textDisplay
from pacman import ClassicGameRules

def train_agent(
    agent_name,
    layout_name,
    num_episodes,
    num_ghosts=1,
    alpha=0.1,
    gamma=0.9,
    epsilon=0.2,
):

    print(f"\n{'='*60}")
    print(f"TRAINING START")
    print(f"Agent: {agent_name}")
    print(f"Layout: {layout_name}")
    print(f"Episodes: {num_episodes}")
    print(f"Ghosts: {num_ghosts}")
    print(f"Params: alpha={alpha}, gamma={gamma}, epsilon={epsilon}")
    print(f"{'='*60}\n")

    # Load layout
    lay = layout.getLayout(layout_name)
    if lay is None:
        print(f"Error: Layout '{layout_name}' not found.")
        return

    # Load agent class
    agent_type = pacman.loadAgent(agent_name, nographics=True)
    
    # Initialize agent with params
    # Note: Our agents now load existing weights automatically in __init__
    agent_instance = agent_type(alpha=alpha, gamma=gamma, epsilon=epsilon)

    ghosts = [ghostAgents.RandomGhost(i + 1) for i in range(int(num_ghosts))]
    
    # Use NullGraphics for maximum speed
    display = textDisplay.NullGraphics()
    rules = ClassicGameRules(timeout=30)
    
    start_time = time.time()
    
    wins = 0
    scores = []
    
    for i in range(num_episodes):
        # Every episode, the agent's 'final' method is called which saves weights
        game = rules.newGame(lay, agent_instance, ghosts, display, quiet=True)
        game.run()
        
        win = game.state.isWin()
        score = game.state.getScore()
        
        if win: wins += 1
        scores.append(score)
        
        if (i + 1) % 50 == 0:
            avg_win_rate = sum(1 for s in scores[-50:] if s > 0) / 50 * 100
            avg_score = sum(scores[-50:]) / 50
            print(f"Completed {i+1}/{num_episodes} episodes | Win Rate (last 50): {avg_win_rate:.1f}% | Avg Score: {avg_score:.1f}")

    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Overall Win Rate: {wins}/{num_episodes} ({wins/num_episodes*100:.1f}%)")
    print(f"Average Score: {sum(scores)/num_episodes:.1f}")
    print(f"Weights saved to backend/src/data/")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Pacman AI agents")
    parser.add_argument("-p", "--agent", type=str, default="ApproxQLearningAgent", help="Agent class name")
    parser.add_argument("-l", "--layout", type=str, default="mediumClassic", help="Layout name")
    parser.add_argument("-n", "--num_episodes", type=int, default=1500, help="Number of episodes")
    parser.add_argument("-k", "--num_ghosts", type=int, default=1, help="Number of ghosts")
    parser.add_argument("--alpha", type=float, default=0.2, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=0.5, help="Exploration rate")
    
    args = parser.parse_args()
    
    train_agent(
        args.agent, 
        args.layout, 
        args.num_episodes, 
        num_ghosts=args.num_ghosts,
        alpha=args.alpha, 
        gamma=args.gamma, 
        epsilon=args.epsilon
    )