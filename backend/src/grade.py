"""
grade.py

Agent evaluation and grading script.
Runs agents across multiple layouts and reports win rates.

Usage:
    python grade.py -p ApproxQLearningAgent
    python grade.py -p ApproxQLearningAgent -n 100
    python grade.py -p ApproxQLearningAgent -l openClassic,testClassic -n 50

This script:
1. Runs the specified agent on multiple layouts
2. Tracks win rates, scores, and learning progress
3. Reports pass/fail based on target thresholds
4. Helps tune hyperparameters for better performance
"""

import sys
import os
import time

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pacman
import layout
import ghostAgents
from game import Agent


# Target win rates for different layout categories
TARGETS = {
    # Small maps: 90%+ win rate
    "smallClassic": {"min_win_rate": 0.90, "category": "small"},
    "testClassic": {"min_win_rate": 0.90, "category": "small"},
    "minimaxClassic": {"min_win_rate": 0.90, "category": "small"},
    
    # Medium maps: 60%+ win rate
    "mediumClassic": {"min_win_rate": 0.60, "category": "medium"},
    "trickyClassic": {"min_win_rate": 0.55, "category": "medium"},
    "contestClassic": {"min_win_rate": 0.50, "category": "medium"},
    
    # Large/open maps: 50%+ win rate
    "openClassic": {"min_win_rate": 0.80, "category": "open"},
    "originalClassic": {"min_win_rate": 0.50, "category": "large"},
    "capsuleClassic": {"min_win_rate": 0.55, "category": "medium"},
    
    # Special maps
    "powerClassic": {"min_win_rate": 0.70, "category": "special"},
    "trappedClassic": {"min_win_rate": 0.40, "category": "hard"},
}

DEFAULT_LAYOUTS = [
    "openClassic",
    "testClassic", 
    "smallClassic",
    "minimaxClassic",
    "mediumClassic",
    "trickyClassic",
]


def run_evaluation(agent_name, layouts=None, num_games=50, num_ghosts=1, 
                   num_training=0, quiet=True, params=None):
    """
    Run evaluation of an agent across specified layouts.
    
    Args:
        agent_name: Name of agent class (e.g., "ApproxQLearningAgent")
        layouts: List of layout names (default: DEFAULT_LAYOUTS)
        num_games: Games per layout
        num_ghosts: Number of ghosts
        num_training: Number of training games (no display)
        quiet: Suppress output
        params: Agent parameters dict
    
    Returns:
        dict: Results per layout and overall grade
    """
    import pacman
    import layout
    import ghostAgents
    from game import Agent
    
    if layouts is None:
        layouts = DEFAULT_LAYOUTS
    
    if params is None:
        params = {}
    
    results = {}
    total_wins = 0
    total_games = 0
    
    print(f"\n{'='*60}")
    print(f"EVALUATING: {agent_name}")
    print(f"Games per layout: {num_games}, Ghosts: {num_ghosts}")
    print(f"Parameters: {params if params else 'default'}")
    print(f"{'='*60}\n")
    
    for lay_name in layouts:
        # Check if layout exists
        lay = layout.getLayout(lay_name)
        if lay is None:
            print(f"  [SKIP] Layout '{lay_name}' not found")
            continue
        
        # Load agent
        agent_class = pacman.loadAgent(agent_name, nographics=True)
        agent_instance = agent_class(**params)
        
        # Create ghosts
        ghosts = [ghostAgents.RandomGhost(i + 1) for i in range(num_ghosts)]
        
        # Run games
        import textDisplay
        
        rules = pacman.ClassicGameRules(timeout=30)
        wins = 0
        scores = []
        
        for game_num in range(num_games):
            # Training games (quiet)
            if game_num < num_training:
                display = textDisplay.NullGraphics()
                rules.quiet = True
            else:
                if quiet:
                    display = textDisplay.NullGraphics()
                    rules.quiet = True
                else:
                    display = textDisplay.NullGraphics()
                    rules.quiet = True
            
            game = rules.newGame(lay, agent_instance, ghosts, display, quiet=True)
            game.run()
            
            score = game.state.getScore()
            win = game.state.isWin()
            scores.append(score)
            if win:
                wins += 1
        
        # Calculate stats
        win_rate = wins / num_games
        avg_score = sum(scores) / num_games if scores else 0
        
        # Get target
        target = TARGETS.get(lay_name, {"min_win_rate": 0.50, "category": "unknown"})
        min_wr = target["min_win_rate"]
        passed = win_rate >= min_wr
        
        results[lay_name] = {
            "wins": wins,
            "games": num_games,
            "win_rate": win_rate,
            "avg_score": avg_score,
            "best_score": max(scores) if scores else 0,
            "worst_score": min(scores) if scores else 0,
            "target": min_wr,
            "passed": passed,
            "category": target["category"],
        }
        
        total_wins += wins
        total_games += num_games
        
        # Print result
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {lay_name}: {wins}/{num_games} wins ({win_rate*100:.1f}%) "
              f"[target: {min_wr*100:.0f}%] avg_score: {avg_score:.1f}")
    
    # Overall grade
    overall_win_rate = total_wins / total_games if total_games > 0 else 0
    
    # Calculate grade
    passed_count = sum(1 for r in results.values() if r["passed"])
    total_layouts = len(results)
    
    if total_layouts > 0:
        pass_rate = passed_count / total_layouts
        if pass_rate >= 0.9:
            grade = "A"
        elif pass_rate >= 0.7:
            grade = "B"
        elif pass_rate >= 0.5:
            grade = "C"
        elif pass_rate >= 0.3:
            grade = "D"
        else:
            grade = "F"
    else:
        grade = "N/A"
    
    print(f"\n{'='*60}")
    print(f"OVERALL: {total_wins}/{total_games} wins ({overall_win_rate*100:.1f}%)")
    print(f"GRADE: {grade} ({passed_count}/{total_layouts} layouts passed)")
    print(f"{'='*60}\n")
    
    return {
        "layouts": results,
        "overall_win_rate": overall_win_rate,
        "grade": grade,
        "passed_count": passed_count,
        "total_layouts": total_layouts,
    }


def quick_test(agent_name, num_games=20, params=None):
    """Quick test on key layouts."""
    print("\n--- QUICK TEST ---")
    return run_evaluation(
        agent_name, 
        layouts=["openClassic", "testClassic", "minimaxClassic"],
        num_games=num_games,
        num_ghosts=1,
        params=params,
    )


def full_test(agent_name, num_games=50, params=None):
    """Full test on all layouts."""
    print("\n--- FULL TEST ---")
    return run_evaluation(
        agent_name,
        layouts=list(TARGETS.keys()),
        num_games=num_games,
        num_ghosts=1,
        params=params,
    )


def tune_parameters(agent_name, base_params=None):
    """
    Grid search over parameters to find best configuration.
    """
    print("\n--- PARAMETER TUNING ---")
    
    if base_params is None:
        base_params = {"alpha": 0.1, "gamma": 0.9, "epsilon": 0.2}
    
    # Parameter ranges to try
    alphas = [0.05, 0.1, 0.2]
    gammas = [0.8, 0.9, 0.95]
    epsilons = [0.1, 0.2, 0.3]
    
    best_wr = 0
    best_params = base_params.copy()
    
    for alpha in alphas:
        for gamma in gammas:
            for epsilon in epsilons:
                params = {"alpha": alpha, "gamma": gamma, "epsilon": epsilon}
                
                result = quick_test(agent_name, num_games=10, params=params)
                wr = result["overall_win_rate"]
                
                print(f"  α={alpha}, γ={gamma}, ε={epsilon}: {wr*100:.1f}%")
                
                if wr > best_wr:
                    best_wr = wr
                    best_params = params.copy()
    
    print(f"\nBest params: {best_params} (win rate: {best_wr*100:.1f}%)")
    return best_params


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Grade Pacman agents")
    parser.add_argument("-p", "--agent", type=str, default="ApproxQLearningAgent",
                        help="Agent class name")
    parser.add_argument("-l", "--layouts", type=str, default=None,
                        help="Comma-separated layout names")
    parser.add_argument("-n", "--num_games", type=int, default=50,
                        help="Number of games per layout")
    parser.add_argument("-k", "--num_ghosts", type=int, default=1,
                        help="Number of ghosts")
    parser.add_argument("-t", "--train", type=int, default=0,
                        help="Number of training games")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test on key layouts")
    parser.add_argument("--full", action="store_true",
                        help="Full test on all layouts")
    parser.add_argument("--tune", action="store_true",
                        help="Tune parameters")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.9,
                        help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=0.2,
                        help="Exploration rate")
    
    args = parser.parse_args()
    
    params = {
        "alpha": args.alpha,
        "gamma": args.gamma,
        "epsilon": args.epsilon,
    }
    
    layouts = None
    if args.layouts:
        layouts = [l.strip() for l in args.layouts.split(",")]
    
    if args.tune:
        tune_parameters(args.agent, params)
    elif args.quick:
        quick_test(args.agent, args.num_games, params)
    elif args.full:
        full_test(args.agent, args.num_games, params)
    else:
        run_evaluation(
            args.agent,
            layouts=layouts,
            num_games=args.num_games,
            num_ghosts=args.num_ghosts,
            num_training=args.train,
            params=params,
        )
