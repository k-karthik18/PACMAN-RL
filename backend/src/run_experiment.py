"""
run_experiment.py
=================
Master experiment script for research-grade training and evaluation.

Steps:
  1. Clears all old weights & CSVs in data/
  2. Trains each agent × each layout for 2000 episodes (headless)
  3. Tests each trained agent × layout for 1000 frozen episodes (alpha=0, epsilon=0.05)
  4. Auto-generates data/results_report.md

Usage (from backend/src/):
    python run_experiment.py

Estimated time: 30-60 minutes total on CPU.
"""

import os
import sys
import glob
import time
import csv
import statistics

import pacman
import layout as layout_mod
import ghostAgents
import textDisplay
from pacman import ClassicGameRules

# ─────────────────────────────────────────────
# EXPERIMENT CONFIGURATION
# ─────────────────────────────────────────────

TRAIN_EPISODES = 2000
TEST_EPISODES  = 1000
NUM_GHOSTS     = 1

AGENTS = [
    {"name": "ApproxQLearningAgent", "alpha": 0.2, "gamma": 0.9, "epsilon": 0.5},
    {"name": "ApproxSarsaAgent",     "alpha": 0.2, "gamma": 0.9, "epsilon": 0.5},
    {"name": "QLearningAgent",       "alpha": 0.2, "gamma": 0.9, "epsilon": 0.5},
]

# QLearningAgent excluded from mediumClassic (state-space explosion)
LAYOUT_MAP = {
    "ApproxQLearningAgent": ["smallClassic", "mediumClassic"],
    "ApproxSarsaAgent":     ["smallClassic", "mediumClassic"],
    "QLearningAgent":       ["smallClassic"],
}

DATA_DIR = "data"


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def clear_data():
    """Delete all old weights and CSV files so experiments start fresh."""
    print("\n" + "="*60)
    print("CLEARING OLD DATA")
    print("="*60)
    patterns = [
        os.path.join(DATA_DIR, "*.pkl"),
        os.path.join(DATA_DIR, "*.csv"),
        os.path.join(DATA_DIR, "results_report.md"),
    ]
    removed = []
    for pattern in patterns:
        for f in glob.glob(pattern):
            os.remove(f)
            removed.append(os.path.basename(f))
    if removed:
        print(f"Removed: {', '.join(removed)}")
    else:
        print("No old files found — starting clean.")
    os.makedirs(DATA_DIR, exist_ok=True)


def _run_episodes(agent_name, layout_name, num_episodes, alpha, gamma, epsilon, label="TRAIN"):
    """Run N episodes, return list of (score, win) tuples."""
    lay = layout_mod.getLayout(layout_name)
    if lay is None:
        print(f"  ERROR: Layout '{layout_name}' not found. Skipping.")
        return []

    agent_type = pacman.loadAgent(agent_name, nographics=True)
    agent_instance = agent_type(alpha=alpha, gamma=gamma, epsilon=epsilon)

    ghosts  = [ghostAgents.RandomGhost(i + 1) for i in range(NUM_GHOSTS)]
    display = textDisplay.NullGraphics()
    rules   = ClassicGameRules(timeout=30)

    results = []
    start   = time.time()

    for i in range(num_episodes):
        game = rules.newGame(lay, agent_instance, ghosts, display, quiet=True)
        game.run()
        score = game.state.getScore()
        win   = game.state.isWin()
        results.append((score, win))

        if (i + 1) % 200 == 0:
            last = results[-200:]
            avg_score = sum(s for s, _ in last) / len(last)
            win_pct   = sum(1 for _, w in last if w) / len(last) * 100
            elapsed   = time.time() - start
            print(
                f"  [{label}] {i+1:>4}/{num_episodes} | "
                f"avg_score(last 200)={avg_score:>7.1f} | "
                f"win%={win_pct:>5.1f}% | "
                f"elapsed={elapsed:.0f}s"
            )

    return results


def train_agent(agent_cfg, layout_name):
    name    = agent_cfg["name"]
    alpha   = agent_cfg["alpha"]
    gamma   = agent_cfg["gamma"]
    epsilon = agent_cfg["epsilon"]

    print(f"\n{'='*60}")
    print(f"TRAINING  {name}  on  {layout_name}")
    print(f"Episodes={TRAIN_EPISODES} | α={alpha} | γ={gamma} | ε={epsilon}")
    print(f"{'='*60}")

    t0 = time.time()
    results = _run_episodes(name, layout_name, TRAIN_EPISODES, alpha, gamma, epsilon, label="TRAIN")
    elapsed = time.time() - t0

    if results:
        wins      = sum(1 for _, w in results if w)
        avg_score = sum(s for s, _ in results) / len(results)
        print(f"\n  ✓ Training done in {elapsed:.1f}s | "
              f"win rate={wins}/{TRAIN_EPISODES} ({wins/TRAIN_EPISODES*100:.1f}%) | "
              f"avg score={avg_score:.1f}")

    return results


def test_agent(agent_cfg, layout_name):
    """Frozen test: alpha=0, epsilon=0.05 (greedy)."""
    name  = agent_cfg["name"]
    gamma = agent_cfg["gamma"]

    print(f"\n{'='*60}")
    print(f"TESTING   {name}  on  {layout_name}  (frozen weights)")
    print(f"Episodes={TEST_EPISODES} | α=0 | ε=0.05")
    print(f"{'='*60}")

    t0 = time.time()
    # Save test results to a separate CSV via the agent's own logger is NOT used
    # (alpha=0 means no weight updates, epsilon=0.05 is near-greedy)
    results = _run_episodes(name, layout_name, TEST_EPISODES,
                            alpha=0.0, gamma=gamma, epsilon=0.05, label="TEST ")
    elapsed = time.time() - t0

    if results:
        wins      = sum(1 for _, w in results if w)
        scores    = [s for s, _ in results]
        avg_score = sum(scores) / len(scores)
        std_score = statistics.stdev(scores) if len(scores) > 1 else 0
        print(f"\n  ✓ Test done in {elapsed:.1f}s | "
              f"win rate={wins}/{TEST_EPISODES} ({wins/TEST_EPISODES*100:.1f}%) | "
              f"avg score={avg_score:.1f} ± {std_score:.1f}")

    # Save test results to data/test_{agent}_{layout}.csv
    safe_agent  = name.replace("Agent", "")
    csv_path    = os.path.join(DATA_DIR, f"test_{safe_agent}_{layout_name}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "score", "win"])
        for idx, (score, win) in enumerate(results, start=1):
            writer.writerow([idx, score, int(win)])
    print(f"  Saved → {csv_path}")

    return results


# ─────────────────────────────────────────────
# REPORT GENERATION
# ─────────────────────────────────────────────

def _summary_from_csv(csv_path, score_col="score", win_col="win", last_n=200):
    """Read a CSV and return avg/best/worst score and win% over last N rows."""
    rows = []
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except FileNotFoundError:
        return None

    rows = rows[-last_n:] if len(rows) >= last_n else rows
    if not rows:
        return None

    scores = [float(r[score_col]) for r in rows]
    wins   = sum(1 for r in rows if r.get(win_col, "0") == "1")
    return {
        "n":          len(rows),
        "avg_score":  round(sum(scores) / len(scores), 1),
        "best_score": round(max(scores), 1),
        "worst_score":round(min(scores), 1),
        "std_score":  round(statistics.stdev(scores) if len(scores) > 1 else 0, 1),
        "win_pct":    round(wins / len(rows) * 100, 1),
    }


def generate_report(train_summary: dict, test_summary: dict):
    """Write data/results_report.md."""
    report_path = os.path.join(DATA_DIR, "results_report.md")

    lines = []
    lines.append("# PACMAN-RL — Experiment Results Report\n")
    lines.append(f"*Auto-generated by `run_experiment.py`*\n")
    lines.append(f"*Training episodes: {TRAIN_EPISODES} | Test episodes: {TEST_EPISODES}*\n\n")
    lines.append("---\n\n")

    # ── Training Summary ──
    lines.append("## Training Summary (last 200 episodes)\n\n")
    lines.append("| Agent | Layout | Avg Score | Best | Worst | Std Dev | Win Rate |\n")
    lines.append("|---|---|---|---|---|---|---|\n")
    for key, s in train_summary.items():
        agent, layout = key
        if s:
            lines.append(
                f"| {agent} | {layout} | {s['avg_score']} | {s['best_score']} | "
                f"{s['worst_score']} | {s['std_score']} | {s['win_pct']}% |\n"
            )
        else:
            lines.append(f"| {agent} | {layout} | — | — | — | — | — |\n")
    lines.append("\n---\n\n")

    # ── Test Summary ──
    lines.append("## Test Summary (frozen weights, epsilon=0.05)\n\n")
    lines.append("| Agent | Layout | Avg Score | Best | Worst | Std Dev | Win Rate |\n")
    lines.append("|---|---|---|---|---|---|---|\n")
    for key, s in test_summary.items():
        agent, layout = key
        if s:
            lines.append(
                f"| {agent} | {layout} | {s['avg_score']} | {s['best_score']} | "
                f"{s['worst_score']} | {s['std_score']} | {s['win_pct']}% |\n"
            )
        else:
            lines.append(f"| {agent} | {layout} | — | — | — | — | — |\n")
    lines.append("\n---\n\n")

    # ── Final Weights ──
    lines.append("## Final Learned Weights\n\n")
    for wfile, label in [
        ("approx_weights.csv",      "ApproxQLearningAgent"),
        ("approx_sarsa_weights.csv","ApproxSarsaAgent"),
    ]:
        wpath = os.path.join(DATA_DIR, wfile)
        try:
            with open(wpath, newline="") as f:
                rows = list(csv.DictReader(f))
            if rows:
                last = rows[-1]
                lines.append(f"### {label}\n\n")
                lines.append("| Feature | Final Weight |\n")
                lines.append("|---|---|\n")
                for k, v in last.items():
                    if k.lower() != "episode":
                        lines.append(f"| {k} | {float(v):.4f} |\n")
                lines.append("\n")
        except FileNotFoundError:
            pass

    lines.append("---\n\n")

    # ── Key Findings ──
    lines.append("## Key Findings\n\n")

    # Compare ApproxQL vs ApproxSARSA on smallClassic (test)
    ql_small  = test_summary.get(("ApproxQLearningAgent", "smallClassic"))
    sa_small  = test_summary.get(("ApproxSarsaAgent",     "smallClassic"))
    tab_small = test_summary.get(("QLearningAgent",       "smallClassic"))

    if ql_small and sa_small:
        winner = "ApproxQLearningAgent" if ql_small["win_pct"] >= sa_small["win_pct"] else "ApproxSarsaAgent"
        lines.append(
            f"- **Best agent on smallClassic**: `{winner}` "
            f"(ApproxQL: {ql_small['win_pct']}% | ApproxSARSA: {sa_small['win_pct']}%)\n"
        )
    if ql_small and tab_small:
        lines.append(
            f"- **Tabular vs Approximate on smallClassic**: "
            f"QLearning={tab_small['win_pct']}% vs ApproxQL={ql_small['win_pct']}% "
            f"— demonstrates {'advantage of function approximation' if ql_small['win_pct'] > tab_small['win_pct'] else 'tabular competitiveness on small maps'}\n"
        )

    ql_med = test_summary.get(("ApproxQLearningAgent", "mediumClassic"))
    sa_med = test_summary.get(("ApproxSarsaAgent",     "mediumClassic"))
    if ql_med and sa_med:
        lines.append(
            f"- **Generalization to mediumClassic**: "
            f"ApproxQL={ql_med['win_pct']}% | ApproxSARSA={sa_med['win_pct']}%\n"
        )

    lines.append(
        f"- **Training duration**: {TRAIN_EPISODES} episodes per agent per layout\n"
    )
    lines.append(
        "- Run `python generate_graphs.py` to produce all 8 publication-quality graphs.\n"
    )

    with open(report_path, "w") as f:
        f.writelines(lines)

    print(f"\n  ✓ Results report saved → {report_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    total_start = time.time()

    # Step 1: Clear data
    clear_data()

    train_summary = {}
    test_summary  = {}

    # Step 2 & 3: Train then test each combination
    for agent_cfg in AGENTS:
        agent_name = agent_cfg["name"]
        layouts    = LAYOUT_MAP[agent_name]

        for layout_name in layouts:
            # -- Train --
            train_agent(agent_cfg, layout_name)

            # Read training CSV for summary (last 200 eps)
            score_csv = {
                "ApproxQLearningAgent": "approx_q_scores.csv",
                "ApproxSarsaAgent":     "approx_sarsa_scores.csv",
                "QLearningAgent":       "qlearning_scores.csv",
            }[agent_name]
            s = _summary_from_csv(
                os.path.join(DATA_DIR, score_csv),
                score_col="score", win_col="win", last_n=200
            )
            train_summary[(agent_name, layout_name)] = s

            # -- Test --
            test_agent(agent_cfg, layout_name)

            safe_agent = agent_name.replace("Agent", "")
            s = _summary_from_csv(
                os.path.join(DATA_DIR, f"test_{safe_agent}_{layout_name}.csv"),
                score_col="score", win_col="win", last_n=TEST_EPISODES
            )
            test_summary[(agent_name, layout_name)] = s

    # Step 4: Generate report
    generate_report(train_summary, test_summary)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS COMPLETE in {total_elapsed/60:.1f} minutes")
    print(f"{'='*60}")
    print("Next step: python generate_graphs.py")


if __name__ == "__main__":
    main()
