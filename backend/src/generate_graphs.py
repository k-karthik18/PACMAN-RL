"""
generate_graphs.py
==================
Reads all CSV files produced by run_experiment.py and generates 8
publication-quality graphs (300 DPI PNG) into data/graphs/.

Usage (from backend/src/):
    python generate_graphs.py

Requires: matplotlib, numpy  (pip install matplotlib numpy)
"""

import os
import csv
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DATA_DIR   = "data"
GRAPHS_DIR = os.path.join(DATA_DIR, "graphs")
os.makedirs(GRAPHS_DIR, exist_ok=True)

ROLLING_WINDOW_SCORE   = 50
ROLLING_WINDOW_WINRATE = 100

AGENT_COLORS = {
    "ApproxQLearningAgent": "#4C9BE8",   # blue
    "ApproxSarsaAgent":     "#E87C4C",   # orange
    "QLearningAgent":       "#7CE87C",   # green
}
AGENT_LABELS = {
    "ApproxQLearningAgent": "Approx Q-Learning",
    "ApproxSarsaAgent":     "Approx SARSA",
    "QLearningAgent":       "Tabular Q-Learning",
}

FEATURE_COLORS = {
    "bias":           "#888888",
    "closestFood":    "#FFD700",
    "ghostDist":      "#FF6B6B",
    "danger":         "#CC0000",
    "nearGhost":      "#FF9999",
    "scaredGhostDist":"#00CCCC",
    "canEatGhost":    "#00FF88",
    "capsuleDist":    "#AA44FF",
    "mobility":       "#44AAFF",
    "stopped":        "#FF44AA",
    "reverse":        "#FFAA00",
}

STYLE = {
    "figure.facecolor": "#1A1A2E",
    "axes.facecolor":   "#16213E",
    "axes.edgecolor":   "#4A4A6A",
    "axes.labelcolor":  "#E0E0FF",
    "axes.titlesize":   13,
    "axes.labelsize":   10,
    "xtick.color":      "#B0B0D0",
    "ytick.color":      "#B0B0D0",
    "text.color":       "#E0E0FF",
    "grid.color":       "#2A2A4A",
    "grid.linewidth":   0.6,
    "legend.facecolor": "#0F3460",
    "legend.edgecolor": "#4A4A6A",
    "legend.fontsize":  8,
    "lines.linewidth":  1.5,
}


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def apply_style():
    plt.rcParams.update(STYLE)


def read_csv(path):
    """Return list of dicts from CSV, or [] if missing."""
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def rolling(values, window):
    """Simple rolling average (edges filled with partial windows)."""
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(np.mean(values[start: i + 1]))
    return np.array(result)


def save(fig, name):
    path = os.path.join(GRAPHS_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {path}")


def score_csv_for(agent_name):
    return {
        "ApproxQLearningAgent": "approx_q_scores.csv",
        "ApproxSarsaAgent":     "approx_sarsa_scores.csv",
        "QLearningAgent":       "qlearning_scores.csv",
    }[agent_name]


def weight_csv_for(agent_name):
    return {
        "ApproxQLearningAgent": "approx_weights.csv",
        "ApproxSarsaAgent":     "approx_sarsa_weights.csv",
    }.get(agent_name)


def feature_csv_for(agent_name):
    return {
        "ApproxQLearningAgent": "approx_q_features.csv",
        "ApproxSarsaAgent":     "approx_sarsa_features.csv",
    }.get(agent_name)


# ─────────────────────────────────────────────
# GRAPH FUNCTIONS
# ─────────────────────────────────────────────

def graph_score_curves(layout_name, graph_num):
    """Graph 1 & 2: Training score curves for all agents on a given layout."""
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(STYLE["figure.facecolor"])

    plotted = False
    agents = ["ApproxQLearningAgent", "ApproxSarsaAgent"]
    if layout_name == "smallClassic":
        agents.append("QLearningAgent")

    for agent_name in agents:
        rows = read_csv(os.path.join(DATA_DIR, score_csv_for(agent_name)))
        if not rows:
            continue
        scores = [float(r["score"]) for r in rows]
        eps    = list(range(1, len(scores) + 1))
        smooth = rolling(scores, ROLLING_WINDOW_SCORE)
        color  = AGENT_COLORS[agent_name]
        label  = AGENT_LABELS[agent_name]
        ax.plot(eps, scores, color=color, alpha=0.18, linewidth=0.6)
        ax.plot(eps, smooth, color=color, label=f"{label} (smooth)", linewidth=2)
        plotted = True

    if not plotted:
        print(f"  [Graph {graph_num}] No data for {layout_name} — skipping.")
        plt.close(fig)
        return

    ax.set_title(f"Training Score Curves — {layout_name}", pad=12)
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Score (rolling avg window={ROLLING_WINDOW_SCORE})")
    ax.axhline(0, color="#555577", linewidth=0.8, linestyle="--")
    ax.grid(True, axis="both")
    ax.legend(loc="lower right")
    save(fig, f"graph{graph_num}_scores_{layout_name}.png")


def graph_winrate_curves(graph_num):
    """Graph 3: Rolling win-rate curves for all agents on both layouts."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.patch.set_facecolor(STYLE["figure.facecolor"])
    fig.suptitle(f"Win Rate Curves (rolling window={ROLLING_WINDOW_WINRATE})", fontsize=13, color=STYLE["text.color"])

    layout_agents = [
        ("smallClassic",  ["ApproxQLearningAgent", "ApproxSarsaAgent", "QLearningAgent"]),
        ("mediumClassic", ["ApproxQLearningAgent", "ApproxSarsaAgent"]),
    ]

    for ax, (layout_name, agents) in zip(axes, layout_agents):
        ax.set_facecolor(STYLE["axes.facecolor"])
        ax.set_title(layout_name, color=STYLE["text.color"])
        ax.set_xlabel("Episode")
        ax.set_ylabel("Win Rate (%)")
        ax.grid(True)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

        for agent_name in agents:
            rows = read_csv(os.path.join(DATA_DIR, score_csv_for(agent_name)))
            if not rows:
                continue
            wins   = [float(r["win"]) for r in rows]
            eps    = list(range(1, len(wins) + 1))
            smooth = rolling(wins, ROLLING_WINDOW_WINRATE)
            color  = AGENT_COLORS[agent_name]
            label  = AGENT_LABELS[agent_name]
            ax.plot(eps, smooth, color=color, label=label, linewidth=2)

        ax.legend(loc="upper left")
        ax.set_ylim(0, 1.05)

    save(fig, f"graph{graph_num}_winrate_curves.png")


def graph_weight_evolution(agent_name, graph_num):
    """Graphs 4 & 5: Weight evolution per episode for an approx agent."""
    apply_style()
    wcsv = weight_csv_for(agent_name)
    if not wcsv:
        return
    rows = read_csv(os.path.join(DATA_DIR, wcsv))
    if not rows:
        print(f"  [Graph {graph_num}] No weight data for {agent_name} — skipping.")
        return

    feature_cols = [k for k in rows[0].keys() if k.lower() != "episode"]
    eps = [int(r.get("Episode", r.get("episode", i + 1))) for i, r in enumerate(rows)]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(STYLE["figure.facecolor"])

    for feat in feature_cols:
        values = [float(r[feat]) for r in rows]
        color  = FEATURE_COLORS.get(feat, "#AAAAAA")
        ax.plot(eps, values, label=feat, color=color, linewidth=1.5, alpha=0.9)

    ax.axhline(0, color="#555577", linewidth=0.8, linestyle="--")
    ax.set_title(f"Feature Weight Evolution — {AGENT_LABELS.get(agent_name, agent_name)}", pad=12)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Weight Value")
    ax.grid(True)
    ax.legend(loc="upper right", ncol=2, fontsize=7)
    save(fig, f"graph{graph_num}_weights_{agent_name}.png")


def graph_feature_values(agent_name, graph_num):
    """Graph 6: Average feature values per episode."""
    apply_style()
    fcsv = feature_csv_for(agent_name)
    if not fcsv:
        return
    rows = read_csv(os.path.join(DATA_DIR, fcsv))
    if not rows:
        print(f"  [Graph {graph_num}] No feature data for {agent_name} — skipping.")
        return

    feat_cols = [k for k in rows[0].keys() if k != "episode"]
    eps = [int(r["episode"]) for r in rows]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(STYLE["figure.facecolor"])

    # Only plot a curated subset for readability
    focus = ["closestFood", "danger", "canEatGhost", "ghostDist", "scaredGhostDist"]
    for feat in focus:
        if feat not in feat_cols:
            continue
        values = [float(r[feat]) for r in rows]
        smooth = rolling(values, 30)
        color  = FEATURE_COLORS.get(feat, "#AAAAAA")
        ax.plot(eps, smooth, label=feat, color=color, linewidth=2)

    ax.set_title(f"Avg Feature Values per Episode — {AGENT_LABELS.get(agent_name, agent_name)}", pad=12)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Feature Value (rolling avg window=30)")
    ax.grid(True)
    ax.legend(loc="upper right")
    save(fig, f"graph{graph_num}_feature_values_{agent_name}.png")


def graph_test_bar(metric, graph_num):
    """Graphs 7 & 8: Test bar charts — avg score or win rate across agents × layouts."""
    apply_style()

    agent_names = ["ApproxQLearningAgent", "ApproxSarsaAgent", "QLearningAgent"]
    layouts     = ["smallClassic", "mediumClassic"]

    data = {}
    for agent in agent_names:
        safe = agent.replace("Agent", "")
        for lay in layouts:
            path = os.path.join(DATA_DIR, f"test_{safe}_{lay}.csv")
            rows = read_csv(path)
            if not rows:
                continue
            if metric == "score":
                vals = [float(r["score"]) for r in rows]
                data[(agent, lay)] = np.mean(vals)
            elif metric == "winrate":
                wins = sum(1 for r in rows if r["win"] == "1")
                data[(agent, lay)] = wins / len(rows) * 100

    if not data:
        print(f"  [Graph {graph_num}] No test data found — skipping.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(STYLE["figure.facecolor"])

    x            = np.arange(len(agent_names))
    bar_width    = 0.35
    layout_colors = {"smallClassic": "#4C9BE8", "mediumClassic": "#E87C4C"}

    for li, lay in enumerate(layouts):
        vals   = [data.get((a, lay), 0) for a in agent_names]
        labels = [AGENT_LABELS[a] for a in agent_names]
        offset = (li - 0.5) * bar_width
        bars   = ax.bar(x + offset, vals, bar_width,
                        label=lay, color=layout_colors[lay], alpha=0.85, edgecolor="#ffffff22")
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (1 if metric == "score" else 0.5),
                        f"{val:.1f}{'%' if metric=='winrate' else ''}",
                        ha="center", va="bottom", fontsize=7, color="#E0E0FF")

    ax.set_xticks(x)
    ax.set_xticklabels([AGENT_LABELS[a] for a in agent_names], rotation=10)
    title_metric = "Average Score" if metric == "score" else "Win Rate (%)"
    ax.set_title(f"Test {title_metric} — Agent × Layout Comparison", pad=12)
    ax.set_ylabel(title_metric)
    ax.grid(True, axis="y", linewidth=0.6)
    ax.legend()
    save(fig, f"graph{graph_num}_test_{metric}_bar.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("GENERATING GRAPHS")
    print("="*60 + "\n")

    # Graph 1: Training scores — smallClassic
    print("[Graph 1] Training score curves — smallClassic")
    graph_score_curves("smallClassic", graph_num=1)

    # Graph 2: Training scores — mediumClassic
    print("[Graph 2] Training score curves — mediumClassic")
    graph_score_curves("mediumClassic", graph_num=2)

    # Graph 3: Win rate curves — all agents both layouts
    print("[Graph 3] Win rate curves")
    graph_winrate_curves(graph_num=3)

    # Graph 4: Weight evolution — ApproxQL
    print("[Graph 4] Weight evolution — ApproxQLearningAgent")
    graph_weight_evolution("ApproxQLearningAgent", graph_num=4)

    # Graph 5: Weight evolution — ApproxSARSA
    print("[Graph 5] Weight evolution — ApproxSarsaAgent")
    graph_weight_evolution("ApproxSarsaAgent", graph_num=5)

    # Graph 6: Feature value curves — ApproxQL
    print("[Graph 6] Feature value curves — ApproxQLearningAgent")
    graph_feature_values("ApproxQLearningAgent", graph_num=6)

    # Graph 7: Test avg score bar chart
    print("[Graph 7] Test avg score bar chart")
    graph_test_bar("score", graph_num=7)

    # Graph 8: Test win rate bar chart
    print("[Graph 8] Test win rate bar chart")
    graph_test_bar("winrate", graph_num=8)

    print(f"\n{'='*60}")
    print(f"ALL 8 GRAPHS SAVED → {GRAPHS_DIR}")
    print("="*60)
    print("Review: data/results_report.md for the full summary table.")


if __name__ == "__main__":
    main()
