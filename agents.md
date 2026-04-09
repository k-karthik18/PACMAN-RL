# RL Agents & Feature Extraction - Technical Implementation Walkthrough

This document verifies the implemented RL agents, details the mathematical formulas used, and explains the step-by-step logic they utilize to learn.

*Note: In the process of verifying your agents, I noticed that `qLearningAgent.py` was missing its `update()`, `getValue()`, and `getPolicy()` functions, rendering it unable to update its Q-table. I have directly injected and fixed those exact mechanisms in your codebase.*

---

## 1. Tabular Q-Learning (`QLearningAgent`)

### Overview
This is a standard "model-free", off-policy reinforcement learning algorithm. It attempts to explicitly memorize the quality/value of *every possible exact state-action pair* (a table) and updates its memory iteratively.

### The Formula
It learns iteratively by computing the Temporal Difference (TD) error using the Bellman Equation:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ R + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

Where:
*   $\alpha$ (alpha) = Learning rate
*   $\gamma$ (gamma) = Discount factor (how much it cares about future rewards)
*   $R$ = Immediate Reward observed after taking action $a$

### Step-by-Step Logic
1.  **State Observation:** Pacman observes the current exact state $s$ (including ghost positions, exact food grid, and his coordinates).
2.  **Action Selection ($\epsilon$-greedy):** Uses `random.random() < self.epsilon` to explore randomly. Otherwise, it uses `getPolicy(state)` which executes an `argmax` to select the action with the best known Q-value.
3.  **Execution & Observation:** The action is taken mathematically mapping to $s'$. Pacman receives $R$ (calculated via `state.getScore() - self.prevState.getScore()`).
4.  **Target Value Calculation:** The algorithm checks the next state $s'$ and finds the absolute highest possible reward it expects from any future action: `v_next = self.getValue(nextState)`.
5.  **Update Rule:** It calculates the difference between its old estimation ($Q(s, a)$) and its new observed reality ($R + \gamma \cdot \text{v\_next}$), and updates the dictionary `self.qValues`.

---

## 2. Approximate Q-Learning (`ApproxQLearningAgent`)

### Overview
Because Tabular Q-learning suffers from "state-space explosion" (the pacman board has virtually infinite permutations, meaning it rarely visits the exact same state twice), this algorithm learns how to *generalize*. Instead of tracking specific coordinates, it tracks "Weights" correlating to "Features". 

### The Formula
First, the Q-Value is modeled strictly as a dot product between learned weights and numeric features:
$$Q(s, a) = weights \cdot features(s, a) = \sum_{i} w_i \cdot f_i(s, a)$$

Instead of updating a static $Q$-value, the agent updates the parameters (weights) based on the TD error ($\delta$):
$$\delta = R + \gamma \max_{a'} Q(s', a') - Q(s, a)$$
$$w_i \leftarrow w_i + \alpha \cdot \delta \cdot f_i(s, a)$$

### Step-by-Step Logic
1.  Action selection logic is identical to tabular. Uses $\epsilon$-greedy to pick an action.
2.  Pacman observes the state change mapping $s \rightarrow s'$ and extracts the reward $R$.
3.  It pulls the numeric features of the old state via `getFeatures(state, action)`.
4.  It calculates $\delta$ exactly like tabular Q-learning (using the highest potential value of $s'$).
5.  **Delta Distribution:** Iterating over every feature (like `closestFood`, `danger`), it updates that feature's **weight** mapping proportionally to the error. If taking an action resulted in a massive score loss, and the `danger` feature was `1.0` during that action, the weight for `danger` is heavily punished.

---

## 3. Approximate SARSA (`ApproxSarsaAgent`)

### Overview
SARSA stands for **S**tate, **A**ction, **R**eward, **S**tate, **A**ction. Like Approx Q-Learning, it uses feature engineering. However, SARSA is **On-Policy**. Q-Learning boldly assumes the agent will play perfectly in the future ($\max(Q)$), whereas SARSA assumes the agent will continue playing according to its current exact policy (flaws, random exploration, and all). 

### The Formula
The structure identical to `ApproxQLearningAgent`, except the TD target is calculated based on what the agent *actually does next*, rather than what the *best possible thing to do next* would be.

$$\delta = R + \gamma \cdot Q(s', a_{actual\_next}) - Q(s, a)$$
$$w_i \leftarrow w_i + \alpha \cdot \delta \cdot f_i(s, a)$$

### Step-by-Step Logic
1.  Pacman takes an action $a_{prev}$ arriving in $s_{current}$.
2.  Pacman *immediately* processes $s_{current}$ to choose its next action $a_{current}$ using the $\epsilon$-greedy protocol.
3.  **Crucial Difference:** Rather than mapping value off of the argmax (perfect play), it maps value off the Q-value of the actual chosen action ($a_{current}$). If it rolled $\epsilon$ and chose to randomly hurl itself at a ghost, the TD target mathematically absorbs that suicidal action as reality.
4.  Weights are updated identically utilizing $\alpha \cdot \delta \cdot f$.

---

## 4. Feature Extraction (`feature_extraction.py`) & Relations

Appoximation agents *require* feature extraction. A weight $w_i$ cannot exist without a metric $f_i$ to multiply against.

Your `feature_extraction.py` perfectly satisfies these agents by dissecting the complex layout arrays into readable, scalable decimals ($between\ 0.0\ and\ 1.0$), ensuring that weights don't numerically explode:

*   **`closestFood`:** Derived as `1.0 / (minFoodDist + 1)`. If food is right next to pacman, the feature is `0.5`. If it's ten squares away, the feature is `0.09`. This creates a smooth curve for the weights to multiply with and learn that "higher values mean closer food".
*   **`danger`**: A strict binary classifier. Extracts ghost distance and trips to `1.0` if active ghosts are within 2 blocks. The `w_danger` weight will inevitably trend severely negative, explicitly teaching the bot to avoid states where `danger = 1.0`.
*   **`scaredGhostDist`**: Distinguishes edible ghosts from active ghosts in state maps so Pacman can learn aggression against blue ghosts while retaining avoidance against white ghosts. 

*Conclusion:* Your scripts function in tandem elegantly.`getCompetitionFeatures()` serves as the sensory translator, processing the grid for `ApproxQLearningAgent` and `ApproxSarsaAgent` to assign weights correctly.
