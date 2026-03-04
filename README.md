📘 RL Pacman – Reinforcement Learning Implementation
📌 Overview

This project implements different Reinforcement Learning agents for the UC Berkeley Pacman environment.

The goal is to understand:

Tabular Q-Learning

Approximate Q-Learning

Exploration vs Exploitation

Curse of Dimensionality

Function Approximation

The environment files (pacman.py, game.py, layouts/) are unchanged.
Only new RL agents were implemented.

🧠 Implemented Agents
1️⃣ Random Agent
🔹 Idea

Chooses a legal action uniformly at random.

🔹 Policy
𝜋
(
𝑎
∣
𝑠
)
=
1
∣
𝐴
(
𝑠
)
∣
π(a∣s)=
∣A(s)∣
1
	​


No learning. No update rule.

🔹 Purpose

Used as baseline for comparison.

▶ Run
py pacman.py -p RandomAgent -l testClassic

Train multiple:

py pacman.py -p RandomAgent -l testClassic -n 50 -q
2️⃣ Tabular Q-Learning Agent
🔹 Theory

Stores:

𝑄
(
𝑠
,
𝑎
)
Q(s,a)

Update rule:

𝑄
(
𝑠
,
𝑎
)
←
𝑄
(
𝑠
,
𝑎
)
+
𝛼
[
𝑟
+
𝛾
max
⁡
𝑎
′
𝑄
(
𝑠
′
,
𝑎
′
)
−
𝑄
(
𝑠
,
𝑎
)
]
Q(s,a)←Q(s,a)+α[r+γ
a
′
max
	​

Q(s
′
,a
′
)−Q(s,a)]

Where:

α = learning rate

γ = discount factor

r = reward

ε = exploration rate

Uses ε-greedy policy:

With probability ε → random action

Otherwise → best action

⚠ Limitation

Full GameState is extremely large.
State space explosion leads to:

Very low win rate

Poor generalization

Curse of dimensionality

💾 Data Storage

Saves Q-table to:

data/qvalues.pkl

Automatically created.

▶ Run
py pacman.py -p QLearningAgent -l testClassic -n 100 -q

With parameters:

py pacman.py -p QLearningAgent -l testClassic -n 100 -q -a epsilon=0.3,alpha=0.1,gamma=0.9
3️⃣ Approximate Q-Learning Agent
🔹 Why Needed?

Tabular Q-learning fails due to huge state space.

Solution:

Use feature-based representation.

🔹 Approximation Formula
𝑄
(
𝑠
,
𝑎
)
=
𝑤
⋅
𝑓
(
𝑠
,
𝑎
)
Q(s,a)=w⋅f(s,a)

Where:

f(s,a) = feature vector

w = learned weights

🔹 TD Error
𝛿
=
𝑟
+
𝛾
max
⁡
𝑄
(
𝑠
′
,
𝑎
′
)
−
𝑄
(
𝑠
,
𝑎
)
δ=r+γmaxQ(s
′
,a
′
)−Q(s,a)
🔹 Weight Update
𝑤
𝑖
←
𝑤
𝑖
+
𝛼
⋅
𝛿
⋅
𝑓
𝑖
(
𝑠
,
𝑎
)
w
i
	​

←w
i
	​

+α⋅δ⋅f
i
	​

(s,a)

This is gradient descent on TD error.

🔹 Implemented Features

Bias term

Inverse distance to nearest food

Inverse distance to nearest ghost

Remaining food count

These allow generalization across states.

💾 Data Storage

Weights saved to:

data/approx_weights.pkl

Automatically created.

▶ Run

Train 100 episodes (fast mode):

py pacman.py -p ApproxQLearningAgent -l testClassic -n 100 -q

Easier training (1 ghost):

py pacman.py -p ApproxQLearningAgent -l testClassic -k 1 -n 200 -q

Tune parameters:

py pacman.py -p ApproxQLearningAgent -l testClassic -n 200 -q -a epsilon=0.3,alpha=0.05
⚡ Speed Options

Use -q for fastest training (no graphics):

py pacman.py -p ApproxQLearningAgent -l testClassic -n 200 -q

Use -t for text display:

py pacman.py -p ApproxQLearningAgent -l testClassic -n 200 -t
📊 Observations
Agent	Expected Behavior
Random	Very low score, near 0 win rate
Tabular Q	Struggles due to large state space
Approx Q	Positive win rate after training
🧠 Key Concepts Learned

Exploration vs Exploitation

Temporal Difference Learning

Off-policy Learning

Curse of Dimensionality

Function Approximation

Linear Q-function

🚀 Suggested Learning Order

Random Agent

Tabular Q-Learning

Approximate Q-Learning

(Optional) SARSA

(Advanced) Approximate SARSA

Deep Q-Network (Future work)

📌 Important Notes

Use py instead of python on your system.

No need to manually create data/ folder.

Q-values and weights are automatically saved.

Training requires multiple episodes (100–300).

🏁 Example Training Workflow

Train silently:

py pacman.py -p ApproxQLearningAgent -l testClassic -k 1 -n 200 -q

Then test visually:

py pacman.py -p ApproxQLearningAgent -l testClassic