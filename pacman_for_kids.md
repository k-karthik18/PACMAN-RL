# 🕹️ Pacman's Brain: Explained for a 10-Year-Old!

Hey there! Have you ever wondered how video game characters know what to do without a human controlling them? In our project, we are teaching Pacman how to play his own game completely by himself! Instead of you using a controller, we've given Pacman a "digital brain".

Let's break down how this whole project works, simply and playfully!

---

## 🏰 The World (How the Project is Built)
This project is split into two halves that talk to each other:

1. **The TV Screen (Frontend):** We built this using something called **React**. This is the part you can actually see in your browser! It draws the maze, colors the ghosts, and shows you Pacman moving around so you can watch him learn. 
2. **The Referee (Backend):** We built this using **Python**. It's the invisible game engine. It knows where all the walls are, keeps track of the score, and decides if Pacman gets eaten. It constantly sends updates to the "TV Screen" so you can watch!

---

## 🧠 Pacman's Brains (The AI Files)

Instead of giving Pacman just one brain, we made a few different ones to see which is smartest!

### 1. The "Random" Brain (`randomAgent.py`)
This Pacman is super silly. On every single step, he rolls a dice to decide where to go. He walks in circles, bumps into corners, and usually walks right into a ghost! *SPLAT!* He never learns anything.

### 2. The "Memorizer" Brain (`qLearningAgent.py`)
This Pacman tries to memorize every single spot on the map! He keeps a giant diary. He writes down: *"If I am exactly in the top-left corner, and the red ghost is 4 steps away, I should go down!"* 
*   **The Problem:** The game has too many possibilities! The diary gets to be thousands of pages long. If he plays on a brand new map he hasn't memorized yet, he gets confused and loses.

### 3. The "Smart" Brain (`approxQLearningAgents.py` & `approxSarsaAgents.py`)
This is where Pacman gets really smart. Instead of memorizing specific spots on the map, he learns **general rules**. He learns that no matter where he is:
*   "Moving closer to ghosts is bad."
*   "Moving closer to food is good."

He learns this by playing hundreds of games and updating his "weights". A weight is just his opinion on something. If he runs into a ghost, his opinion of ghosts becomes super negative. Next game, he will run away!

### 4. His Magic Glasses (`feature_extraction.py`)
Pacman doesn't actually have eyes. So how does he know where food is? This file acts like his magic glasses! It translates the giant, messy game board into simple numbers his brain can use. It whispers to him:
*   *"The nearest dot is exactly 3 steps away."*
*   *"DANGER! A ghost is right behind you!"*

Without this file, the "Smart" brains wouldn't be able to see anything!

---

## 🏆 The Super Brain: `testAgents.py` (Why it's so good!)

You might have noticed a very special file called `testAgents.py`. If you run it, it wins over 90% of the time! You might be wondering: *Did this brain practice for millions of hours to get that good?*

**The Big Secret:** It actually *didn't learn at all!*

While the other "Smart" brains have to play hundreds of games, dying over and over to slowly figure out the rules of the world... the `testAgents` brain is like giving Pacman a **cheat sheet!**

Instead of using AI (Artificial Intelligence) to learn, we (the programmers) just hardcoded the perfect instructions directly into his brain:
1. *"Look at all the directions you can walk."*
2. *"If a direction puts you right next to a ghost, subtract 10,000 points from that idea! Don't do it!"*
3. *"If a direction gets you closer to food, add 10 points to that idea!"*
4. *"Just do whichever idea has the most points!"*

Because we gave him the exact answers before he even started playing, he doesn't need to practice. He just reads our cheat sheet on every single step, dodges the ghosts perfectly, and eats all the food! 

It proves a really cool lesson in computer science: Even though Machine Learning (letting the computer figure it out) is amazing, sometimes just giving the computer the exact, strict rules (a "Reflex Agent") is the fastest way to build a winner!
