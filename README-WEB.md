# Pacman AI – Web UI

Single-page web app: choose map, agent, episodes, and agent parameters (alpha, gamma, epsilon for Q-Learning / Approx Sarsa), then run and watch Pacman in the browser.

## Stack

- **Frontend:** React + Vite (single page, white/black/navy theme)
- **Backend:** Python FastAPI (runs game and streams state over WebSocket)

## Run

1. **Backend** (from project root `Pacman-AI`; `src` must live at `backend/src`):

   ```bash
   cd Pacman-AI
   pip install -r backend/requirements.txt
   uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Frontend**:

   ```bash
   cd frontend
   npm install
   npm run dev
   ```

   Open http://localhost:5173. The dev server proxies `/api` and `/ws` to the backend on port 8000.

3. **Run a game:** Pick layout, agent, number of episodes (and agent params if needed), then click **Run**. The game view updates live; score and episode are shown above the canvas.

## Agent parameters

- **Random Agent:** No parameters.
- **Q-Learning / Approx Q-Learning / Approx Sarsa:** Alpha, gamma, and epsilon are configurable in the form and sent to the backend; they are no longer fixed in code.
