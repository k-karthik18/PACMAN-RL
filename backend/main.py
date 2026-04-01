"""
FastAPI backend: run Pacman games and stream state over WebSocket.
Run from project root: uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
(src is expected at backend/src)
"""
import asyncio
import os
import sys
import queue
import threading
from typing import Optional
import csv

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Ensure backend/src is on path so we can import pacman, layout, etc. (src lives inside backend)
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BACKEND_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

app = FastAPI(title="Pacman AI Web")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Layout names (without .lay) for dropdown
LAYOUTS = [
    "smallClassic",
    "mediumClassic",
    "capsuleClassic",
    "contestClassic",
    "minimaxClassic",
    "openClassic",
    "originalClassic",
    "powerClassic",
    "testClassic",
    "trappedClassic",
    "trickyClassic",
]

# Pacman agents that can be selected
AGENTS = [
    {"id": "RandomAgent", "label": "Random Agent", "params": []},
    {"id": "ManualAgent", "label": "Manual Play", "params": []},
    {"id": "QLearningAgent", "label": "Q-Learning", "params": ["alpha", "gamma", "epsilon"]},
    {"id": "ApproxQLearningAgent", "label": "Approx Q-Learning", "params": ["alpha", "gamma", "epsilon"]},
    {"id": "ApproxSarsaAgent", "label": "Approx Sarsa", "params": ["alpha", "gamma", "epsilon"]},
]


def get_layout_preview(layout_name: str):
    """Return serialized initial state for a layout (for frontend preview)."""
    cwd = os.getcwd()
    try:
        os.chdir(SRC_DIR)
        import layout as layout_mod
        from game import Directions
        lay = layout_mod.getLayout(layout_name)
        if lay is None:
            return None
        walls = []
        for x in range(lay.width):
            for y in range(lay.height):
                if lay.walls[x][y]:
                    walls.append([x, y])
        food = [list(p) for p in lay.food.asList()]
        capsules = [list(c) for c in lay.capsules]
        agents = []
        for is_pacman, pos in lay.agentPositions:
            agents.append({
                "pos": list(pos),
                "direction": Directions.STOP,
                "isPacman": is_pacman,
                "scaredTimer": 0,
            })
        return {
            "walls": walls,
            "width": lay.width,
            "height": lay.height,
            "food": food,
            "capsules": capsules,
            "agents": agents,
            "score": 0,
            "win": False,
            "lose": False,
        }
    finally:
        os.chdir(cwd)


def run_game_thread(
    layout_name: str,
    pacman_agent: str,
    agent_params: dict,
    num_episodes: int,
    frame_delay: float,
    state_queue: queue.Queue,
    trace_queue: Optional[queue.Queue] = None,
    action_queue: Optional[queue.Queue] = None,
    num_ghosts: int = 1,
):
    """Run games in a thread; push state updates to state_queue. For ManualAgent, action_queue must be provided."""
    cwd = os.getcwd()
    try:
        os.chdir(SRC_DIR)
        import layout
        import pacman
        from ghostAgents import RandomGhost
        from webDisplay import WebDisplay

        lay = layout.getLayout(layout_name)
        if lay is None:
            state_queue.put({"type": "error", "message": f"Layout '{layout_name}' not found."})
            return

        if pacman_agent == "ManualAgent":
            from manualAgent import ManualAgent
            if action_queue is None:
                state_queue.put({"type": "error", "message": "Manual play requires action queue."})
                return
            pacman_agent_instance = ManualAgent(action_queue)
        else:
            pacman_type = pacman.loadAgent(pacman_agent, nographics=True)
            opts = {}
            for k, v in (agent_params or {}).items():
                if v is None or v == "":
                    continue
                try:
                    if isinstance(v, (int, float)):
                        opts[k] = v
                    else:
                        opts[k] = float(v) if k in ("alpha", "gamma", "epsilon") else v
                except (ValueError, TypeError):
                    opts[k] = v
            pacman_agent_instance = pacman_type(**opts)

            if trace_queue is not None and pacman_agent == "ApproxQLearningAgent":
                try:
                    from approxQLearningAgents import set_trace_queue
                    set_trace_queue(trace_queue)
                except Exception:
                    pass

            if trace_queue is not None and pacman_agent == "QLearningAgent":
                try:
                    from qLearningAgent import set_trace_queue
                    set_trace_queue(trace_queue)
                except Exception:
                    pass

        ghosts = [RandomGhost(i + 1) for i in range(num_ghosts)]
        display = WebDisplay(state_queue, frame_delay=frame_delay)
        rules = pacman.ClassicGameRules(timeout=30)

        wins = 0
        losses = 0
        scores = []
        for i in range(num_episodes):
            state_queue.put({"type": "episode_start", "episode": i + 1})
            game = rules.newGame(lay, pacman_agent_instance, ghosts, display, quiet=True, catchExceptions=True)
            game.run()
            score = game.state.data.score
            win = getattr(game.state.data, "_win", False)
            lose = getattr(game.state.data, "_lose", False)
            if win:
                wins += 1
            if lose:
                losses += 1
            scores.append(score)
            weights_snapshot = None
            try:
                if pacman_agent == "ApproxQLearningAgent":
                    weights_snapshot = read_weights_csv_latest("approx_weights.csv")
                elif pacman_agent == "ApproxSarsaAgent":
                    weights_snapshot = read_weights_csv_latest("approx_sarsa_weights.csv")
            except Exception:
                weights_snapshot = None

            state_queue.put({
                "type": "episode_end",
                "episode": i + 1,
                "score": score,
                "win": win,
                "lose": lose,
                "weights": weights_snapshot,
            })

        total = num_episodes
        win_ratio = (wins / total * 100) if total else 0
        avg_score = sum(scores) / total if total else 0
        state_queue.put({
            "type": "done",
            "metrics": {
                "totalGames": total,
                "wins": wins,
                "losses": losses,
                "winRatio": round(win_ratio, 1),
                "avgScore": round(avg_score, 1),
                "bestScore": max(scores) if scores else 0,
                "worstScore": min(scores) if scores else 0,
            },
        })
    except Exception as e:
        state_queue.put({"type": "error", "message": str(e)})
        import traceback
        traceback.print_exc()
    finally:
        os.chdir(cwd)


@app.get("/api/layouts")
def list_layouts():
    return {"layouts": LAYOUTS}


@app.get("/api/agents")
def list_agents():
    return {"agents": AGENTS}


@app.get("/api/layout/{layout_name}/preview")
def layout_preview(layout_name: str):
    """Return initial game state for the layout so the frontend can show the map before Run."""
    preview = get_layout_preview(layout_name)
    if preview is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Layout '{layout_name}' not found.")
    return preview


def _read_csv_rows(csv_path: str):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def read_weights_csv_latest(filename: str):
    csv_path = os.path.join(SRC_DIR, "data", filename)
    if not os.path.exists(csv_path):
        return None
    rows = _read_csv_rows(csv_path)
    if not rows:
        return None
    return rows[-1]


@app.get("/api/weights")
def list_weights_files():
    data_dir = os.path.join(SRC_DIR, "data")
    files = []
    if os.path.isdir(data_dir):
        for name in os.listdir(data_dir):
            if name.endswith(".csv"):
                files.append(name)
    files.sort()
    return {"files": files}


@app.get("/api/weights/{filename}")
def get_weights_file(filename: str):
    safe = os.path.basename(filename)
    csv_path = os.path.join(SRC_DIR, "data", safe)
    from fastapi import HTTPException
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="weights file not found")
    return {"rows": _read_csv_rows(csv_path)}


@app.websocket("/ws/run")
async def websocket_run(websocket: WebSocket):
    await websocket.accept()
    try:
        config = await websocket.receive_json()
    except Exception:
        await websocket.close(code=4000)
        return

    layout_name = config.get("layout", "smallClassic")
    pacman_agent = config.get("agent", "RandomAgent")
    agent_params = config.get("agentParams") or {}
    num_episodes = max(1, min(int(config.get("numEpisodes", 1)), 1000))
    num_ghosts = max(1, min(int(config.get("numGhosts", 1)), 4))
    frame_delay = max(0.01, min(float(config.get("frameDelay", 0.08)), 0.5))
    is_manual = pacman_agent == "ManualAgent"
    if is_manual:
        num_episodes = 1

    state_queue = queue.Queue()
    trace_queue = queue.Queue()
    action_queue = queue.Queue() if is_manual else None
    loop = asyncio.get_event_loop()

    def run():
        run_game_thread(
            layout_name, pacman_agent, agent_params, num_episodes, frame_delay,
            state_queue, trace_queue=trace_queue, action_queue=action_queue,
            num_ghosts=num_ghosts,
        )

    thread = threading.Thread(target=run)
    thread.start()

    async def receive_actions():
        while thread.is_alive():
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=0.5)
                data = __import__("json").loads(raw)
                if data.get("type") == "action" and data.get("direction") and action_queue is not None:
                    action_queue.put(data["direction"])
            except asyncio.TimeoutError:
                pass
            except WebSocketDisconnect:
                break
            except Exception:
                pass

    if is_manual:
        recv_task = asyncio.create_task(receive_actions())

    try:
        while True:
            try:
                while True:
                    tmsg = trace_queue.get_nowait()
                    await websocket.send_json(tmsg)
            except queue.Empty:
                pass
            try:
                msg = await loop.run_in_executor(None, lambda: state_queue.get(timeout=0.1))
            except queue.Empty:
                if not thread.is_alive():
                    break
                await asyncio.sleep(0.02)
                continue
            if msg.get("type") == "done" or msg.get("type") == "error":
                await websocket.send_json(msg)
                if msg.get("type") == "error":
                    break
                break
            await websocket.send_json(msg)
    except WebSocketDisconnect:
        pass
    if is_manual:
        recv_task.cancel()
        try:
            await recv_task
        except asyncio.CancelledError:
            pass
    thread.join(timeout=1.0)
