import { useState, useEffect, useRef, useCallback } from 'react'

const API = ''
const WS_URL = (() => {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${protocol}//${window.location.host}/ws/run`
})()

const SPEED_PRESETS = [
  { id: '1x', label: '1x', delay: 0.12 },
  { id: '2x', label: '2x', delay: 0.06 },
  { id: '3x', label: '3x', delay: 0.04 },
  { id: '4x', label: '4x', delay: 0.03 },
  { id: '5x', label: '5x', delay: 0.024 },
  { id: '8x', label: '8x', delay: 0.015 },
  { id: 'max', label: 'Max', delay: 0.01 },
]

const KEY_TO_DIR = {
  ArrowUp: 'North',
  ArrowDown: 'South',
  ArrowLeft: 'West',
  ArrowRight: 'East',
}

export default function App() {
  const [layouts, setLayouts] = useState([])
  const [agents, setAgents] = useState([])
  const [layout, setLayout] = useState('smallClassic')
  const [agent, setAgent] = useState('RandomAgent')
  const [numEpisodes, setNumEpisodes] = useState(1)
  const [agentParams, setAgentParams] = useState({ alpha: 0.05, gamma: 0.9, epsilon: 0.3 })
  const [speedPreset, setSpeedPreset] = useState('2x')
  const [running, setRunning] = useState(false)
  const [error, setError] = useState(null)
  const [gameState, setGameState] = useState(null)
  const [episode, setEpisode] = useState(0)
  const [manualHint, setManualHint] = useState(false)
  const [lastMetrics, setLastMetrics] = useState(null)
  const [activeTab, setActiveTab] = useState('game')
  const [traces, setTraces] = useState([])
  const [latestWeights, setLatestWeights] = useState(null)
  const canvasRef = useRef(null)
  const wsRef = useRef(null)

  const frameDelay = SPEED_PRESETS.find(p => p.id === speedPreset)?.delay ?? 0.06
  const isManual = agent === 'ManualAgent'

  useEffect(() => {
    fetch(API + '/api/layouts').then(r => r.json()).then(d => setLayouts(d.layouts || []))
    fetch(API + '/api/agents').then(r => r.json()).then(d => setAgents(d.agents || []))
  }, [])

  useEffect(() => {
    fetch(API + `/api/layout/${encodeURIComponent(layout)}/preview`)
      .then(r => r.ok ? r.json() : null)
      .then(data => { if (data) setGameState(data) })
      .catch(() => {})
  }, [layout])

  const agentMeta = agents.find(a => a.id === agent)
  const showParams = agentMeta && agentMeta.params && agentMeta.params.length > 0

  const draw = useCallback((state, canvas) => {
    if (!state || !canvas) return
    const ctx = canvas.getContext('2d')
    const w = state.width || 1
    const h = state.height || 1
    const marginX = 24
    const marginTop = 34
    const marginBottom = 24
    const availW = Math.max(1, canvas.width - marginX * 2)
    const availH = Math.max(1, canvas.height - marginTop - marginBottom)

    // keep square cells and center the logical layout inside the inset area
    const cellW = Math.min(availW / w, availH / h)
    const cellH = cellW
    const offsetX = (canvas.width - w * cellW) / 2
    const offsetY = (canvas.height - h * cellH) / 2

    const wallSet = new Set((state.walls || []).map(([x, y]) => `${x},${y}`))
    const isWall = (x, y) => x >= 0 && x < w && y >= 0 && y < h && wallSet.has(`${x},${y}`)

    const toPx = (gx, gy) => ({
      x: offsetX + marginX + gx * cellW,
      y: offsetY + marginTop + (h - 1 - gy) * cellH,
    })

    ctx.fillStyle = '#000005'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    // Draw grid background
    ctx.strokeStyle = '#151c28'
    ctx.lineWidth = 0.5
    for (let i = 0; i <= w; i++) {
      const { x, y } = toPx(i, 0)
      ctx.beginPath()
      ctx.moveTo(x, offsetY)
      ctx.lineTo(x, offsetY + h * cellH)
      ctx.stroke()
    }
    for (let i = 0; i <= h; i++) {
      const { x, y } = toPx(0, i)
      ctx.beginPath()
      ctx.moveTo(offsetX, y)
      ctx.lineTo(offsetX + w * cellW, y)
      ctx.stroke()
    }

    // Walls
    ctx.strokeStyle = '#4a6fa5'
    ctx.lineWidth = Math.max(3, cellW * 0.2)
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'

    const drawSegment = (x1, y1, x2, y2) => {
      const p1 = toPx(x1, y1)
      const p2 = toPx(x2, y2)
      ctx.beginPath()
      ctx.moveTo(p1.x, p1.y)
      ctx.lineTo(p2.x, p2.y)
      ctx.stroke()
    }

    // Horizontal segments
    for (let y = 0; y < h; y++) {
      let start = null
      for (let x = 0; x <= w; x++) {
        const here = x < w && isWall(x, y)
        if (here && start === null) start = x
        if ((!here || x === w) && start !== null) {
          drawSegment(start + 0, y + 0.5, x - 1 + 1, y + 0.5)
          start = null
        }
      }
    }

    // Vertical segments
    for (let x = 0; x < w; x++) {
      let start = null
      for (let y = 0; y <= h; y++) {
        const here = y < h && isWall(x, y)
        if (here && start === null) start = y
        if ((!here || y === h) && start !== null) {
          drawSegment(x + 0.5, start + 0, x + 0.5, y - 1 + 1)
          start = null
        }
      }
    }

    // Food: small glowing yellow dots (image 2 style)
    ctx.shadowColor = '#ffd966'
    ctx.shadowBlur = 6
    ctx.fillStyle = '#ffd966'
    for (const [x, y] of state.food || []) {
      const { x: cx, y: cy } = toPx(x + 0.5, y + 0.5)
      ctx.beginPath()
      ctx.arc(cx, cy, cellW * 0.12, 0, Math.PI * 2)
      ctx.fill()
    }

    // Capsules: large orange glowing circles (image 2 style)
    ctx.fillStyle = '#ff9900'
    ctx.shadowColor = '#ff9900'
    ctx.shadowBlur = 20
    for (const [x, y] of state.capsules || []) {
      const { x: cx, y: cy } = toPx(x + 0.5, y + 0.5)
      ctx.beginPath()
      ctx.arc(cx, cy, cellW * 0.42, 0, Math.PI * 2)
      ctx.fill()
    }
    ctx.shadowBlur = 0
    ctx.shadowColor = 'transparent'

    const agentsList = state.agents || []
    const dirAngle = { North: -0.5 * Math.PI, South: 0.5 * Math.PI, East: 0, West: Math.PI, Stop: 0 }
    agentsList.forEach((a, i) => {
      if (!a.pos) return
      const [gx, gy] = a.pos
      const { x: cx, y: cy } = toPx(gx + 0.5, gy + 0.5)
      const r = cellW * 0.4
      if (a.isPacman) {
        // Pac-Man: bright yellow circle with glow + single black eye dot
        ctx.shadowColor = '#ffdd00'
        ctx.shadowBlur = 18
        ctx.fillStyle = '#ffdd00'
        ctx.beginPath()
        ctx.arc(cx, cy, r, 0, Math.PI * 2)
        ctx.fill()
        ctx.shadowBlur = 0
        // Single small eye
        ctx.fillStyle = '#111'
        ctx.beginPath()
        ctx.arc(cx + r * 0.28, cy - r * 0.32, r * 0.13, 0, Math.PI * 2)
        ctx.fill()
      } else {
        ctx.shadowBlur = 0

        // Ghost colors per index (orange, blue, pink, red)
        const ghostColors = ['#ff7722', '#4488ff', '#ff88cc', '#ff4444']
        const bodyColor = a.scaredTimer > 0 ? '#2244cc' : (ghostColors[(i - 1) % ghostColors.length] || '#ff7722')
        ctx.fillStyle = bodyColor

        // Classic ghost shape: semicircle top + wavy bottom
        const gTop = cy - r * 0.9
        const gBot = cy + r * 0.85
        const gLeft = cx - r * 0.75
        const gRight = cx + r * 0.75
        const gW = r * 1.5
        const numWaves = 3
        const waveW = gW / numWaves

        ctx.beginPath()
        // Rounded top (semicircle)
        ctx.arc(cx, cy - r * 0.1, r * 0.75, Math.PI, 0, false)
        // Right side straight down
        ctx.lineTo(gRight, gBot)
        // Wavy bottom (3 bumps going left)
        for (let w = numWaves - 1; w >= 0; w--) {
          const wx = gLeft + w * waveW
          const wMid = wx + waveW * 0.5
          ctx.quadraticCurveTo(wMid, gBot - r * 0.35, wx, gBot)
        }
        // Left side back up
        ctx.lineTo(gLeft, cy - r * 0.1)
        ctx.closePath()
        ctx.fill()

        // White eye sclera
        const eyeOX = r * 0.27
        const eyeOY = r * 0.05
        const scleraR = r * 0.22
        ctx.fillStyle = '#ffffff'
        ctx.beginPath()
        ctx.arc(cx - eyeOX, cy - eyeOY, scleraR, 0, Math.PI * 2)
        ctx.fill()
        ctx.beginPath()
        ctx.arc(cx + eyeOX, cy - eyeOY, scleraR, 0, Math.PI * 2)
        ctx.fill()

        // Colored pupils
        const pupilColor = a.scaredTimer > 0 ? '#aaccff' : '#1133aa'
        ctx.fillStyle = pupilColor
        ctx.beginPath()
        ctx.arc(cx - eyeOX + r * 0.06, cy - eyeOY + r * 0.05, scleraR * 0.5, 0, Math.PI * 2)
        ctx.fill()
        ctx.beginPath()
        ctx.arc(cx + eyeOX + r * 0.06, cy - eyeOY + r * 0.05, scleraR * 0.5, 0, Math.PI * 2)
        ctx.fill()
      }
    })

    // Score
    ctx.shadowBlur = 0
    ctx.fillStyle = '#fff'
    ctx.font = 'bold 24px "JetBrains Mono", "Courier New", monospace'
    ctx.textAlign = 'center'
    ctx.fillText(`SCORE: ${state.score ?? 0}`, canvas.width / 2, canvas.height - 3)
  }, [])

  useEffect(() => {
    if (!gameState) return
    const canvas = canvasRef.current
    if (canvas) draw(gameState, canvas)
  }, [gameState, draw])

  useEffect(() => {
    if (!isManual || !running || !wsRef.current) return
    const onKeyDown = (e) => {
      const dir = KEY_TO_DIR[e.key]
      if (dir && wsRef.current?.readyState === WebSocket.OPEN) {
        e.preventDefault()
        wsRef.current.send(JSON.stringify({ type: 'action', direction: dir }))
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [isManual, running])

  const run = () => {
    setError(null)
    setEpisode(0)
    setLastMetrics(null)
    setTraces([])
    setLatestWeights(null)
    setRunning(true)
    setManualHint(isManual)
    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    const config = {
      layout,
      agent,
      numEpisodes: isManual ? 1 : numEpisodes,
      frameDelay,
      agentParams: showParams ? agentParams : {},
    }
    ws.onopen = () => ws.send(JSON.stringify(config))
    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data)
      if (msg.type === 'init' && msg.state) setGameState(msg.state)
      if (msg.type === 'frame' && msg.state) setGameState(msg.state)
      if (msg.type === 'trace') {
        setTraces(prev => {
          const next = prev.length > 200 ? prev.slice(prev.length - 200) : prev
          return [...next, msg]
        })
      }
      if (msg.type === 'episode_start') setEpisode(msg.episode)
      if (msg.type === 'episode_end') {
        setEpisode(msg.episode)
        if (msg.weights) setLatestWeights(msg.weights)
      }
      if (msg.type === 'done') {
        setRunning(false)
        setLastMetrics(msg.metrics || null)
      }
      if (msg.type === 'error') {
        setError(msg.message)
        setRunning(false)
      }
    }
    ws.onerror = () => {
      setError('WebSocket error. Is the backend running on port 8000?')
      setRunning(false)
    }
    ws.onclose = () => {
      setRunning(false)
      wsRef.current = null
    }
  }

  return (
    <div className="app">
      <nav className="navbar">
        <div className="nav-left">
          <h1 className="logo">Pacman AI</h1>
          <div className="nav-tabs">
            <button
              className={`nav-tab ${activeTab === 'game' ? 'active' : ''}`}
              onClick={() => setActiveTab('game')}
            >
              Game
            </button>
            <button
              className={`nav-tab ${activeTab === 'stats' ? 'active' : ''}`}
              onClick={() => setActiveTab('stats')}
            >
              Stats
            </button>
          </div>
        </div>

        <div className="nav-right">
          <span className="status-badge">
            {running ? '● Running' : '○ Idle'}
          </span>
        </div>
      </nav>

      {activeTab === 'game' ? (
        <div className="main">
          <aside className="sidebar">

          <div className="config-section">
            <h3>Configuration</h3>
            
            <div className="config-item">
              <label>Map</label>
              <select value={layout} onChange={e => setLayout(e.target.value)} disabled={running}>
                {layouts.map(l => <option key={l} value={l}>{l}</option>)}
              </select>
            </div>

            <div className="config-item">
              <label>Agent</label>
              <select value={agent} onChange={e => setAgent(e.target.value)} disabled={running}>
                {agents.map(a => <option key={a.id} value={a.id}>{a.label}</option>)}
              </select>
            </div>

            {!isManual && (
              <div className="config-item">
                <label>Episodes</label>
                <input
                  type="number"
                  min={1}
                  max={100}
                  value={numEpisodes}
                  onChange={e => setNumEpisodes(Number(e.target.value) || 1)}
                  disabled={running}
                />
              </div>
            )}

            <div className="config-item">
              <label>Speed</label>
              <div className="speed-selector">
                {SPEED_PRESETS.map(p => (
                  <button
                    key={p.id}
                    className={`speed-option ${speedPreset === p.id ? 'active' : ''}`}
                    onClick={() => setSpeedPreset(p.id)}
                    disabled={running}
                  >
                    {p.label}
                  </button>
                ))}
              </div>
            </div>

            {showParams && (
              <>
                <div className="config-item">
                  <label>Parameters</label>
                  <div className="params-grid">
                    {['alpha', 'gamma', 'epsilon'].map(p => (
                      <div key={p} className="param-input">
                        <span>{p}</span>
                        <input
                          type="number"
                          min={0}
                          max={1}
                          step={0.05}
                          value={agentParams[p] ?? 0.1}
                          onChange={e => setAgentParams(prev => ({ ...prev, [p]: Number(e.target.value) }))}
                          disabled={running}
                        />
                      </div>
                    ))}
                  </div>
                </div>
              </>
            )}

            <button 
              className="run-button" 
              onClick={run} 
              disabled={running}
            >
              {running ? (
                <>
                  <span className="spinner"></span>
                  Running...
                </>
              ) : (
                isManual ? '▶ Start Game' : '▶ Run Game'
              )}
            </button>

            {error && <div className="error-message">{error}</div>}
          </div>

          {lastMetrics && (
            <div className="stats-section">
              <h3>Results</h3>
              <div className="stats-grid">
                <div className="stat-item">
                  <span className="stat-label">Win Rate</span>
                  <span className="stat-value highlight">{lastMetrics.winRatio}%</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Wins</span>
                  <span className="stat-value">{lastMetrics.wins}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Losses</span>
                  <span className="stat-value">{lastMetrics.losses}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Avg Score</span>
                  <span className="stat-value">{lastMetrics.avgScore}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Best</span>
                  <span className="stat-value">{lastMetrics.bestScore}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Worst</span>
                  <span className="stat-value">{lastMetrics.worstScore}</span>
                </div>
              </div>
            </div>
          )}
        </aside>

        <main className="game-area">
          <div className="game-container">
            <div className="game-header">
              <div className="game-info">
                <span className="game-label">{layout}</span>
                {episode > 0 && (
                  <span className="game-label">
                    Episode {episode}{numEpisodes > 1 ? ` / ${numEpisodes}` : ''}
                  </span>
                )}
              </div>
              <div className="score-display">
                <span className="score-label">Score</span>
                <span className="score-value">{gameState ? gameState.score : '0'}</span>
              </div>
            </div>

            {manualHint && running && (
              <div className="hint-banner">
                <span className="hint-icon">⌨️</span>
                Use arrow keys to move
              </div>
            )}

            <div className="canvas-wrapper">
              <canvas 
                ref={canvasRef} 
                width={1280} 
                height={720} 
                className="game-canvas"
              />
            </div>
          </div>
        </main>
      </div>
      ) : (
        <div className="stats-full">
          <section className="stats-page">
            <h2 className="stats-title">Live Calculations</h2>
            <p className="stats-subtitle">Action selection, Q-values, TD error and weight updates streamed while the agent runs.</p>

            <div className="stats-grid-full">
              <div className="stats-panel">
                <div className="stats-panel-header">
                  <div className="stats-panel-title">Weights</div>
                  <div className="stats-panel-subtitle">Episode {episode || 0}</div>
                </div>
                <div className="stats-panel-body">
                  {latestWeights ? (
                    <div style={{ overflowX: 'auto' }}>
                      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                        <tbody>
                          {Object.entries(latestWeights).map(([k, v]) => (
                            <tr key={k}>
                              <td style={{ padding: '8px 10px', opacity: 0.8, borderBottom: '1px solid rgba(255,255,255,0.06)' }}>{k}</td>
                              <td style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>{String(v)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <div style={{ opacity: 0.75 }}>Run Approx Q-Learning (or Approx SARSA) to see live weight updates.</div>
                  )}
                </div>
              </div>

              <div className="stats-panel stats-panel-trace">
                <div className="stats-panel-header">
                  <div className="stats-panel-title">Trace</div>
                  <div className="stats-panel-subtitle">Latest events</div>
                </div>
                <div className="stats-panel-body stats-trace-body">
                  {traces.length === 0 ? (
                    <div style={{ opacity: 0.75 }}>No trace yet.</div>
                  ) : (
                    traces.slice(-120).map((t, idx) => (
                      <div key={idx} className="trace-row">
                        {t.event === 'select_action' ? (
                          <>
                            <div className="trace-title">select_action: {t.selected} (ε={t.epsilon})</div>
                            <div className="trace-json">{JSON.stringify(t.q_values)}</div>
                          </>
                        ) : (
                          <>
                            <div className="trace-title">update: a={t.action} r={t.reward} td={t.td_error}</div>
                            <div className="trace-json">{JSON.stringify(t.features)}</div>
                          </>
                        )}
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>
          </section>
        </div>
      )}
    </div>
  )
}