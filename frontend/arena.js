const arenaCanvas = document.getElementById("arena-canvas")
const ctx         = arenaCanvas.getContext("2d")

const statReward  = document.getElementById("stat-reward")
const statCatches = document.getElementById("stat-catches")
const statPrey    = document.getElementById("stat-prey")
const statSteps   = document.getElementById("stat-steps")
const ckptLabel   = document.getElementById("checkpoint")

const AGENT_RADIUS = 16
const PREY_RADIUS  = 10

let epCatches  = 0
let lastSteps  = 0
let socket     = null

function connect() {
    socket = new WebSocket("ws://localhost:8000/ws")

    socket.onmessage = (event) => {
        const msg = JSON.parse(event.data)

        if (msg.type === "info") {
            ckptLabel.textContent = "checkpoint: " + msg.checkpoint
            return
        }

        if (msg.type === "state") {
            if (msg.game.steps < lastSteps) epCatches = 0
            lastSteps  = msg.game.steps

            render(msg.game)
            updateStats(msg)

            if (msg.heatmap) updateHeatmap(msg.heatmap)
        }
    }

    socket.onclose = () => {
        setTimeout(connect, 2000)
    }
}

function render(game) {
    ctx.clearRect(0, 0, arenaCanvas.width, arenaCanvas.height)

    ctx.fillStyle = "rgba(15, 15, 15, 0.85)"
    ctx.fillRect(0, 0, arenaCanvas.width, arenaCanvas.height)

    for (const p of game.prey) {
        ctx.beginPath()
        ctx.arc(p.x, p.y, PREY_RADIUS, 0, Math.PI * 2)
        ctx.fillStyle   = "#4fc3f7"
        ctx.shadowColor = "#4fc3f7"
        ctx.shadowBlur  = 8
        ctx.fill()
        ctx.shadowBlur  = 0
    }

    ctx.beginPath()
    ctx.arc(game.agent.x, game.agent.y, AGENT_RADIUS, 0, Math.PI * 2)
    ctx.fillStyle   = "#ef5350"
    ctx.shadowColor = "#ef5350"
    ctx.shadowBlur  = 16
    ctx.fill()
    ctx.shadowBlur  = 0
}

function updateStats(msg) {
    epCatches += msg.catches
    statReward.textContent  = msg.reward.toFixed(2)
    statCatches.textContent = epCatches
    statPrey.textContent    = msg.game.prey_remaining
    statSteps.textContent   = msg.game.steps
}

connect()