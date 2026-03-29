const heatmapCanvas = document.getElementById("heatmap-canvas")
const heatCtx       = heatmapCanvas.getContext("2d")

let currentHeatmap = null

function updateHeatmap(grid) {
    currentHeatmap = grid
    drawHeatmap()
}

function drawHeatmap() {
    if (!currentHeatmap) return

    const rows    = currentHeatmap.length
    const cols    = currentHeatmap[0].length
    const cellW   = heatmapCanvas.width  / cols
    const cellH   = heatmapCanvas.height / rows

    // Flatten to find min/max for normalisation
    const flat = currentHeatmap.flat()
    const mn   = Math.min(...flat)
    const mx   = Math.max(...flat)
    const range = mx - mn || 1

    heatCtx.clearRect(0, 0, heatmapCanvas.width, heatmapCanvas.height)

    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const t     = (currentHeatmap[r][c] - mn) / range  // 0..1
            const color = heatmapColor(t)
            heatCtx.fillStyle = color
            heatCtx.fillRect(c * cellW, r * cellH, cellW, cellH)
        }
    }
}

function heatmapColor(t) {
    // Purple (low) -> black (mid) -> amber (high)
    const r = Math.round(t > 0.5 ? (t - 0.5) * 2 * 220 : 80 * (1 - t * 2))
    const g = Math.round(t > 0.5 ? (t - 0.5) * 2 * 140 : 0)
    const b = Math.round(t < 0.5 ? (0.5 - t) * 2 * 180 : 0)
    return `rgba(${r},${g},${b},0.55)`
}