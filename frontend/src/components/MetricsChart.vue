<script setup lang="ts">
import { ref, watch, onMounted, onUnmounted } from 'vue'
import { Chart, LineElement, PointElement, LinearScale, CategoryScale, Legend, Tooltip } from 'chart.js'
import type { Metric } from '../api'

Chart.register(LineElement, PointElement, LinearScale, CategoryScale, Legend, Tooltip)

const props = defineProps<{
  metrics: Metric[]
  names: string[]
  title: string
}>()

const canvas = ref<HTMLCanvasElement | null>(null)
let chart: Chart | null = null

const COLORS = ['#3b82f6', '#22c55e', '#f59e0b', '#ef4444', '#a855f7', '#06b6d4']

function buildData() {
  const datasets = props.names.map((name, i) => {
    const points = props.metrics.filter((m) => m.name === name)
    return {
      label: name,
      data: points.map((m) => m.value),
      borderColor: COLORS[i % COLORS.length],
      backgroundColor: 'transparent',
      pointRadius: 0,
      tension: 0.2,
    }
  })
  const steps = props.metrics
    .filter((m) => m.name === props.names[0])
    .map((m) => m.step)
  return { labels: steps, datasets }
}

function render() {
  if (!canvas.value) return
  const data = buildData()
  if (chart) {
    chart.data = data
    chart.update()
    return
  }
  chart = new Chart(canvas.value, {
    type: 'line',
    data,
    options: {
      animation: false,
      responsive: true,
      plugins: { legend: { labels: { color: '#ccc' } }, tooltip: {} },
      scales: {
        x: { ticks: { color: '#888' }, grid: { color: '#222' } },
        y: { ticks: { color: '#888' }, grid: { color: '#222' } },
      },
    },
  })
}

onMounted(render)
watch(() => props.metrics, render, { deep: true })
onUnmounted(() => { chart?.destroy() })
</script>

<template>
  <div class="chart-wrap">
    <div class="chart-title">{{ title }}</div>
    <canvas ref="canvas" />
  </div>
</template>

<style scoped>
.chart-wrap { background: #1e1e1e; border: 1px solid #333; border-radius: 6px; padding: 1rem; }
.chart-title { font-size: 0.85rem; color: #888; margin-bottom: 0.5rem; }
</style>
