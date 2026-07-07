<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRoute } from 'vue-router'
import { api, type Project, type Run, type Metric, type DescribeOutput } from '../api'
import PhaseStatus from '../components/PhaseStatus.vue'
import MetricsChart from '../components/MetricsChart.vue'

const route = useRoute()
const projectId = route.params.projectId as string

const project = ref<Project | null>(null)
const runs = ref<Run[]>([])
const metrics = ref<Metric[]>([])
const lastRowId = ref(0)
const logs = ref('')
const describe = ref<DescribeOutput | null>(null)
const error = ref('')
const servePort = ref<number | null>(null)

let pollTimer: ReturnType<typeof setInterval> | null = null

const latestRun = computed(() => runs.value[0] ?? null)
const isRunning = computed(() => latestRun.value?.status === 'running')

const phaseLabels = computed<Record<string, string>>(() => {
  if (!describe.value) return {}
  return Object.fromEntries(describe.value.phases.map((p) => [p.name, p.label]))
})

const phaseOrder = computed(() => describe.value?.phase_order ?? [])

async function load() {
  try {
    project.value = await api.projects.get(projectId)
    runs.value = await api.runs.list(projectId)
    if (project.value && !describe.value) {
      describe.value = await api.presets.describe(project.value.preset_name)
    }
  } catch (e) {
    error.value = String(e)
  }
}

async function loadMetrics() {
  const run = latestRun.value
  if (!run) return
  const newMetrics = await api.runs.metrics(projectId, run.id, { since_rowid: lastRowId.value })
  if (newMetrics.length) {
    lastRowId.value = Math.max(...newMetrics.map((m) => m.rowid))
    metrics.value = [...metrics.value, ...newMetrics]
  }
  const logResp = await api.runs.logs(projectId, run.id)
  logs.value = logResp.content
}

async function startRun() {
  try {
    const run = await api.runs.start(projectId)
    runs.value = [run, ...runs.value]
    metrics.value = []
    lastRowId.value = 0
    startPolling()
  } catch (e) {
    error.value = String(e)
  }
}

async function stopRun() {
  if (!latestRun.value) return
  await api.runs.stop(projectId, latestRun.value.id)
  await load()
  stopPolling()
}

async function startServe() {
  if (!latestRun.value) return
  const info = await api.runs.startServe(projectId, latestRun.value.id)
  servePort.value = info.port
}

function startPolling() {
  if (pollTimer) return
  pollTimer = setInterval(async () => {
    await load()
    await loadMetrics()
    if (!isRunning.value) stopPolling()
  }, 5000)
}

function stopPolling() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null }
}

onMounted(async () => {
  await load()
  await loadMetrics()
  if (isRunning.value) startPolling()
})
onUnmounted(stopPolling)

const lossMetrics = computed(() => metrics.value.filter((m) => m.name === 'loss' || m.name === 'bpb'))
const perfMetrics = computed(() => metrics.value.filter((m) => m.name === 'mfu' || m.name === 'gpu_power_w'))
</script>

<template>
  <div>
    <p v-if="error" class="error">{{ error }}</p>
    <template v-if="project">
      <div class="header">
        <h1>{{ project.name }}</h1>
        <span class="preset-badge">{{ project.preset_name }}</span>
      </div>

      <div class="actions">
        <button class="btn-primary" :disabled="isRunning" @click="startRun">
          {{ latestRun ? 'Re-run' : 'Start Run' }}
        </button>
        <button v-if="isRunning" class="btn-danger" @click="stopRun">Stop</button>
        <button
          v-if="latestRun?.status === 'done' && describe?.can_serve"
          class="btn-secondary"
          @click="startServe"
        >
          Serve Model
        </button>
        <RouterLink
          v-if="servePort && latestRun"
          :to="`/projects/${projectId}/runs/${latestRun.id}/chat`"
          class="btn-secondary"
        >
          Open Chat (port {{ servePort }})
        </RouterLink>
      </div>

      <div v-if="latestRun" class="run-info">
        <span class="run-status" :class="latestRun.status">{{ latestRun.status }}</span>
        <span class="muted">run {{ latestRun.id.slice(0, 8) }}</span>
      </div>

      <template v-if="latestRun && phaseOrder.length">
        <h2>Phases</h2>
        <PhaseStatus
          :phase-order="phaseOrder"
          :phase-labels="phaseLabels"
          :events="latestRun.phase_events"
        />
      </template>

      <template v-if="lossMetrics.length">
        <h2>Loss</h2>
        <MetricsChart :metrics="lossMetrics" :names="['loss', 'bpb']" title="Loss / BPB" />
      </template>

      <template v-if="perfMetrics.length">
        <h2>Performance</h2>
        <MetricsChart :metrics="perfMetrics" :names="['mfu', 'gpu_power_w']" title="MFU / GPU Power" />
      </template>

      <template v-if="logs">
        <h2>Logs</h2>
        <pre class="logs">{{ logs }}</pre>
      </template>
    </template>
  </div>
</template>

<style scoped>
h1 { font-size: 1.5rem; }
h2 { margin: 1.5rem 0 0.75rem; font-size: 1rem; color: #aaa; text-transform: uppercase; letter-spacing: 0.05em; }
.header { display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem; }
.preset-badge { background: #1e3a5f; color: #7dd3fc; font-size: 0.8rem; padding: 0.2rem 0.6rem; border-radius: 999px; }
.actions { display: flex; gap: 0.75rem; margin-bottom: 1rem; flex-wrap: wrap; }
.btn-primary { background: #2563eb; color: #fff; border: none; border-radius: 6px; padding: 0.5rem 1.25rem; cursor: pointer; }
.btn-primary:disabled { opacity: 0.4; cursor: default; }
.btn-primary:not(:disabled):hover { background: #1d4ed8; }
.btn-danger { background: #7a1c1c; color: #fff; border: none; border-radius: 6px; padding: 0.5rem 1.25rem; cursor: pointer; }
.btn-danger:hover { background: #a02020; }
.btn-secondary { background: #1e1e1e; color: #e0e0e0; border: 1px solid #444; border-radius: 6px; padding: 0.5rem 1.25rem; cursor: pointer; text-decoration: none; display: inline-block; }
.btn-secondary:hover { border-color: #888; }
.run-info { display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem; }
.run-status { font-size: 0.85rem; padding: 0.2rem 0.6rem; border-radius: 999px; background: #333; }
.run-status.running { background: #1e3a5f; color: #7dd3fc; }
.run-status.done { background: #14532d; color: #86efac; }
.run-status.failed { background: #450a0a; color: #fca5a5; }
.run-status.stopped { background: #3a3a1a; color: #fde68a; }
.muted { color: #666; font-size: 0.85rem; }
.logs { background: #111; border: 1px solid #333; border-radius: 6px; padding: 1rem; font-size: 0.8rem; overflow-x: auto; white-space: pre-wrap; max-height: 300px; overflow-y: auto; color: #aaa; }
.error { color: #f66; margin-bottom: 1rem; }
</style>
