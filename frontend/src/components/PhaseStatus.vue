<script setup lang="ts">
import type { PhaseEvent } from '../api'

defineProps<{
  phaseOrder: string[]
  phaseLabels: Record<string, string>
  events: PhaseEvent[]
}>()

function statusFor(events: PhaseEvent[], phase: string): string {
  const phaseEvents = events.filter((e) => e.phase === phase)
  if (!phaseEvents.length) return 'pending'
  const last = phaseEvents[phaseEvents.length - 1]
  if (last.event === 'completed') return 'done'
  if (last.event === 'failed') return 'failed'
  return 'running'
}
</script>

<template>
  <div class="phases">
    <div
      v-for="phase in phaseOrder"
      :key="phase"
      class="phase"
      :class="statusFor(events, phase)"
    >
      <span class="dot" />
      <span class="label">{{ phaseLabels[phase] ?? phase }}</span>
      <span class="status-text">{{ statusFor(events, phase) }}</span>
    </div>
  </div>
</template>

<style scoped>
.phases { display: flex; flex-direction: column; gap: 0.5rem; }
.phase { display: flex; align-items: center; gap: 0.75rem; padding: 0.6rem 1rem; border-radius: 6px; background: #1e1e1e; border: 1px solid #333; }
.dot { width: 10px; height: 10px; border-radius: 50%; background: #444; flex-shrink: 0; }
.phase.running .dot { background: #3b82f6; animation: pulse 1.2s infinite; }
.phase.done .dot { background: #22c55e; }
.phase.failed .dot { background: #ef4444; }
.label { flex: 1; font-size: 0.95rem; }
.status-text { font-size: 0.8rem; color: #777; }
.phase.running .status-text { color: #3b82f6; }
.phase.done .status-text { color: #22c55e; }
.phase.failed .status-text { color: #ef4444; }
@keyframes pulse { 0%,100% { opacity:1 } 50% { opacity:0.4 } }
</style>
