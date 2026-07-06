<script setup lang="ts">
import { ref, watch, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { api, type DescribeOutput, type CompatResult } from '../api'
import ConfigForm from '../components/ConfigForm.vue'

const router = useRouter()
const presets = ref<string[]>([])
const selectedPreset = ref('')
const describe = ref<DescribeOutput | null>(null)
const compat = ref<CompatResult | null>(null)
const config = ref<Record<string, unknown>>({})
const projectName = ref('')
const error = ref('')
const loading = ref(false)

onMounted(async () => {
  presets.value = await api.presets.list()
  if (presets.value.length) selectedPreset.value = presets.value[0]
})

watch(selectedPreset, async (name) => {
  describe.value = null
  compat.value = null
  config.value = {}
  if (!name) return
  loading.value = true
  try {
    describe.value = await api.presets.describe(name)
    compat.value = await api.presets.checkCompat(name)
  } catch (e) {
    error.value = String(e)
  } finally {
    loading.value = false
  }
})

async function create() {
  if (!projectName.value.trim()) { error.value = 'Name required'; return }
  if (!selectedPreset.value) { error.value = 'Select a preset'; return }
  try {
    const p = await api.projects.create({
      name: projectName.value.trim(),
      preset_name: selectedPreset.value,
      config: config.value,
    })
    router.push(`/projects/${p.id}`)
  } catch (e) {
    error.value = String(e)
  }
}
</script>

<template>
  <div>
    <h1>New Project</h1>
    <p v-if="error" class="error">{{ error }}</p>

    <div class="field">
      <label>Name</label>
      <input v-model="projectName" placeholder="my-llm" />
    </div>

    <div class="field">
      <label>Preset</label>
      <select v-model="selectedPreset">
        <option v-for="p in presets" :key="p" :value="p">{{ p }}</option>
      </select>
    </div>

    <div v-if="compat" class="compat" :class="compat.compatible ? 'ok' : 'fail'">
      {{ compat.compatible ? '✓' : '✗' }} {{ compat.details }}
    </div>

    <div v-if="loading" class="muted">Loading preset info…</div>

    <ConfigForm
      v-if="describe"
      :schema="describe.config_schema"
      v-model="config"
    />

    <button class="btn-primary" :disabled="!compat?.compatible" @click="create">
      Create & Configure
    </button>
  </div>
</template>

<style scoped>
h1 { margin-bottom: 1.5rem; }
.field { display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem; }
label { min-width: 80px; }
input, select { background: #111; border: 1px solid #444; color: #e0e0e0; border-radius: 4px; padding: 0.4rem 0.6rem; }
.compat { padding: 0.5rem 0.75rem; border-radius: 4px; margin-bottom: 1rem; font-size: 0.9rem; }
.compat.ok { background: #1a3a1a; color: #6f6; border: 1px solid #2a6a2a; }
.compat.fail { background: #3a1a1a; color: #f66; border: 1px solid #6a2a2a; }
.btn-primary { margin-top: 1.5rem; background: #2563eb; color: #fff; border: none; border-radius: 6px; padding: 0.6rem 1.5rem; cursor: pointer; font-size: 1rem; }
.btn-primary:disabled { opacity: 0.4; cursor: default; }
.btn-primary:not(:disabled):hover { background: #1d4ed8; }
.error { color: #f66; margin-bottom: 1rem; }
.muted { color: #777; }
</style>
