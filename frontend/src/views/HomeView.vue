<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { api, type Project } from '../api'

const router = useRouter()
const projects = ref<Project[]>([])
const error = ref('')

async function load() {
  try {
    projects.value = await api.projects.list()
  } catch (e) {
    error.value = String(e)
  }
}

async function deleteProject(id: string) {
  if (!confirm('Delete project?')) return
  await api.projects.delete(id)
  await load()
}

onMounted(load)
</script>

<template>
  <div>
    <h1>Projects</h1>
    <p v-if="error" class="error">{{ error }}</p>
    <p v-if="!projects.length && !error" class="muted">No projects yet. <RouterLink to="/new">Create one.</RouterLink></p>
    <div class="project-list">
      <div v-for="p in projects" :key="p.id" class="project-card" @click="router.push(`/projects/${p.id}`)">
        <div class="project-name">{{ p.name }}</div>
        <div class="project-meta">{{ p.preset_name }}</div>
        <button class="btn-danger" @click.stop="deleteProject(p.id)">Delete</button>
      </div>
    </div>
  </div>
</template>

<style scoped>
h1 { margin-bottom: 1rem; }
.project-list { display: flex; flex-direction: column; gap: 0.75rem; }
.project-card {
  background: #1e1e1e; border: 1px solid #333; border-radius: 6px;
  padding: 1rem 1.25rem; cursor: pointer; display: flex; align-items: center; gap: 1rem;
}
.project-card:hover { border-color: #555; }
.project-name { font-weight: 600; flex: 1; }
.project-meta { color: #888; font-size: 0.85rem; }
.btn-danger { background: #7a1c1c; color: #fff; border: none; border-radius: 4px; padding: 0.3rem 0.75rem; cursor: pointer; }
.btn-danger:hover { background: #a02020; }
.error { color: #f66; margin-bottom: 1rem; }
.muted { color: #777; }
</style>
