<script setup lang="ts">
import { ref } from 'vue'

const props = defineProps<{ port?: number }>()
const messages = ref<{ role: 'user' | 'model'; text: string }[]>([])
const input = ref('')
const sending = ref(false)
const error = ref('')

// Port comes from query param since we can't easily pass it through route params
import { useRoute } from 'vue-router'
const route = useRoute()
const chatPort = props.port ?? Number(route.query.port)

async function send() {
  const text = input.value.trim()
  if (!text || sending.value) return
  messages.value.push({ role: 'user', text })
  input.value = ''
  sending.value = true
  error.value = ''
  try {
    const resp = await fetch(`http://localhost:${chatPort}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text }),
    })
    const data = await resp.json() as { reply: string }
    messages.value.push({ role: 'model', text: data.reply })
  } catch (e) {
    error.value = String(e)
  } finally {
    sending.value = false
  }
}
</script>

<template>
  <div class="chat">
    <h1>Chat</h1>
    <p v-if="!chatPort" class="error">No serve port — start serving first.</p>
    <div class="messages">
      <div v-for="(m, i) in messages" :key="i" class="message" :class="m.role">
        <span class="role">{{ m.role === 'user' ? 'You' : 'Model' }}</span>
        <span class="text">{{ m.text }}</span>
      </div>
    </div>
    <p v-if="error" class="error">{{ error }}</p>
    <div class="input-row">
      <input v-model="input" placeholder="Type a message…" @keydown.enter="send" :disabled="!chatPort" />
      <button @click="send" :disabled="!chatPort || sending">Send</button>
    </div>
  </div>
</template>

<style scoped>
h1 { margin-bottom: 1rem; }
.messages { display: flex; flex-direction: column; gap: 0.75rem; margin-bottom: 1rem; min-height: 200px; }
.message { display: flex; gap: 0.75rem; }
.role { min-width: 60px; font-weight: 600; font-size: 0.85rem; color: #888; padding-top: 0.1rem; }
.message.user .role { color: #7dd3fc; }
.message.model .role { color: #86efac; }
.text { background: #1e1e1e; border-radius: 6px; padding: 0.5rem 0.75rem; flex: 1; font-size: 0.9rem; }
.input-row { display: flex; gap: 0.75rem; }
input { flex: 1; background: #111; border: 1px solid #444; color: #e0e0e0; border-radius: 6px; padding: 0.5rem 0.75rem; font-size: 0.95rem; }
button { background: #2563eb; color: #fff; border: none; border-radius: 6px; padding: 0.5rem 1.25rem; cursor: pointer; }
button:disabled { opacity: 0.4; cursor: default; }
.error { color: #f66; margin-bottom: 0.75rem; }
</style>
