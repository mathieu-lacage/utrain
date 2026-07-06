<script setup lang="ts">
import type { ConfigSchema, FieldSchema } from '../api'

const props = defineProps<{ schema: ConfigSchema; modelValue: Record<string, unknown> }>()
const emit = defineEmits<{ 'update:modelValue': [value: Record<string, unknown>] }>()

function set(key: string, value: unknown) {
  emit('update:modelValue', { ...props.modelValue, [key]: value })
}

function inputType(f: FieldSchema): string {
  if (f.type === 'bool') return 'checkbox'
  if (f.type === 'int' || f.type === 'float') return 'number'
  return 'text'
}

function val(f: FieldSchema): unknown {
  const v = props.modelValue[f.key]
  return v !== undefined ? v : f.default
}
</script>

<template>
  <div>
    <template v-if="schema.globals.groups.length">
      <h3>Global</h3>
      <div v-for="group in schema.globals.groups" :key="group.name" class="group">
        <h4>{{ group.label }}</h4>
        <div v-for="field in group.fields" :key="field.key" class="field">
          <label :title="field.description">{{ field.label }}</label>
          <select v-if="field.type === 'enum'" :value="val(field)" @change="set(field.key, ($event.target as HTMLSelectElement).value)">
            <option v-for="opt in field.options" :key="opt" :value="opt">{{ opt }}</option>
          </select>
          <input
            v-else
            :type="inputType(field)"
            :value="val(field)"
            :min="field.min"
            :max="field.max"
            :step="field.type === 'float' ? 'any' : undefined"
            @input="set(field.key, field.type === 'bool'
              ? ($event.target as HTMLInputElement).checked
              : field.type === 'int'
                ? parseInt(($event.target as HTMLInputElement).value)
                : field.type === 'float'
                  ? parseFloat(($event.target as HTMLInputElement).value)
                  : ($event.target as HTMLInputElement).value
            )"
          />
        </div>
      </div>
    </template>
    <template v-for="(phaseSchema, phaseName) in schema.phases" :key="phaseName">
      <h3>{{ phaseName }}</h3>
      <div v-for="group in phaseSchema.groups" :key="group.name" class="group">
        <h4>{{ group.label }}</h4>
        <div v-for="field in group.fields" :key="phaseName + '.' + field.key" class="field">
          <label :title="field.description">{{ field.label }}</label>
          <input
            :type="inputType(field)"
            :value="val({ ...field, key: phaseName + '.' + field.key })"
            :min="field.min"
            :max="field.max"
            :step="field.type === 'float' ? 'any' : undefined"
            @input="set(phaseName + '.' + field.key, field.type === 'int'
              ? parseInt(($event.target as HTMLInputElement).value)
              : field.type === 'float'
                ? parseFloat(($event.target as HTMLInputElement).value)
                : ($event.target as HTMLInputElement).value
            )"
          />
        </div>
      </div>
    </template>
  </div>
</template>

<style scoped>
h3 { margin: 1.25rem 0 0.5rem; color: #aaa; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }
h4 { margin-bottom: 0.5rem; font-size: 0.95rem; color: #ccc; }
.group { background: #1e1e1e; border: 1px solid #333; border-radius: 6px; padding: 1rem; margin-bottom: 0.75rem; }
.field { display: flex; align-items: center; gap: 1rem; margin-bottom: 0.5rem; }
label { min-width: 140px; font-size: 0.9rem; color: #bbb; cursor: help; }
input, select { background: #111; border: 1px solid #444; color: #e0e0e0; border-radius: 4px; padding: 0.3rem 0.5rem; font-size: 0.9rem; }
input[type="checkbox"] { width: 1rem; height: 1rem; }
</style>
