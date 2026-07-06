export interface Project {
  id: string
  name: string
  preset_name: string
  config: Record<string, unknown>
  created_at: number
}

export interface PhaseEvent {
  phase: string
  event: string
  timestamp: number
}

export interface Run {
  id: string
  project_id: string
  run_dir: string
  status: string
  pid: number | null
  started_at: number | null
  ended_at: number | null
  phase_events: PhaseEvent[]
}

export interface Metric {
  step: number
  timestamp: number
  phase: string
  name: string
  value: number
}

export interface DescribeOutput {
  name: string
  version: string
  phases: { name: string; label: string }[]
  phase_order: string[]
  config_schema: ConfigSchema
  can_serve: boolean
}

export interface FieldSchema {
  key: string
  label: string
  type: 'int' | 'float' | 'str' | 'bool' | 'enum'
  default: number | string | boolean | null
  description?: string
  required?: boolean
  min?: number
  max?: number
  options?: string[]
}

export interface FieldGroup {
  name: string
  label: string
  fields: FieldSchema[]
}

export interface ConfigSchema {
  globals: { groups: FieldGroup[] }
  phases: Record<string, { groups: FieldGroup[] }>
}

export interface CompatResult {
  compatible: boolean
  details: string
}

export interface ServeInfo {
  port: number
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const resp = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!resp.ok) {
    const text = await resp.text()
    throw new Error(`${resp.status} ${resp.statusText}: ${text}`)
  }
  if (resp.status === 204) return undefined as unknown as T
  return resp.json() as Promise<T>
}

export const api = {
  presets: {
    list: () => request<string[]>('/api/presets'),
    describe: (name: string) => request<DescribeOutput>(`/api/presets/${name}/describe`),
    checkCompat: (name: string) =>
      request<CompatResult>(`/api/presets/${name}/check-compat`, { method: 'POST' }),
  },

  projects: {
    list: () => request<Project[]>('/api/projects'),
    create: (body: { name: string; preset_name: string; config: Record<string, unknown> }) =>
      request<Project>('/api/projects', { method: 'POST', body: JSON.stringify(body) }),
    get: (id: string) => request<Project>(`/api/projects/${id}`),
    delete: (id: string) => request<void>(`/api/projects/${id}`, { method: 'DELETE' }),
  },

  runs: {
    list: (projectId: string) => request<Run[]>(`/api/projects/${projectId}/runs`),
    start: (projectId: string) =>
      request<Run>(`/api/projects/${projectId}/runs`, { method: 'POST' }),
    get: (projectId: string, runId: string) =>
      request<Run>(`/api/projects/${projectId}/runs/${runId}`),
    stop: (projectId: string, runId: string) =>
      request<void>(`/api/projects/${projectId}/runs/${runId}/stop`, { method: 'POST' }),
    metrics: (projectId: string, runId: string, params?: { phase?: string; name?: string; since_step?: number }) => {
      const qs = new URLSearchParams()
      if (params?.phase) qs.set('phase', params.phase)
      if (params?.name) qs.set('name', params.name)
      if (params?.since_step !== undefined) qs.set('since_step', String(params.since_step))
      return request<Metric[]>(`/api/projects/${projectId}/runs/${runId}/metrics?${qs}`)
    },
    logs: (projectId: string, runId: string, stderr = false, tail = 200) =>
      request<{ content: string }>(
        `/api/projects/${projectId}/runs/${runId}/logs?stderr=${stderr}&tail=${tail}`
      ),
    startServe: (projectId: string, runId: string) =>
      request<ServeInfo>(`/api/projects/${projectId}/runs/${runId}/serve`, { method: 'POST' }),
    stopServe: (projectId: string, runId: string) =>
      request<void>(`/api/projects/${projectId}/runs/${runId}/serve`, { method: 'DELETE' }),
  },
}
