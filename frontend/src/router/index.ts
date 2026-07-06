import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'
import NewProjectView from '../views/NewProjectView.vue'
import ProjectView from '../views/ProjectView.vue'
import ChatView from '../views/ChatView.vue'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', component: HomeView },
    { path: '/new', component: NewProjectView },
    { path: '/projects/:projectId', component: ProjectView },
    { path: '/projects/:projectId/runs/:runId/chat', component: ChatView },
  ],
})

export default router
