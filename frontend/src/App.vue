<template>
  <nav>
    <RouterLink to="/">utrain</RouterLink>
    <RouterLink to="/new">+ New Project</RouterLink>
  </nav>
  <main>
    <RouterView />
  </main>
</template>

<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, sans-serif; background: #0f0f0f; color: #e0e0e0; }
nav { display: flex; gap: 1rem; padding: 0.75rem 1.5rem; background: #1a1a1a; border-bottom: 1px solid #333; }
nav a { color: #e0e0e0; text-decoration: none; font-weight: 600; }
nav a:hover { color: #fff; }
main { padding: 1.5rem; max-width: 1100px; margin: 0 auto; }
</style>
