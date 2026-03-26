import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/v1': 'http://localhost:8005',
      '/admin': 'http://localhost:8005',
      '/static': 'http://localhost:8005',
    }
  }
})
