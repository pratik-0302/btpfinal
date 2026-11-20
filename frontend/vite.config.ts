import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Development only: the dev server listens on 127.0.0.1 and forwards /api to the local FastAPI server.
// For normal use, `npm run build` and open the FastAPI server (it serves frontend/dist).
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: false },
    },
  },
  // one ~660 kB bundle (React + Recharts) is fine for a page served from this computer
  build: { outDir: "dist", sourcemap: false, chunkSizeWarningLimit: 1000 },
});
