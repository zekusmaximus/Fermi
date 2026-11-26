import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// Dev server proxies /api to the local FastAPI desk (`uv run fermi-desk serve`).
// The production bundle is served by FastAPI itself from web/dist, same origin.
export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
  build: { outDir: "dist", emptyOutDir: true, sourcemap: false },
  test: { environment: "node", include: ["src/**/*.test.ts"] },
});
