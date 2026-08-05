import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "url";
import { dirname } from "path";

// Separate Vite app for the labelling console. `root` is pinned to this folder
// so it builds console/web/index.html (NOT the public app's root index.html),
// and outputs to console/web/dist which the console backend serves. In dev it
// proxies the console API to the console backend on :8788 (NOT public :8787).
const root = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  root,
  plugins: [react()],
  build: { outDir: "dist", emptyOutDir: true },
  server: {
    port: 5174,
    proxy: { "/console-api": "http://localhost:8788" },
  },
});
