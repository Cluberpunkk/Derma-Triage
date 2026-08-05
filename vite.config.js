import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    // forward /api calls to the backend proxy so the API key stays server-side
    proxy: { "/api": "http://localhost:8787" },
  },
});
