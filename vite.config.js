import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    // forward /api calls to the backend proxy so the API key stays server-side
    proxy: { "/api": "http://localhost:8787" },

    // DEV ONLY — phone testing over an ngrok tunnel.
    // Vite rejects requests whose Host header it does not recognise (a DNS-rebinding
    // protection), which is why the phone saw "Blocked request. This host is not
    // allowed." Wildcards cover the rotating subdomain ngrok issues on each restart.
    // This has no effect on `vite build`; the production bundle is served by whatever
    // host you deploy behind, so nothing here loosens anything in a real deployment.
    allowedHosts: [".ngrok-free.dev", ".ngrok-free.app"],

    // Fail loudly rather than drifting to 5174 when 5173 is taken. A silent port
    // change desyncs the dev server from the tunnel, which points at a fixed port —
    // the symptom is a stale or dead page on the phone with no error on either side.
    strictPort: true,
  },
});
