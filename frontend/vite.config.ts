import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// In development, requests to /api are forwarded to the FastAPI backend.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": "http://localhost:8000" } },
  test: { environment: "jsdom", setupFiles: ["./src/test/setup.ts"], globals: true },
});