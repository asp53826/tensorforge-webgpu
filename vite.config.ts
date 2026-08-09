import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "/tensorforge-webgpu/",
  plugins: [react()],
  build: {
    target: "es2022",
    sourcemap: true
  },
  test: {
    environment: "node",
    include: ["src/test/**/*.test.ts"]
  }
});
