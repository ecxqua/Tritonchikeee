import type { Config } from "tailwindcss";

export default {
  content: [
    "./src/renderer/**/*.{ts,tsx,js,jsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
} satisfies Config;