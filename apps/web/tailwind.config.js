/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{ts,tsx}',
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui"],
      },
      colors: {
        bg: '#0b0f19',
        panel: '#0f1524',
        panelAlt: '#111827',
        text: '#E5E7EB',
        accent: '#7c3aed',
        accent2: '#06b6d4'
      },
      boxShadow: {
        glow: '0 0 0 1px rgba(124,58,237,0.2), 0 0 30px rgba(124,58,237,0.15)'
      }
    },
  },
  plugins: [],
}
