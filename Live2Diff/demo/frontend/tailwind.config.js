/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./src/**/*.{html,js,svelte,ts}', '../**/*.py'],
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#eef8ff',
          100: '#d8eeff',
          200: '#b2ddff',
          300: '#7ec4ff',
          400: '#45a5ff',
          500: '#1383ff',
          600: '#005fe0',
          700: '#004bb4',
          800: '#093f8f',
          900: '#103774'
        },
        neon: {
          cyan: '#5eead4',
          violet: '#8b5cf6',
          pink: '#ec4899'
        }
      },
      boxShadow: {
        soft: '0 20px 60px rgba(15, 23, 42, 0.12)',
        glass: '0 24px 80px rgba(15, 23, 42, 0.16)',
        neon: '0 0 0 1px rgba(94, 234, 212, 0.18), 0 16px 48px rgba(34, 211, 238, 0.18)',
        hero: '0 24px 120px rgba(59, 130, 246, 0.22)'
      },
      backgroundImage: {
        'grid-fade':
          'linear-gradient(to right, rgba(148, 163, 184, 0.14) 1px, transparent 1px), linear-gradient(to bottom, rgba(148, 163, 184, 0.14) 1px, transparent 1px)'
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translate3d(0, 0, 0)' },
          '50%': { transform: 'translate3d(0, -10px, 0)' }
        },
        'pulse-glow': {
          '0%, 100%': { opacity: '0.45', transform: 'scale(1)' },
          '50%': { opacity: '0.85', transform: 'scale(1.03)' }
        },
        'fade-up': {
          '0%': { opacity: '0', transform: 'translate3d(0, 24px, 0)' },
          '100%': { opacity: '1', transform: 'translate3d(0, 0, 0)' }
        },
        shimmer: {
          '0%': { backgroundPosition: '200% 0' },
          '100%': { backgroundPosition: '-200% 0' }
        },
        'border-beam': {
          '0%': { backgroundPosition: '0% 50%' },
          '100%': { backgroundPosition: '200% 50%' }
        },
        'aurora-shift': {
          '0%': { transform: 'translate3d(-8%, -4%, 0) scale(1)' },
          '50%': { transform: 'translate3d(6%, 3%, 0) scale(1.08)' },
          '100%': { transform: 'translate3d(-8%, -4%, 0) scale(1)' }
        }
      },
      animation: {
        float: 'float 6s ease-in-out infinite',
        'pulse-glow': 'pulse-glow 4.2s ease-in-out infinite',
        'fade-up': 'fade-up 0.7s cubic-bezier(0.22, 1, 0.36, 1) both',
        shimmer: 'shimmer 2.8s linear infinite',
        'border-beam': 'border-beam 8s linear infinite',
        aurora: 'aurora-shift 18s ease-in-out infinite'
      }
    }
  },
  plugins: []
};
