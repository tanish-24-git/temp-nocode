# LLM Platform Frontend

Modern React frontend for the LLM Fine-Tuning Platform with dark/light mode support.

## Features

✨ **Modern UI** - Built with React, Vite, and TailwindCSS
🌓 **Dark/Light Mode** - Automatic theme switching with localStorage persistence
🎨 **Beautiful Design** - Gradient backgrounds, smooth animations, glassmorphism
📱 **Responsive** - Works on desktop, tablet, and mobile
🚀 **Fast** - Vite for instant HMR and optimized builds

## Pages

- **Home** - Hero section, features, workflow explanation, and CTA
- **Playground** - Interactive pipeline builder with drag-and-drop agents
- **Jobs** - Monitor running and completed training jobs
- **Models** - Browse and download trained models

## Getting Started

### Install Dependencies

```bash
cd llm-platform-ui
npm install
```

### Run Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Build for Production

```bash
npm run build
```

## Project Structure

```
llm-platform-ui/
├── src/
│   ├── components/
│   │   └── Navbar.jsx          # Navigation bar with theme toggle
│   ├── context/
│   │   └── ThemeContext.jsx    # Dark/light mode context
│   ├── pages/
│   │   ├── HomePage.jsx        # Landing page
│   │   ├── PlaygroundPage.jsx  # Pipeline builder
│   │   ├── JobsPage.jsx        # Jobs monitoring
│   │   └── ModelsPage.jsx      # Models gallery
│   ├── App.jsx                 # Main app with routing
│   ├── main.jsx                # Entry point
│   └── index.css               # Global styles
├── index.html
├── vite.config.js
├── tailwind.config.js
├── postcss.config.js
└── package.json
```

## Tech Stack

- **React 18** - UI library
- **Vite** - Build tool
- **React Router** - Client-side routing
- **TailwindCSS** - Utility-first CSS
- **Framer Motion** - Animations
- **Lucide React** - Icons
- **TanStack Query** - Data fetching
- **React Hot Toast** - Notifications

## API Integration

The frontend proxies API requests to the backend:

```javascript
// vite.config.js
proxy: {
  '/api': {
    target: 'http://localhost:8000',
    changeOrigin: true,
  }
}
```

## Theme System

The app supports dark and light modes with automatic detection:

```jsx
import { useTheme } from './context/ThemeContext'

function MyComponent() {
  const { theme, toggleTheme } = useTheme()
  
  return (
    <button onClick={toggleTheme}>
      Toggle to {theme === 'dark' ? 'light' : 'dark'} mode
    </button>
  )
}
```

## License

MIT
