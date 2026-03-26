import { useState } from 'react'
import { Monitor, FileImage, LayoutTemplate, Zap } from 'lucide-react'
import LivePreview from './components/LivePreview'
import TemplateManager from './components/TemplateManager'

function App() {
  const [activeTab, setActiveTab] = useState('preview')

  return (
    <>
      <header className="app-header glass-panel" style={{ borderRadius: 0, borderTop: 0, borderLeft: 0, borderRight: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div style={{ background: 'var(--accent-color)', padding: '0.5rem', borderRadius: '8px' }}>
            <Zap size={20} color="white" />
          </div>
          <h1 style={{ margin: 0, fontSize: '1.25rem', fontWeight: 600 }}>POD Liveview Admin</h1>
        </div>
        <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
          Mug Mockup Service v1.2
        </div>
      </header>

      <div className="main-layout">
        <aside className="sidebar glass-panel" style={{ borderRadius: 0, borderTop: 0, borderBottom: 0, borderLeft: 0 }}>
          <nav>
            <div 
              className={`nav-link ${activeTab === 'preview' ? 'active' : ''}`}
              onClick={() => setActiveTab('preview')}
              style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}
            >
              <Monitor size={18} /> Live Preview
            </div>
            <div 
              className={`nav-link ${activeTab === 'templates' ? 'active' : ''}`}
              onClick={() => setActiveTab('templates')}
              style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}
            >
              <LayoutTemplate size={18} /> Templates
            </div>
            <div 
              className={`nav-link ${activeTab === 'assets' ? 'active' : ''}`}
              onClick={() => setActiveTab('assets')}
              style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', opacity: 0.5, pointerEvents: 'none' }}
            >
              <FileImage size={18} /> Maps Editor (Dev)
            </div>
          </nav>
        </aside>

        <main className="content-area">
          {activeTab === 'preview' && <LivePreview />}
          {activeTab === 'templates' && <TemplateManager />}
        </main>
      </div>
    </>
  )
}

export default App
