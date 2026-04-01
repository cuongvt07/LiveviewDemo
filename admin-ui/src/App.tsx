import { useState } from 'react'
import { Monitor, FileImage, LayoutTemplate, Zap, Menu, ChevronLeft, ChevronRight } from 'lucide-react'
import t from './i18n/translate'
import LivePreview from './components/LivePreview'
import TemplateManager from './components/TemplateManager'

function App() {
  const [activeTab, setActiveTab] = useState('preview')
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false)

  return (
    <div className={`app-root ${isSidebarCollapsed ? 'sidebar-collapsed' : ''}`}>
      <header className="app-header glass-panel" style={{ borderRadius: 0, borderTop: 0, borderLeft: 0, borderRight: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <button 
            className="btn-icon sidebar-toggle" 
            onClick={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
            style={{ background: 'transparent', border: 'none', color: 'var(--text-primary)', cursor: 'pointer' }}
          >
            <Menu size={20} />
          </button>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div style={{ background: 'var(--accent-color)', padding: '0.5rem', borderRadius: '8px' }}>
              <Zap size={20} color="white" />
            </div>
            <h1 style={{ margin: 0, fontSize: '1.25rem', fontWeight: 600 }}>{t('app.title')}</h1>
          </div>
        </div>
        <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
          {t('app.version', { version: '1.2' })}
        </div>
      </header>

      <div className="main-layout">
        <aside className="sidebar glass-panel">
          <nav>
            <div 
              className={`nav-link ${activeTab === 'preview' ? 'active' : ''}`}
              onClick={() => setActiveTab('preview')}
              title={t('app.nav.preview')}
            >
              <Monitor size={18} /> <span>{t('app.nav.preview')}</span>
            </div>
            <div 
              className={`nav-link ${activeTab === 'templates' ? 'active' : ''}`}
              onClick={() => setActiveTab('templates')}
              title={t('app.nav.templates')}
            >
              <LayoutTemplate size={18} /> <span>{t('app.nav.templates')}</span>
            </div>
            <div 
              className={`nav-link ${activeTab === 'assets' ? 'active' : ''}`}
              onClick={() => setActiveTab('assets')}
              title={t('app.nav.maps_editor')}
              style={{ opacity: 0.5, pointerEvents: 'none' }}
            >
              <FileImage size={18} /> <span>{t('app.nav.maps_editor')}</span>
            </div>
          </nav>

          <button 
            className="sidebar-collapse-btn" 
            onClick={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
          >
            {isSidebarCollapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
          </button>
        </aside>

        <main className="content-area">
          {activeTab === 'preview' && <LivePreview />}
          {activeTab === 'templates' && <TemplateManager />}
        </main>
      </div>
    </div>
  )
}

export default App
