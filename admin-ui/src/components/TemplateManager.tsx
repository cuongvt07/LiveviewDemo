import { useState, useEffect } from 'react'

export default function TemplateManager() {
  const [templates, setTemplates] = useState<any[]>([])
  
  // Reload only works if API is configured. For the MVP UI we'll just show info.
  useEffect(() => {
    fetch('/v1/templates')
      .then(res => res.json())
      .then(data => setTemplates(data.templates || []))
      .catch(console.error)
  }, [])

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
        <div>
          <h2 style={{ margin: 0, fontSize: '1.5rem' }}>Active Templates</h2>
          <p style={{ margin: '0.5rem 0 0', color: 'var(--text-secondary)' }}>Quản lý và cấu hình các template 3D (Cốc, Áo, v.v)</p>
        </div>
        <button className="btn btn-primary">+ Add Template</button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '1.5rem' }}>
        {templates.map(t => (
          <div key={t.id} className="glass-panel" style={{ padding: '1rem' }}>
            <div style={{ 
              width: '100%', 
              aspectRatio: '1', 
              background: 'rgba(0,0,0,0.5)', 
              borderRadius: '8px', 
              marginBottom: '1rem',
              backgroundImage: `url(${t.preview_url})`,
              backgroundSize: 'cover',
              backgroundPosition: 'center'
            }}></div>
            <h3 style={{ margin: '0 0 0.5rem', fontSize: '1.125rem' }}>{t.name}</h3>
            <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginBottom: '1rem' }}>
              Slug: {t.id} <br />
              Size: {t.output_size[0]}x{t.output_size[1]}
            </div>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
               <button className="btn btn-outline" style={{ flex: 1, padding: '0.5rem' }}>Config</button>
               <button className="btn btn-outline" style={{ flex: 1, padding: '0.5rem' }}>Archive</button>
            </div>
          </div>
        ))}
        {templates.length === 0 && (
          <div className="glass-panel" style={{ padding: '2rem', textAlign: 'center', gridColumn: '1 / -1' }}>
            <p style={{ color: 'var(--text-secondary)' }}>Chưa có template nào được active.</p>
          </div>
        )}
      </div>
    </div>
  )
}
