import { useState, useEffect } from 'react'

export default function TemplateManager() {
  const [templates, setTemplates] = useState<any[]>([])
  const [selectedHistory, setSelectedHistory] = useState<{ slug: string, history: any[] } | null>(null);
  const [loadingHistory, setLoadingHistory] = useState(false);

  useEffect(() => {
    fetchTemplates();
  }, [])

  const fetchTemplates = () => {
    fetch('/v1/templates')
      .then(res => res.json())
      .then(data => setTemplates(data.templates || []))
      .catch(console.error);
  };

  const showHistory = (slug: string) => {
    setLoadingHistory(true);
    fetch(`/v1/templates/${slug}/config-history`, {
      headers: { 'Authorization': 'Bearer ' + (localStorage.getItem('admin_token') || 'change-me-in-production') }
    })
      .then(res => res.json())
      .then(data => {
        setSelectedHistory({ slug, history: data.history || [] });
        setLoadingHistory(false);
      })
      .catch(err => {
        console.error(err);
        setLoadingHistory(false);
      });
  };

  const handleRollback = (slug: string, historyId: number) => {
    if (!window.confirm(`Rollback template ${slug} to version #${historyId}?`)) return;
    
    fetch(`/v1/templates/${slug}/rollback/${historyId}`, {
      method: 'POST',
      headers: { 'Authorization': 'Bearer ' + (localStorage.getItem('admin_token') || 'change-me-in-production') }
    })
      .then(res => res.json())
      .then(data => {
        alert(data.message);
        setSelectedHistory(null);
        fetchTemplates();
      })
      .catch(console.error);
  };

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
               <button className="btn btn-outline" style={{ flex: 1, padding: '0.5rem' }} onClick={() => showHistory(t.id)}>History</button>
            </div>
          </div>
        ))}

        {selectedHistory && (
          <div className="modal-overlay" style={{
            position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
            background: 'rgba(0,0,0,0.8)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000
          }}>
            <div className="glass-panel" style={{ width: '500px', maxHeight: '80vh', overflowY: 'auto', padding: '1.5rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1rem' }}>
                <h3 style={{ margin: 0 }}>Config History: {selectedHistory.slug}</h3>
                <button className="btn-mini" onClick={() => setSelectedHistory(null)}>✕</button>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                {loadingHistory ? (
                  <p>Loading history...</p>
                ) : (
                  <>
                    {selectedHistory.history.map((h: any) => (
                      <div key={h.id} style={{ padding: '10px', border: '1px solid #333', borderRadius: '6px', background: 'rgba(255,255,255,0.03)' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '4px' }}>
                          <span style={{ color: '#60a5fa' }}>#{h.id}</span>
                          <span style={{ color: '#666' }}>{new Date(h.changed_at).toLocaleString()}</span>
                        </div>
                        <div style={{ fontSize: '0.9rem', marginBottom: '8px' }}>{h.change_note || 'No note'}</div>
                        <button className="btn btn-mini" style={{ width: '100%' }} onClick={() => handleRollback(selectedHistory.slug, h.id)}>Rollback to this</button>
                      </div>
                    ))}
                    {selectedHistory.history.length === 0 && <p>No history found.</p>}
                  </>
                )}
              </div>
            </div>
          </div>
        )}
        {templates.length === 0 && (
          <div className="glass-panel" style={{ padding: '2rem', textAlign: 'center', gridColumn: '1 / -1' }}>
            <p style={{ color: 'var(--text-secondary)' }}>Chưa có template nào được active.</p>
          </div>
        )}
      </div>
    </div>
  )
}
