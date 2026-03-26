import { useState, useEffect, useMemo } from 'react'
import PrintAreaEditor from './PrintAreaEditor'

export default function LivePreview() {
  const [mode, setMode] = useState<'template' | 'adhoc'>('template')
  const [templates, setTemplates] = useState<any[]>([])
  const [selectedTemplate, setSelectedTemplate] = useState('')
  const [designFile, setDesignFile] = useState<File | null>(null)
  
  // Custom mockups
  const [mockupFile, setMockupFile] = useState<File | null>(null)
  const [printArea, setPrintArea] = useState<any>(null)
  const [warpConfigObj, setWarpConfigObj] = useState<any>({ warp_type: 'cylinder', theta_max_deg: 52, curve: 0.15 })
  
  // Results
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [renderTime, setRenderTime] = useState<number | null>(null)

  useEffect(() => {
    fetch('/v1/templates')
      .then(res => res.json())
      .then(data => {
        setTemplates(data.templates || [])
        if (data.templates?.length > 0) setSelectedTemplate(data.templates[0].id)
      })
      .catch(console.error)
  }, [])

  const mockupPreviewUrl = useMemo(() => {
    return mockupFile ? URL.createObjectURL(mockupFile) : null
  }, [mockupFile])

  const handleRender = async () => {
    if (!designFile) return
    if (mode === 'template' && !selectedTemplate) return
    if (mode === 'adhoc' && !mockupFile) return

    setLoading(true)
    const formData = new FormData()
    formData.append('design_image', designFile)

    try {
      let url = ''
      if (mode === 'template') {
        formData.append('template_id', selectedTemplate)
        url = '/v1/mockup/render'
      } else {
        formData.append('mockup_image', mockupFile!)
        if (printArea) {
          const configJson = {
            print_area: printArea,
            warp: warpConfigObj
          }
          formData.append('config_json', JSON.stringify(configJson))
        }
        url = '/v1/mockup/render-adhoc'
      }

      const res = await fetch(url, {
        method: 'POST',
        body: formData,
      })
      
      if (!res.ok) {
        let errMsg = 'Unknown error';
        try {
          const err = await res.json();
          errMsg = err.message || err.detail?.message || err.detail || JSON.stringify(err);
        } catch(e) {
          errMsg = `HTTP ${res.status}: ${res.statusText}`;
        }
        alert(`Lỗi Render: ${errMsg}`);
        setLoading(false);
        return
      }

      const timeStr = res.headers.get('X-Processing-Time-Ms')
      if (timeStr) setRenderTime(parseInt(timeStr, 10))

      const blob = await res.blob()
      setPreviewUrl(URL.createObjectURL(blob))
    } catch (e: any) {
      alert(`Lỗi kết nối: ${e.message}`)
    }
    setLoading(false)
  }

  const isValid = designFile && (mode === 'template' ? selectedTemplate !== '' : mockupFile !== null)

  return (
    <div style={{ display: 'flex', gap: '2rem', height: '100%' }}>
      {/* Settings Form */}
      <div className="glass-panel" style={{ width: '400px', padding: '1.5rem', display: 'flex', flexDirection: 'column', overflowY: 'auto' }}>
        <h2 style={{ marginTop: 0, marginBottom: '1.5rem', fontSize: '1.25rem' }}>Render Settings</h2>
        
        {/* Mode Switcher */}
        <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1.5rem', background: 'rgba(0,0,0,0.2)', padding: '0.25rem', borderRadius: '8px' }}>
          <button 
            style={{ flex: 1, padding: '0.5rem', borderRadius: '6px', border: 'none', background: mode === 'template' ? 'var(--surface-color)' : 'transparent', color: mode === 'template' ? 'white' : 'var(--text-secondary)', cursor: 'pointer', transition: 'all 0.2s' }}
            onClick={() => setMode('template')}
          >
            Từ Template
          </button>
          <button 
            style={{ flex: 1, padding: '0.5rem', borderRadius: '6px', border: 'none', background: mode === 'adhoc' ? 'var(--surface-color)' : 'transparent', color: mode === 'adhoc' ? 'white' : 'var(--text-secondary)', cursor: 'pointer', transition: 'all 0.2s' }}
            onClick={() => setMode('adhoc')}
          >
            Tự Upload 2 Ảnh
          </button>
        </div>

        {mode === 'template' ? (
          <div className="input-group">
            <label className="input-label">Select Template</label>
            <select 
              className="input-field" 
              value={selectedTemplate} 
              onChange={e => setSelectedTemplate(e.target.value)}
            >
              {templates.length === 0 && <option value="">Đang tải...</option>}
              {templates.map(t => (
                <option key={t.id} value={t.id}>{t.name}</option>
              ))}
            </select>
          </div>
        ) : (
          <div className="input-group">
            <label className="input-label">Upload Base Mockup (Phôi)</label>
            <input 
              type="file" 
              accept="image/png, image/jpeg" 
              className="input-field"
              onChange={e => setMockupFile(e.target.files?.[0] || null)}
            />
            
            {mockupPreviewUrl && (
              <div style={{ marginTop: '1rem' }}>
                <PrintAreaEditor 
                  imageUrl={mockupPreviewUrl} 
                  warpConfig={{
                    ...warpConfigObj,
                    tilt_deg: warpConfigObj.tilt_deg || 0,
                    rotate_deg: warpConfigObj.rotate_deg || 0,
                    persp_strength: warpConfigObj.persp_strength || 0
                  }}
                  onCoordinatesChange={setPrintArea}
                  onConfigChange={(newConfig) => {
                    setWarpConfigObj((prev: any) => ({ ...prev, ...newConfig }));
                  }}
                />
              </div>
            )}
          </div>
        )}

        <div className="input-group" style={{ marginTop: '1rem' }}>
          <label className="input-label">Upload Design (Artwork)</label>
          <input 
            type="file" 
            accept="image/png, image/jpeg" 
            className="input-field"
            onChange={e => setDesignFile(e.target.files?.[0] || null)}
          />
        </div>

        <div style={{ marginTop: 'auto', paddingTop: '1.5rem' }}>
          <button 
            className="btn btn-primary" 
            style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '0.5rem', padding: '1rem' }}
            onClick={handleRender}
            disabled={loading || !isValid}
          >
            {loading ? 'Đang xử lý...' : 'Gen Mockup'}
          </button>
        </div>
      </div>

      {/* Preview Area */}
      <div className="glass-panel" style={{ flex: 1, padding: '1.5rem', display: 'flex', flexDirection: 'column' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
          <h2 style={{ margin: 0, fontSize: '1.25rem' }}>Result</h2>
          {renderTime && (
            <div style={{ background: 'rgba(59, 130, 246, 0.2)', color: 'var(--accent-color)', padding: '0.25rem 0.75rem', borderRadius: '100px', fontSize: '0.875rem', fontWeight: 600 }}>
              ⚡ {renderTime} ms
            </div>
          )}
        </div>
        
        <div className="preview-container" style={{ flex: 1, maxWidth: 'none', background: 'rgba(0,0,0,0.1)' }}>
          {previewUrl ? (
            <img src={previewUrl} alt="Mockup Result" className="preview-image" />
          ) : (
            <div className="empty-state">
              <div style={{ fontSize: '3rem', marginBottom: '1rem', opacity: 0.5 }}>🎨</div>
              <div>Bấm Gen Mockup để xem phần thiết kế uốn cong theo cốc</div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
