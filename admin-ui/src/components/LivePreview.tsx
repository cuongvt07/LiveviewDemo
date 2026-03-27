import { useState, useEffect, useMemo, useRef } from 'react'
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
  const [lockedSnapshot, setLockedSnapshot] = useState<{ printArea: any; warpConfig: any } | null>(null)
  const liveEditorSnapshotRef = useRef<{ printArea: any; warpConfig: any } | null>(null)
  const [showEditor, setShowEditor] = useState(false)
  const [productType, setProductType] = useState('') // Initial empty state to enforce selection
  
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
        formData.append('output_format', 'png')
        const liveSnapshot = liveEditorSnapshotRef.current
        const effectivePrintArea = liveSnapshot?.printArea || lockedSnapshot?.printArea || printArea
        const effectiveWarp = liveSnapshot?.warpConfig || lockedSnapshot?.warpConfig || warpConfigObj
        if (effectivePrintArea) {
          const isCylinderProduct = String(effectiveWarp?.product_type || productType).startsWith('cylinder')
          console.info('[Gen Mockup] effective adhoc payload', {
            print_area: effectivePrintArea,
            warp: effectiveWarp,
          })
          const configJson = {
            print_area: effectivePrintArea,
            warp: effectiveWarp,
            ...(isCylinderProduct ? {
              color: {
                enable_color_match: false,
                match_strength: 0,
              },
              lighting: {
                shadow_strength: 0,
                displacement_strength: 0,
                specular_strength: 0,
                specular_threshold: 245,
              },
              edge: {
                feather_px: 0,
              },
              render: {
                preserve_original_color: true,
              },
            } : {}),
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
  const hasEffectivePrintArea = Boolean(liveEditorSnapshotRef.current?.printArea || lockedSnapshot?.printArea || printArea)

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
          <div className="adhoc-workflow">
            {/* STEP 1: MODULE SELECTION */}
            <div className="input-group" style={{ border: '1px solid var(--accent-color)', padding: '1rem', borderRadius: '8px', marginBottom: '1.5rem' }}>
              <label className="input-label" style={{ color: 'var(--accent-color)', fontWeight: 'bold' }}>Step 1: Chọn loại sản phẩm (Module)</label>
              <select 
                className="input-field" 
                value={productType} 
                onChange={e => {
                  setProductType(e.target.value);
                  // Reset if module changes
                  setPrintArea(null);
                  setLockedSnapshot(null);
                  liveEditorSnapshotRef.current = null;
                  setMockupFile(null);
                }}
              >
                <option value="">-- Chọn Module --</option>
                <optgroup label="🍵 MUGS MODULE (Cylindrical Warp)">
                  <option value="cylinder_ceramic">Ceramic Mug</option>
                  <option value="cylinder_glass">Glass Mug</option>
                  <option value="cylinder_travel">Travel Mug</option>
                </optgroup>
                <optgroup label="👕 CLOTHES MODULE (TPS Warp)">
                  <option value="apparel_cotton">T-Shirt</option>
                  <option value="apparel_hoodie">Hoodie</option>
                  <option value="apparel_totebag">Tote Bag</option>
                </optgroup>
                <optgroup label="🖼 OTHER">
                  <option value="flat_print">Flat Print</option>
                  <option value="plastic_case">Phone Case</option>
                </optgroup>
              </select>
            </div>

            {/* STEP 2: UPLOAD (Only if Module selected) */}
            {productType && (
              <div className="input-group animate-in" style={{ padding: '1rem', background: 'rgba(255,255,255,0.03)', borderRadius: '8px' }}>
                <label className="input-label" style={{ fontWeight: 'bold' }}>Step 2: Upload Base Mockup (Phôi)</label>
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                  <input 
                    type="file" 
                    accept="image/png, image/jpeg" 
                    className="input-field"
                    onChange={e => {
                      setMockupFile(e.target.files?.[0] || null);
                      setLockedSnapshot(null);
                      liveEditorSnapshotRef.current = null;
                      if (e.target.files?.[0]) setShowEditor(true);
                    }}
                  />
                  {mockupFile && (
                    <button className="btn btn-outline" onClick={() => setShowEditor(true)} style={{ padding: '0.75rem' }}>
                      Edit Area
                    </button>
                  )}
                </div>
                
                {mockupFile && printArea ? (
                  <div style={{ marginTop: '0.5rem', fontSize: '0.8rem', color: '#10b981' }}>
                    ✓ Print Area & {productType.startsWith('cylinder') ? 'Curve' : 'Mesh'} Configured
                  </div>
                ) : mockupFile && (
                  <div style={{ marginTop: '0.5rem', fontSize: '0.8rem', color: 'var(--accent-color)' }}>
                    ⚠ Cần Calibrate Print Area
                  </div>
                )}
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

      {/* Fullscreen Editor Modal */}
      {showEditor && mockupPreviewUrl && (
        <div className="modal-overlay" style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.8)', zIndex: 1000, display: 'flex', alignItems: 'center', justifyItems: 'center', padding: '2rem' }}>
          <div className="glass-panel" style={{ width: '100%', height: '100%', background: 'var(--bg-color)', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            <div style={{ padding: '1rem', borderBottom: '1px solid var(--border-color)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '1rem' }}>
              <h3 style={{ margin: 0, whiteSpace: 'nowrap' }}>Calibrate Print Area & Mesh</h3>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap', justifyContent: 'flex-end' }}>
                <label className="input-label" style={{ marginBottom: 0 }}>Upload Design (Artwork)</label>
                <input
                  type="file"
                  accept="image/png, image/jpeg"
                  className="input-field"
                  style={{ width: '260px', maxWidth: '45vw' }}
                  onChange={e => setDesignFile(e.target.files?.[0] || null)}
                />
                <button
                  className="btn btn-primary"
                  onClick={handleRender}
                  disabled={loading || !designFile || !mockupFile || !hasEffectivePrintArea}
                  title={!designFile ? 'Hãy upload artwork trước khi gen' : (!hasEffectivePrintArea ? 'Hãy chỉnh vùng in trước khi gen' : 'Gen ngay với lưới hiện tại')}
                >
                  {loading ? 'Đang Gen...' : 'Gen Mockup'}
                </button>
                <button className="btn btn-outline" onClick={() => setShowEditor(false)}>Done & Close</button>
              </div>
            </div>
            <div style={{ flex: 1, overflow: 'hidden', display: 'flex', minHeight: 0 }}>
              <div style={{ flex: 1, minWidth: 0, overflow: 'hidden' }}>
                <PrintAreaEditor 
                  imageUrl={mockupPreviewUrl} 
                  designFile={designFile}
                  productTypeProp={productType}
                  onProductTypeChange={setProductType}
                  initialPrintArea={printArea}
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
                  onLockedSnapshotChange={setLockedSnapshot}
                  onLiveSnapshotChange={(snapshot) => {
                    liveEditorSnapshotRef.current = snapshot;
                  }}
                />
              </div>
              <aside
                className="glass-panel"
                style={{
                  width: '550px',
                  minWidth: '320px',
                  borderLeft: '1px solid var(--border-color)',
                  background: 'rgba(10, 16, 28, 0.85)',
                  padding: '0.9rem',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.6rem',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <h4 style={{ margin: 0, fontSize: '0.95rem' }}>Gen Result (Popup)</h4>
                  {renderTime && (
                    <span style={{ fontSize: '0.78rem', color: 'var(--accent-color)' }}>⚡ {renderTime} ms</span>
                  )}
                </div>
                <div
                  style={{
                    flex: 1,
                    minHeight: 0,
                    borderRadius: '8px',
                    border: '1px solid var(--border-color)',
                    background: 'rgba(0,0,0,0.25)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    overflow: 'hidden',
                  }}
                >
                  {previewUrl ? (
                    <img
                      src={previewUrl}
                      alt="Popup Render Result"
                      style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                    />
                  ) : (
                    <div style={{ color: 'var(--text-secondary)', fontSize: '0.82rem', textAlign: 'center', padding: '1rem' }}>
                      Bấm Gen Mockup để xem kết quả ngay trong popup.
                    </div>
                  )}
                </div>
              </aside>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
