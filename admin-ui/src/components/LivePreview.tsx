import { useState, useEffect, useMemo, useRef } from 'react'
import PrintAreaEditor from './PrintAreaEditor'
import LibraryPicker from './LibraryPicker'

function isObjectUrl(value: string | null): boolean {
  return Boolean(value && value.startsWith('blob:'))
}

function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onloadend = () => {
      if (typeof reader.result === 'string') resolve(reader.result)
      else reject(new Error('Không thể đọc ảnh preview'))
    }
    reader.onerror = () => reject(reader.error || new Error('Không thể đọc ảnh preview'))
    reader.readAsDataURL(blob)
  })
}

export default function LivePreview() {
  const [mode, setMode] = useState<'template' | 'adhoc'>('template')
  const [templates, setTemplates] = useState<any[]>([])
  const [selectedTemplate, setSelectedTemplate] = useState('')
  const [designFile, setDesignFile] = useState<File | null>(null)
  
  // Custom mockups
  const [mockupFile, setMockupFile] = useState<File | null>(null)
  const [printArea, setPrintArea] = useState<any>(null)
  const [warpConfigObj, setWarpConfigObj] = useState<any>({
    warp_type: 'cylinder',
    theta_max_deg: 52,
    curve: 0.15,
    edge_squeeze: 0,
    squeeze_power: 2,
    center_focus_width: 0,
    mesh_density_strength: 2,
    light_pos_x: 0.62,
    light_pos_y: 0.32,
    light_height: 55,
    light_contrast: 50,
    light_highlight: 60,
    light_softness: 55,
  })
  const [lockedSnapshot, setLockedSnapshot] = useState<{ printArea: any; warpConfig: any } | null>(null)
  const [liveEditorSnapshot, setLiveEditorSnapshot] = useState<{ printArea: any; warpConfig: any } | null>(null)
  const liveEditorSnapshotRef = useRef<{ printArea: any; warpConfig: any } | null>(null)
  const [showEditor, setShowEditor] = useState(false)
  const [productType, setProductType] = useState('') // Initial empty state to enforce selection

  // Library picker state
  const [pickerType, setPickerType] = useState<'bases' | 'artworks' | null>(null)
  const [libraryMockupUrl, setLibraryMockupUrl] = useState<string | null>(null)
  const [libraryDesignUrl, setLibraryDesignUrl] = useState<string | null>(null)
  const [sourceProductUrl, setSourceProductUrl] = useState('')
  const [urlImporting, setUrlImporting] = useState(false)
  const [urlImportSummary, setUrlImportSummary] = useState<any | null>(null)
  const [resolvedTemplatePreviewUrl, setResolvedTemplatePreviewUrl] = useState<string | null>(null)
  
  // Results
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [renderTime, setRenderTime] = useState<number | null>(null)

  useEffect(() => {
    fetch('/v1/templates')
      .then(res => res.json())
      .then(data => {
        setTemplates(data.templates || [])
        if (data.templates?.length > 0) {
          const first = data.templates[0].id
          setSelectedTemplate(first)
        }
      })
      .catch(console.error)
  }, [])

  const mockupPreviewUrl = useMemo(() => {
    if (libraryMockupUrl) return libraryMockupUrl
    return mockupFile ? URL.createObjectURL(mockupFile) : null
  }, [mockupFile, libraryMockupUrl])

  // designPreviewUrl removed (unused) to satisfy TypeScript

  useEffect(() => {
    return () => {
      // Internal blob URL cleanup only
      if (mockupFile) {
        // We don't have the previous URL easily here, but usually useMemo handles it if we return a cleanup
      }
    }
  }, [mockupFile])

  const effectiveSnapshot = liveEditorSnapshotRef.current || liveEditorSnapshot
  const effectivePrintArea = effectiveSnapshot?.printArea || lockedSnapshot?.printArea || printArea
  const effectiveWarp = effectiveSnapshot?.warpConfig || lockedSnapshot?.warpConfig || warpConfigObj
  const hasEffectivePrintArea = Boolean(effectivePrintArea)
  const isValid = Boolean(
    (designFile || libraryDesignUrl) && (
      mode === 'template'
        ? selectedTemplate !== ''
        : (mockupFile || libraryMockupUrl) && hasEffectivePrintArea
    )
  )

  useEffect(() => {
    setPreviewUrl(prev => {
      if (prev && isObjectUrl(prev)) URL.revokeObjectURL(prev)
      return null
    })
    setRenderTime(null)
  }, [designFile, libraryDesignUrl, mockupFile, libraryMockupUrl, mode, selectedTemplate])

  useEffect(() => {
    setResolvedTemplatePreviewUrl(null)
  }, [designFile, libraryDesignUrl, mockupFile, libraryMockupUrl, sourceProductUrl])

  useEffect(() => {
    if (!isValid) return

    const timer = setTimeout(() => {
      void handleRender()
    }, 250)

    return () => clearTimeout(timer)
  }, [
    isValid,
    mode,
    selectedTemplate,
    designFile,
    libraryDesignUrl,
    mockupFile,
    libraryMockupUrl,
    effectivePrintArea,
    effectiveWarp,
    productType,
  ])

  const handleRender = async () => {
    if (!designFile && !libraryDesignUrl) return
    if (mode === 'template' && !selectedTemplate) return
    if (mode === 'adhoc' && !mockupFile && !libraryMockupUrl) return
    if (mode === 'adhoc' && !effectivePrintArea) return

    setLoading(true)
    const formData = new FormData()
    
    if (designFile) {
      formData.append('design_image', designFile)
    } else if (libraryDesignUrl) {
      formData.append('design_url', libraryDesignUrl)
    }

    try {
      let url = ''
      if (mode === 'template') {
        formData.append('template_id', selectedTemplate)
        url = '/v1/mockup/render'
      } else {
        if (mockupFile) {
          formData.append('mockup_image', mockupFile)
        } else if (libraryMockupUrl) {
          formData.append('mockup_url', libraryMockupUrl)
        }
        formData.append('output_format', 'png')
        if (effectivePrintArea) {
          const isCylinderProduct = String(effectiveWarp?.product_type || productType).startsWith('cylinder')
          const configJson = {
            print_area: effectivePrintArea,
            warp: effectiveWarp,
            ...(isCylinderProduct ? {
              color: { enable_color_match: false, match_strength: 0 },
              lighting: {
                shadow_strength: 0.45,
                displacement_strength: 0,
                specular_strength: 0.3,
                specular_threshold: 180,
                light_pos_x: effectiveWarp?.light_pos_x ?? 0.62,
                light_pos_y: effectiveWarp?.light_pos_y ?? 0.32,
                light_height: effectiveWarp?.light_height ?? 55,
                light_contrast: effectiveWarp?.light_contrast ?? 50,
                light_highlight: effectiveWarp?.light_highlight ?? 60,
                light_softness: effectiveWarp?.light_softness ?? 55,
              },
              edge: { feather_px: 0 },
              render: { preserve_original_color: false },
            } : {}),
          }
          formData.append('config_json', JSON.stringify(configJson))
        }
        url = '/v1/mockup/render-adhoc'
      }

      const res = await fetch(url, { method: 'POST', body: formData })
      
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
      setResolvedTemplatePreviewUrl(null)
      setPreviewUrl(prev => {
        if (prev && isObjectUrl(prev)) URL.revokeObjectURL(prev)
        return URL.createObjectURL(blob)
      })
    } catch (e: any) {
      alert(`Lỗi kết nối: ${e.message}`)
    } finally {
      setLoading(false)
    }
  }

  const handleSaveAsTemplate = async () => {
    const slug = prompt('Nhập slug cho template mới (ví dụ: my-mug-01):');
    if (!slug) return;
    const name = prompt('Nhập tên hiển thị cho template:', slug);
    if (!name) return;

    setLoading(true);
    try {
      const previewDataUrl = previewUrl
        ? await fetch(previewUrl).then(async response => blobToDataUrl(await response.blob()))
        : null

      const urlAnalysisMeta = urlImportSummary ? {
        source_url: urlImportSummary.source_url,
        parsed: urlImportSummary.parsed,
        design_lookup_key: urlImportSummary.design_lookup_key,
        mockup_family_key: urlImportSummary.mockup_family_key || urlImportSummary.mockup_policy_key,
        mockup_policy_key: urlImportSummary.mockup_policy_key,
        mockup_view: urlImportSummary.mockup_view,
        design_url: libraryDesignUrl || urlImportSummary.design_url || null,
        mockup_url: libraryMockupUrl || urlImportSummary.mockup_url || null,
        design_source_url: urlImportSummary.design_source_url || null,
        design_source_mode: urlImportSummary.design_source_mode || null,
      } : null
      // Nếu user upload file local (chưa có trong library), tự động upload lên trước
      let resolvedMockupUrl = libraryMockupUrl || '';
      if (!resolvedMockupUrl && mockupFile) {
        const uploadForm = new FormData();
        uploadForm.append('file', mockupFile);

        const uploadRes = await fetch('/admin/library/bases/upload', {
          method: 'POST',
          body: uploadForm,
        });

        if (!uploadRes.ok) {
          let errMsg = 'Upload mockup lên thư viện thất bại';
          try {
            const errBody = await uploadRes.json();
            errMsg = errBody.detail || errBody.message || errMsg;
          } catch (_) { /* ignore parse error */ }
          throw new Error(errMsg);
        }

        const uploadData = await uploadRes.json();
        resolvedMockupUrl = uploadData.url; // e.g. /static/bases/filename.jpg
        // Cập nhật state để UI nhận biết ảnh đã nằm trong library
        setLibraryMockupUrl(resolvedMockupUrl);
      }

      if (!resolvedMockupUrl) {
        throw new Error('Chưa có ảnh mockup. Vui lòng upload hoặc chọn từ thư viện.');
      }

      const res = await fetch('/admin/templates/save-adhoc', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          slug,
          name,
          product_type: productType || 'mug',
          config: {
            print_area: effectivePrintArea,
            warp: effectiveWarp,
            ...(urlAnalysisMeta ? { url_analysis: urlAnalysisMeta } : {}),
          },
          mockup_url: resolvedMockupUrl,
          output_width: 1500,
          output_height: 1500,
          preview_data_url: previewDataUrl,
        }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.message || err.detail?.message || 'Lỗi khi lưu template');
      }

      alert('Đã lưu cấu hình thành Template thành công!');
      // Refresh templates
      const tRes = await fetch('/v1/templates');
      const tData = await tRes.json();
      setTemplates(tData.templates || []);
      setMode('template');
      setSelectedTemplate(slug);
      setShowEditor(false);
    } catch (e: any) {
      alert(`Lỗi: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleImportFromSourceUrl = async () => {
    const normalized = sourceProductUrl.trim()
    if (!normalized) return

    setUrlImporting(true)
    try {
      const res = await fetch('/admin/url-analysis/import', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ source_url: normalized }),
      })

      const data = await res.json().catch(() => ({}))
      if (!res.ok) {
        throw new Error(data?.message || data?.detail?.message || data?.detail || `HTTP ${res.status}`)
      }

      setUrlImportSummary(data)
      if (data.template_found && data.existing_template) {
        setMode('template')
        setSelectedTemplate(data.existing_template.template_id || data.existing_template.slug)
        setShowEditor(false)
        setMockupFile(null)
        setDesignFile(null)
        setLibraryMockupUrl(null)
        setLibraryDesignUrl(null)
        setPrintArea(null)
        setLockedSnapshot(null)
        setLiveEditorSnapshot(null)
        liveEditorSnapshotRef.current = null
        setPreviewUrl(prev => {
          if (prev && isObjectUrl(prev)) URL.revokeObjectURL(prev)
          return null
        })
        setResolvedTemplatePreviewUrl(data.existing_template.preview_url || null)
        setRenderTime(null)
        return
      }

      setResolvedTemplatePreviewUrl(null)
      setProductType(data.product_type || 'cylinder_ceramic')
      setMockupFile(null)
      setDesignFile(null)
      setLibraryMockupUrl(data.mockup_url || null)
      setLibraryDesignUrl(data.design_url || null)
      setPrintArea(data.print_area_preset || null)
      setWarpConfigObj((prev: any) => ({
        ...prev,
        ...(data.warp_config_preset || {}),
        product_type: data.product_type || prev.product_type,
      }))
      setLockedSnapshot(null)
      setLiveEditorSnapshot(null)
      liveEditorSnapshotRef.current = null
      if (data.mockup_url) {
        setShowEditor(true)
      }
    } catch (e: any) {
      alert(`Lỗi phân tích URL: ${e.message}`)
    } finally {
      setUrlImporting(false)
    }
  }

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
            Tự Cấu Hình
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
            <div className="input-group" style={{ padding: '1rem', background: 'rgba(59,130,246,0.08)', border: '1px solid rgba(59,130,246,0.25)', borderRadius: '8px', marginBottom: '1.5rem' }}>
              <label className="input-label" style={{ fontWeight: 'bold' }}>URL LiveView / CDN</label>
              <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'stretch' }}>
                <input
                  type="text"
                  className="input-field"
                  placeholder="Dán URL CDN Printerval để tự fill mockup và artwork"
                  value={sourceProductUrl}
                  onChange={e => setSourceProductUrl(e.target.value)}
                />
                <button
                  className="btn btn-outline"
                  style={{ whiteSpace: 'nowrap', padding: '0 0.9rem' }}
                  onClick={handleImportFromSourceUrl}
                  disabled={urlImporting || !sourceProductUrl.trim()}
                >
                  {urlImporting ? 'Đang phân tích...' : 'Phân tích URL'}
                </button>
              </div>
              <div style={{ marginTop: '0.5rem', fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                URL sẽ được phân tích để chọn mockup local trong `public/mockups` và tải artwork tham chiếu về local.
              </div>
              {urlImportSummary && (
                <div style={{ marginTop: '0.75rem', padding: '0.75rem', borderRadius: '8px', background: 'rgba(255,255,255,0.03)', fontSize: '0.8rem', lineHeight: 1.6 }}>
                  {urlImportSummary?.template_found && urlImportSummary?.existing_template ? (
                    <>
                      <div><strong>Template đã có:</strong> {urlImportSummary?.existing_template?.slug || '-'}</div>
                      <div><strong>Ảnh template:</strong> {urlImportSummary?.existing_template?.preview_url || '-'}</div>
                      <div><strong>View đã dùng:</strong> {urlImportSummary?.existing_template?.mockup_view || '-'}</div>
                    </>
                  ) : (
                    <>
                      <div><strong>Template:</strong> {urlImportSummary?.parsed?.template || '-'}</div>
                      <div><strong>Màu:</strong> {urlImportSummary?.parsed?.color_slug || urlImportSummary?.parsed?.color_hex || '-'}</div>
                      <div><strong>Mockup policy:</strong> {urlImportSummary?.mockup_policy_key || '-'}</div>
                      <div><strong>View đang chọn:</strong> {urlImportSummary?.mockup_view || '-'}</div>
                      <div><strong>Mockup local:</strong> {urlImportSummary?.mockup_url || '-'}</div>
                      <div><strong>Artwork local:</strong> {urlImportSummary?.design_url || '-'}</div>
                      <div><strong>Preset vùng in:</strong> {urlImportSummary?.print_area_preset_source || '-'} ({typeof urlImportSummary?.print_area_preset_confidence === 'number' ? `${Math.round(urlImportSummary.print_area_preset_confidence * 100)}%` : '-'})</div>
                      <div><strong>Mode tải artwork:</strong> {urlImportSummary?.design_source_mode || '-'}</div>
                      {urlImportSummary?.warning && (
                        <div style={{ marginTop: '0.35rem', color: '#fbbf24' }}>{urlImportSummary.warning}</div>
                      )}
                    </>
                  )}
                </div>
              )}
            </div>

            {/* STEP 1: MODULE SELECTION */}
            <div className="input-group" style={{ border: '1px solid var(--accent-color)', padding: '1rem', borderRadius: '8px', marginBottom: '1.5rem' }}>
              <label className="input-label" style={{ color: 'var(--accent-color)', fontWeight: 'bold' }}>Step 1: Chọn loại sản phẩm (Module)</label>
              <select 
                className="input-field" 
                value={productType} 
                onChange={e => {
                  setProductType(e.target.value);
                  setPrintArea(null);
                  setLockedSnapshot(null);
                  setLiveEditorSnapshot(null);
                  liveEditorSnapshotRef.current = null;
                  setMockupFile(null);
                  setLibraryMockupUrl(null);
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
              </select>
            </div>

            {/* STEP 2: UPLOAD OR LIB */}
            {productType && (
              <div className="input-group animate-in" style={{ padding: '1rem', background: 'rgba(255,255,255,0.03)', borderRadius: '8px' }}>
                <label className="input-label" style={{ fontWeight: 'bold' }}>Step 2: Chọn ảnh gốc (Mockup Base)</label>
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                  {!libraryMockupUrl ? (
                    <input 
                      type="file" 
                      accept="image/png, image/jpeg" 
                      className="input-field"
                      onChange={e => {
                        setMockupFile(e.target.files?.[0] || null);
                        setLibraryMockupUrl(null);
                        setLockedSnapshot(null);
                        setLiveEditorSnapshot(null);
                        liveEditorSnapshotRef.current = null;
                        if (e.target.files?.[0]) setShowEditor(true);
                      }}
                    />
                  ) : (
                    <div className="input-field" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', background: 'rgba(59,130,246,0.1)' }}>
                      <span style={{ fontSize: '0.8rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {libraryMockupUrl.split('/').pop()}
                      </span>
                      <button onClick={() => setLibraryMockupUrl(null)} style={{ background: 'none', border: 'none', color: 'var(--accent-color)', cursor: 'pointer' }}>×</button>
                    </div>
                  )}
                  <button className="btn btn-outline" onClick={() => setPickerType('bases')} style={{ padding: '0 0.75rem' }}>Lib</button>
                  {mockupPreviewUrl && (
                    <button className="btn btn-outline" onClick={() => setShowEditor(true)} style={{ padding: '0 0.75rem' }}>
                      Edit
                    </button>
                  )}
                </div>
                
                {mockupPreviewUrl && printArea ? (
                  <div style={{ marginTop: '0.5rem', fontSize: '0.8rem', color: '#10b981' }}>
                    ✓ Đã cấu hình Mesh & Print Area
                  </div>
                ) : mockupPreviewUrl && (
                  <div style={{ marginTop: '0.5rem', fontSize: '0.8rem', color: 'var(--accent-color)' }}>
                    ⚠ Cần Calibrate Area
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        <div className="input-group" style={{ marginTop: '1rem' }}>
          <label className="input-label">Step 3: Upload Design (Artwork)</label>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            {!libraryDesignUrl ? (
              <input 
                type="file" 
                accept="image/png, image/jpeg" 
                className="input-field"
                onChange={e => {
                  setDesignFile(e.target.files?.[0] || null);
                  setLibraryDesignUrl(null);
                }}
              />
            ) : (
              <div className="input-field" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', background: 'rgba(59,130,246,0.1)' }}>
                <span style={{ fontSize: '0.8rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {libraryDesignUrl.split('/').pop()}
                </span>
                <button onClick={() => setLibraryDesignUrl(null)} style={{ background: 'none', border: 'none', color: 'var(--accent-color)', cursor: 'pointer' }}>×</button>
              </div>
            )}
            <button className="btn btn-outline" onClick={() => setPickerType('artworks')} style={{ padding: '0 0.75rem' }}>Lib</button>
          </div>
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
          {(previewUrl || resolvedTemplatePreviewUrl) ? (
            <img src={previewUrl || resolvedTemplatePreviewUrl || ''} alt="Mockup Result" className="preview-image" />
          ) : loading ? (
            <div className="empty-state">
              <div style={{ fontSize: '3rem', marginBottom: '1rem', opacity: 0.5 }}>⏳</div>
              <div>Đang tự render mockup...</div>
            </div>
          ) : (
            <div className="empty-state">
              <div style={{ fontSize: '3rem', marginBottom: '1rem', opacity: 0.5 }}>🎨</div>
              <div>Bấm Gen Mockup để xem thiết bị hiển thị</div>
            </div>
          )}
        </div>
      </div>

      {/* Editor Modal */}
      {showEditor && mockupPreviewUrl && (
        <div className="modal-overlay" style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.85)', zIndex: 1000, display: 'flex', alignItems: 'center', justifyItems: 'center', padding: '2rem' }}>
          <div className="glass-panel" style={{ width: '100%', height: '100%', background: 'var(--bg-color)', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            <div style={{ padding: '1rem', borderBottom: '1px solid var(--border-color)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <button
                  className="btn btn-outline"
                  onClick={handleSaveAsTemplate}
                  disabled={loading || !hasEffectivePrintArea}
                >
                  Lưu Template
                </button>
                <h3 style={{ margin: 0 }}>Trình Chỉnh Vùng In & Mesh</h3>
              </div>
              <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
                <label className="btn btn-outline" style={{ display: 'inline-flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                  Upload Artwork
                  <input type="file" accept="image/png, image/jpeg" style={{ display: 'none' }} onChange={e => {
                    const f = e.target.files?.[0] || null;
                    setDesignFile(f);
                    setLibraryDesignUrl(null);
                  }} />
                </label>
                <label className="btn btn-outline" style={{ display: 'inline-flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                  Đổi Ảnh Gốc
                  <input type="file" accept="image/png, image/jpeg" style={{ display: 'none' }} onChange={e => {
                    const f = e.target.files?.[0] || null;
                    setMockupFile(f);
                    setLibraryMockupUrl(null);
                    setLockedSnapshot(null);
                    setLiveEditorSnapshot(null);
                    liveEditorSnapshotRef.current = null;
                    if (f) setShowEditor(true);
                  }} />
                </label>
                <button
                  className="btn btn-primary"
                  onClick={handleRender}
                  disabled={loading || (!designFile && !libraryDesignUrl) || !hasEffectivePrintArea}
                >
                  {loading ? 'Đang Gen...' : 'Gen Thử Mockup'}
                </button>
                <button className="btn btn-outline" onClick={() => setShowEditor(false)}>Đóng Trình Chỉnh</button>
              </div>
            </div>
            <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
              <div style={{ flex: 1 }}>
                <PrintAreaEditor 
                  imageUrl={mockupPreviewUrl} 
                  designFile={designFile}
                  designUrl={libraryDesignUrl}
                  productTypeProp={productType}
                  onProductTypeChange={setProductType}
                  initialPrintArea={printArea}
                  warpConfig={warpConfigObj}
                  onCoordinatesChange={setPrintArea}
                  onConfigChange={setWarpConfigObj}
                  onLiveSnapshotChange={setLiveEditorSnapshot}
                />
              </div>
              <aside style={{ width: '400px', borderLeft: '1px solid var(--border-color)', padding: '1rem', display: 'flex', flexDirection: 'column' }}>
                <h4>Xem nhanh kết quả</h4>
                <div style={{ flex: 1, background: '#000', borderRadius: '8px', overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  {(previewUrl || resolvedTemplatePreviewUrl) ? <img src={previewUrl || resolvedTemplatePreviewUrl || ''} style={{ width: '100%', height: '100%', objectFit: 'contain' }} /> : <small>Bấm Gen Thử</small>}
                </div>
              </aside>
            </div>
          </div>
        </div>
      )}

      {pickerType && (
        <LibraryPicker 
          type={pickerType} 
          onClose={() => setPickerType(null)} 
          onSelect={(item) => {
            if (pickerType === 'bases') {
              setLibraryMockupUrl(item.url);
              setMockupFile(null);
              setLockedSnapshot(null);
              setLiveEditorSnapshot(null);
              liveEditorSnapshotRef.current = null;
              setShowEditor(true);
            } else {
              setLibraryDesignUrl(item.url);
              setDesignFile(null);
            }
            setPickerType(null);
          }}
        />
      )}
    </div>
  )
}
