import { useState, useEffect, useMemo, useRef } from 'react'
import t from '../i18n/translate'
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
      else reject(new Error(t('live_preview.error.cannot_read_preview')))
    }
    reader.onerror = () => reject(reader.error || new Error(t('live_preview.error.cannot_read_preview')))
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
  const renderRequestSeqRef = useRef(0)
  const [showEditor, setShowEditor] = useState(false)
  const [productType, setProductType] = useState('') // Initial empty state to enforce selection

  // Library picker state
  const [pickerType, setPickerType] = useState<'bases' | 'artworks' | null>(null)
  const [libraryMockupUrl, setLibraryMockupUrl] = useState<string | null>(null)
  const [libraryDesignUrl, setLibraryDesignUrl] = useState<string | null>(null)
  const [sourceProductUrl, setSourceProductUrl] = useState('')
  const [urlImporting, setUrlImporting] = useState(false)
  const [urlImportSummary, setUrlImportSummary] = useState<any | null>(null)
  const [templateLookupLoading, setTemplateLookupLoading] = useState(false)
  const [templateLookupResult, setTemplateLookupResult] = useState<any | null>(null)
  const [templateMatches, setTemplateMatches] = useState<any[]>([])
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
    renderRequestSeqRef.current += 1
    setPreviewUrl(prev => {
      if (prev && isObjectUrl(prev)) URL.revokeObjectURL(prev)
      return null
    })
    setLoading(false)
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
    showEditor,
  ])

  const handleRender = async () => {
    if (!designFile && !libraryDesignUrl) return
    if (mode === 'template' && !selectedTemplate) return
    if (mode === 'adhoc' && !mockupFile && !libraryMockupUrl) return
    if (mode === 'adhoc' && !effectivePrintArea) return

    const requestSeq = ++renderRequestSeqRef.current
    const isStale = () => requestSeq !== renderRequestSeqRef.current

    setLoading(true)

    const buildFormData = () => {
      const fd = new FormData()
      if (designFile) fd.append('design_image', designFile)
      else if (libraryDesignUrl) fd.append('design_url', libraryDesignUrl)
      return fd
    }

    try {
      let url = ''
      const fd = buildFormData()
      if (mode === 'template') {
        fd.append('template_id', selectedTemplate)
        url = '/v1/mockup/render'
      } else {
        if (mockupFile) {
          fd.append('mockup_image', mockupFile)
        } else if (libraryMockupUrl) {
          fd.append('mockup_url', libraryMockupUrl)
        }
        fd.append('output_format', 'png')
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
          fd.append('config_json', JSON.stringify(configJson))
        }

        // Keep queue-based async render for normal page mode.
        // In editor modal, use sync render to avoid queue lag while dragging mesh.
        const useAsyncAdhoc = !showEditor
        url = useAsyncAdhoc ? '/v1/mockup/render-async-adhoc' : '/v1/mockup/render-adhoc'
      }

      let res = await fetch(url, { method: 'POST', body: fd })
      if (isStale()) return

      if (!res.ok && url.endsWith('/render-async-adhoc')) {
        console.warn(`/render-async-adhoc failed (${res.status}). Falling back to /render-adhoc`)
        try {
          res = await fetch('/v1/mockup/render-adhoc', { method: 'POST', body: fd })
        } catch (_e) {
          // handled by generic error branch below
        }
      }
      if (isStale()) return

      if (!res.ok) {
        let errMsg = 'Unknown error'
        try {
          const err = await res.json()
          errMsg = err.message || err.detail?.message || err.detail || JSON.stringify(err)
        } catch (_e) {
          errMsg = `HTTP ${res.status}: ${res.statusText}`
        }
        if (!isStale()) {
          alert(t('error.render', { msg: errMsg }))
          setLoading(false)
        }
        return
      }

      const contentType = res.headers.get('content-type') || ''
      if (mode === 'template' || contentType.startsWith('image/')) {
        const timeStr = res.headers.get('X-Processing-Time-Ms')
        if (isStale()) return
        if (timeStr) setRenderTime(parseInt(timeStr, 10))

        const blob = await res.blob()
        if (isStale()) return

        setResolvedTemplatePreviewUrl(null)
        setPreviewUrl(prev => {
          if (prev && isObjectUrl(prev)) URL.revokeObjectURL(prev)
          return URL.createObjectURL(blob)
        })
      } else {
        const body = await res.json().catch(() => ({}))
        if (isStale()) return

        const jobId = body?.job_id
        if (!jobId) {
          if (!isStale()) {
            alert(t('error.no_job_id'))
            setLoading(false)
          }
          return
        }

        const pollUrl = `/v1/mockup/render-status/${jobId}`
        let lastPreview: string | null = null

        for (let i = 0; i < 120; i++) {
          if (isStale()) return
          try {
            await new Promise(r => setTimeout(r, i === 0 ? 300 : 500))
            if (isStale()) return

            const st = await fetch(pollUrl)
            if (!st.ok) continue

            const data = await st.json()
            if (isStale()) return

            if (data?.result?.processing_time_ms) {
              setRenderTime(Number(data.result.processing_time_ms))
            }

            if (data?.result?.preview_url && data.result.preview_url !== lastPreview) {
              lastPreview = data.result.preview_url
              const p = await fetch(data.result.preview_url)
              if (p.ok) {
                const blob = await p.blob()
                if (isStale()) return
                setPreviewUrl(prev => {
                  if (prev && isObjectUrl(prev)) URL.revokeObjectURL(prev)
                  return URL.createObjectURL(blob)
                })
              }
            }

            if (data?.status === 'done' && data?.result?.final_url) {
              const f = await fetch(data.result.final_url)
              if (f.ok) {
                const blob = await f.blob()
                if (isStale()) return
                setPreviewUrl(prev => {
                  if (prev && isObjectUrl(prev)) URL.revokeObjectURL(prev)
                  return URL.createObjectURL(blob)
                })
              }
              break
            }
          } catch (_e) {
            if (isStale()) return
          }
        }
      }
    } catch (e: any) {
      if (!isStale()) {
        alert(t('error.connection', { msg: e.message }))
      }
    } finally {
      if (!isStale()) {
        setLoading(false)
      }
    }
  }
  const handleSaveAsTemplate = async () => {
    const slug = prompt(t('live_preview.prompt.slug'));
    if (!slug) return;
    const name = prompt(t('live_preview.prompt.name'), slug);
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
      // Náº¿u user upload file local (chÆ°a cÃ³ trong library), tá»± Ä‘á»™ng upload lÃªn trÆ°á»›c
      let resolvedMockupUrl = libraryMockupUrl || '';
      if (!resolvedMockupUrl && mockupFile) {
        const uploadForm = new FormData();
        uploadForm.append('file', mockupFile);

        const uploadRes = await fetch('/admin/library/bases/upload', {
          method: 'POST',
          body: uploadForm,
        });

        if (!uploadRes.ok) {
          let errMsg = t('live_preview.upload.upload_error');
          try {
            const errBody = await uploadRes.json();
            errMsg = errBody.detail || errBody.message || errMsg;
          } catch (_) { /* ignore parse error */ }
          throw new Error(errMsg);
        }

        const uploadData = await uploadRes.json();
        resolvedMockupUrl = uploadData.url; // e.g. /static/bases/filename.jpg
        // Cáº­p nháº­t state Ä‘á»ƒ UI nháº­n biáº¿t áº£nh Ä‘Ã£ náº±m trong library
        setLibraryMockupUrl(resolvedMockupUrl);
      }

      if (!resolvedMockupUrl) {
        throw new Error(t('live_preview.upload.no_mockup'));
      }

      const res = await fetch('/admin/templates/save-adhoc', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        // If printArea contains mesh_control_dst/src, include mesh payload normalized to template output size
        body: (() => {
          const cfg: any = { print_area: effectivePrintArea, warp: effectiveWarp };
          const ptsDst = effectivePrintArea?.mesh_control_dst;
          const ptsSrc = effectivePrintArea?.mesh_control_src;
          const OUT_W = 1500, OUT_H = 1500;
          if (Array.isArray(ptsDst) && ptsDst.length > 0) {
            const n = ptsDst.length;
            let side = Math.round(Math.sqrt(n));
            let cols = Math.max(1, side - 1);
            let rows = Math.max(1, side - 1);
            if (side * side !== n) {
              // fallback: estimate cols by scanning unique x counts per row if possible
              cols = Math.max(1, Math.round(Math.sqrt(n)));
              rows = Math.max(1, Math.ceil(n / cols) - 1);
            }
            const points = ptsDst.map((p: number[]) => [Number(p[0]) / OUT_W, Number(p[1]) / OUT_H, false]);
            cfg.mesh = {
              enabled: true,
              cols: cols,
              rows: rows,
              spacing: 'cosine',
              tension: 0.35,
              corner_blend_radius: 0.08,
              symmetry_lock: true,
              max_displacement: 0.15,
              points,
            };
          } else if (Array.isArray(ptsSrc) && ptsSrc.length > 0) {
            const n = ptsSrc.length;
            let side = Math.round(Math.sqrt(n));
            let cols = Math.max(1, side - 1);
            let rows = Math.max(1, side - 1);
            if (side * side !== n) {
              cols = Math.max(1, Math.round(Math.sqrt(n)));
              rows = Math.max(1, Math.ceil(n / cols) - 1);
            }
            const points = ptsSrc.map((p: number[]) => [Number(p[0]) / OUT_W, Number(p[1]) / OUT_H, false]);
            cfg.mesh = {
              enabled: true,
              cols: cols,
              rows: rows,
              spacing: 'cosine',
              tension: 0.35,
              corner_blend_radius: 0.08,
              symmetry_lock: true,
              max_displacement: 0.15,
              points,
            };
          }

          return JSON.stringify({
            slug,
            name,
            product_type: productType || 'mug',
            config: Object.assign(cfg, (urlAnalysisMeta ? { url_analysis: urlAnalysisMeta } : {})),
            mockup_url: resolvedMockupUrl,
            output_width: OUT_W,
            output_height: OUT_H,
            preview_data_url: previewDataUrl,
          });
        })(),
      });

        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.message || err.detail?.message || t('save.error.default'));
        }

      alert(t('success.template_saved'));
      // Refresh templates
      const tRes = await fetch('/v1/templates');
      const tData = await tRes.json();
      setTemplates(tData.templates || []);
      setMode('template');
      setSelectedTemplate(slug);
      setShowEditor(false);
    } catch (e: any) {
      alert(t('save.error', { msg: e.message }));
    } finally {
      setLoading(false);
    }
  };

  const clearBlobPreview = () => {
    setPreviewUrl(prev => {
      if (prev && isObjectUrl(prev)) URL.revokeObjectURL(prev)
      return null
    })
    setRenderTime(null)
  }

  const handleSelectMatchedTemplate = (template: any) => {
    setSelectedTemplate(template?.template_id || template?.slug || '')
    setResolvedTemplatePreviewUrl(template?.preview_url || null)
    clearBlobPreview()
  }

  const handleLookupTemplatesFromUrl = async () => {
    const normalized = sourceProductUrl.trim()
    if (!normalized) return

    setTemplateLookupLoading(true)
    try {
      setUrlImportSummary(null)
      const res = await fetch('/admin/url-analysis/template-matches', {
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

      const matches = Array.isArray(data?.matching_templates) ? data.matching_templates : []
      setMode('template')
      setTemplateLookupResult(data)
      setTemplateMatches(matches)
      setDesignFile(null)
      setLibraryDesignUrl(data?.design_url || null)
      setShowEditor(false)
      setMockupFile(null)
      setLibraryMockupUrl(null)
      setPrintArea(null)
      setLockedSnapshot(null)
      setLiveEditorSnapshot(null)
      liveEditorSnapshotRef.current = null
      clearBlobPreview()

      if (matches.length > 0) {
        const firstMatch = matches[0]
        setSelectedTemplate(firstMatch.template_id || firstMatch.slug || '')
        setResolvedTemplatePreviewUrl(data?.selected_preview_url || firstMatch.preview_url || null)
      } else {
        setSelectedTemplate('')
        setResolvedTemplatePreviewUrl(null)
      }
    } catch (e: any) {
      alert(t('lookup.template_url_error', { msg: e.message }))
    } finally {
      setTemplateLookupLoading(false)
    }
  }

  const handleImportFromSourceUrl = async () => {
    const normalized = sourceProductUrl.trim()
    if (!normalized) return

    setUrlImporting(true)
    try {
      setTemplateLookupResult(null)
      setTemplateMatches([])
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
      alert(t('lookup.import_error', { msg: e.message }))
    } finally {
      setUrlImporting(false)
    }
  }

  return (
    <div style={{ display: 'flex', gap: '2rem', height: '100%' }}>
      {/* Settings Form */}
      <div className="glass-panel" style={{ width: '400px', padding: '1.5rem', display: 'flex', flexDirection: 'column', overflowY: 'auto' }}>
        <h2 style={{ marginTop: 0, marginBottom: '1.5rem', fontSize: '1.25rem' }}>{t('live_preview.render_settings')}</h2>
        
        {/* Mode Switcher */}
        <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1.5rem', background: 'rgba(0,0,0,0.2)', padding: '0.25rem', borderRadius: '8px' }}>
          <button 
            style={{ flex: 1, padding: '0.5rem', borderRadius: '6px', border: 'none', background: mode === 'template' ? 'var(--surface-color)' : 'transparent', color: mode === 'template' ? 'white' : 'var(--text-secondary)', cursor: 'pointer', transition: 'all 0.2s' }}
            onClick={() => setMode('template')}
          >
            {t('live_preview.mode.from_template')}
          </button>
          <button 
            style={{ flex: 1, padding: '0.5rem', borderRadius: '6px', border: 'none', background: mode === 'adhoc' ? 'var(--surface-color)' : 'transparent', color: mode === 'adhoc' ? 'white' : 'var(--text-secondary)', cursor: 'pointer', transition: 'all 0.2s' }}
            onClick={() => setMode('adhoc')}
          >
            {t('live_preview.mode.adhoc')}
          </button>
        </div>

        {mode === 'template' ? (
          <>
            <div className="input-group" style={{ padding: '1rem', background: 'rgba(59,130,246,0.08)', border: '1px solid rgba(59,130,246,0.25)', borderRadius: '8px', marginBottom: '1rem' }}>
              <label className="input-label" style={{ fontWeight: 'bold' }}>{t('live_preview.url_label')}</label>
              <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'stretch' }}>
                <input
                  type="text"
                  className="input-field"
                  placeholder={t('live_preview.placeholder.cdn_url')}
                  value={sourceProductUrl}
                  onChange={e => setSourceProductUrl(e.target.value)}
                />
                <button
                  className="btn btn-outline"
                  style={{ whiteSpace: 'nowrap', padding: '0 0.9rem' }}
                  onClick={handleLookupTemplatesFromUrl}
                  disabled={templateLookupLoading || !sourceProductUrl.trim()}
                >
                  {templateLookupLoading ? t('live_preview.url_lookup.sending') : t('live_preview.url_lookup.send_button')}
                </button>
              </div>
                <div style={{ marginTop: '0.5rem', fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                  {t('live_preview.help.url_analysis')}
                </div>
              {templateLookupResult && (
                <div style={{ marginTop: '0.75rem', padding: '0.75rem', borderRadius: '8px', background: 'rgba(255,255,255,0.03)', fontSize: '0.8rem', lineHeight: 1.6 }}>
                  <div><strong>{t('live_preview.template')}</strong> {templateLookupResult?.parsed?.template || '-'}</div>
                  <div><strong>{t('live_preview.template_color')}</strong> {templateLookupResult?.parsed?.color_slug || templateLookupResult?.parsed?.color_hex || '-'}</div>
                  <div><strong>{t('live_preview.template_count')}</strong> {templateLookupResult?.template_count ?? 0}</div>
                  <div><strong>{t('live_preview.lookup_key')}</strong> {templateLookupResult?.design_lookup_key || '-'}</div>
                  {templateLookupResult?.warning && (
                    <div style={{ marginTop: '0.35rem', color: '#fbbf24' }}>{templateLookupResult.warning}</div>
                  )}
                </div>
              )}
            </div>

            {templateMatches.length > 0 && (
              <div className="input-group" style={{ marginBottom: '1rem' }}>
                <label className="input-label">{t('live_preview.matches.title')}</label>
                <div style={{ display: 'grid', gap: '0.75rem', maxHeight: '320px', overflowY: 'auto' }}>
                  {templateMatches.map((template) => {
                    const isActive = selectedTemplate === (template.template_id || template.slug)
                    return (
                      <button
                        key={template.template_id || template.slug}
                        type="button"
                        onClick={() => handleSelectMatchedTemplate(template)}
                        style={{
                          display: 'flex',
                          gap: '0.75rem',
                          alignItems: 'center',
                          width: '100%',
                          padding: '0.75rem',
                          borderRadius: '10px',
                          border: isActive ? '1px solid var(--accent-color)' : '1px solid rgba(255,255,255,0.08)',
                          background: isActive ? 'rgba(59,130,246,0.12)' : 'rgba(255,255,255,0.03)',
                          color: 'inherit',
                          cursor: 'pointer',
                          textAlign: 'left',
                        }}
                      >
                        <div style={{ width: '84px', height: '84px', borderRadius: '8px', overflow: 'hidden', background: 'rgba(0,0,0,0.2)', flexShrink: 0 }}>
                          {template.preview_url ? (
                            <img src={template.preview_url} alt={template.name || template.slug} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                          ) : (
                            <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>{t('live_preview.matches.no_preview')}</div>
                          )}
                        </div>
                        <div style={{ minWidth: 0 }}>
                          <div style={{ fontWeight: 700, marginBottom: '0.2rem' }}>{template.name || template.slug}</div>
                          <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{t('live_preview.slug_label')} {template.slug || '-'}</div>
                          <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{t('live_preview.view_label')} {template.mockup_view || '-'}</div>
                        </div>
                      </button>
                    )
                  })}
                </div>
              </div>
            )}

            <div className="input-group">
              <label className="input-label">{t('live_preview.or_manual')}</label>
              <select 
                className="input-field" 
                value={selectedTemplate} 
                onChange={e => setSelectedTemplate(e.target.value)}
              >
                <option value="">{t('live_preview.select_template_placeholder')}</option>
                {templates.length === 0 && <option value="">{t('template_manager.loading')}</option>}
                {templates.map(t => (
                  <option key={t.id} value={t.id}>{t.name}</option>
                ))}
              </select>
            </div>
          </>
        ) : (
          <div className="adhoc-workflow">
            <div className="input-group" style={{ padding: '1rem', background: 'rgba(59,130,246,0.08)', border: '1px solid rgba(59,130,246,0.25)', borderRadius: '8px', marginBottom: '1.5rem' }}>
              <label className="input-label" style={{ fontWeight: 'bold' }}>{t('live_preview.url_label')}</label>
              <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'stretch' }}>
                <input
                  type="text"
                  className="input-field"
                  placeholder={t('live_preview.placeholder.cdn_url')}
                  value={sourceProductUrl}
                  onChange={e => setSourceProductUrl(e.target.value)}
                />
                <button
                  className="btn btn-outline"
                  style={{ whiteSpace: 'nowrap', padding: '0 0.9rem' }}
                  onClick={handleImportFromSourceUrl}
                  disabled={urlImporting || !sourceProductUrl.trim()}
                >
                  {urlImporting ? t('live_preview.url_lookup.sending') : t('live_preview.url_lookup.send_button')}
                </button>
              </div>
              <div style={{ marginTop: '0.5rem', fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                {t('live_preview.url_analysis_hint')}
              </div>
              {urlImportSummary && (
                <div style={{ marginTop: '0.75rem', padding: '0.75rem', borderRadius: '8px', background: 'rgba(255,255,255,0.03)', fontSize: '0.8rem', lineHeight: 1.6 }}>
                  {urlImportSummary?.template_found && urlImportSummary?.existing_template ? (
                    <>
                      <div><strong>{t('live_preview.template_found')}</strong> {urlImportSummary?.existing_template?.slug || '-'}</div>
                      <div><strong>{t('live_preview.template_image')}</strong> {urlImportSummary?.existing_template?.preview_url || '-'}</div>
                      <div><strong>{t('live_preview.template_view')}</strong> {urlImportSummary?.existing_template?.mockup_view || '-'}</div>
                    </>
                  ) : (
                    <>
                      <div><strong>{t('live_preview.template')}</strong> {urlImportSummary?.parsed?.template || '-'}</div>
                      <div><strong>{t('live_preview.color')}</strong> {urlImportSummary?.parsed?.color_slug || urlImportSummary?.parsed?.color_hex || '-'}</div>
                      <div><strong>{t('live_preview.mockup_policy')}</strong> {urlImportSummary?.mockup_policy_key || '-'}</div>
                      <div><strong>{t('live_preview.mockup_view')}</strong> {urlImportSummary?.mockup_view || '-'}</div>
                      <div><strong>{t('live_preview.mockup_local')}</strong> {urlImportSummary?.mockup_url || '-'}</div>
                      <div><strong>{t('live_preview.artwork_local')}</strong> {urlImportSummary?.design_url || '-'}</div>
                      <div><strong>{t('live_preview.print_area_preset')}</strong> {urlImportSummary?.print_area_preset_source || '-'} ({typeof urlImportSummary?.print_area_preset_confidence === 'number' ? `${Math.round(urlImportSummary.print_area_preset_confidence * 100)}%` : '-'})</div>
                      <div><strong>{t('live_preview.design_source_mode')}</strong> {urlImportSummary?.design_source_mode || '-'}</div>
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
              <label className="input-label" style={{ color: 'var(--accent-color)', fontWeight: 'bold' }}>{t('live_preview.step1_select_module')}</label>
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
                <option value="">{t('live_preview.select_module_placeholder')}</option>
                <optgroup label={t('live_preview.module_groups.mugs')}>
                  <option value="cylinder_ceramic">{t('print_area.products.mugs.ceramic')}</option>
                  <option value="cylinder_glass">{t('print_area.products.mugs.glass')}</option>
                  <option value="cylinder_travel">{t('print_area.products.mugs.travel')}</option>
                </optgroup>
                <optgroup label={t('live_preview.module_groups.clothes')}>
                  <option value="apparel_cotton">{t('print_area.products.clothes.tshirt')}</option>
                  <option value="apparel_hoodie">{t('print_area.products.clothes.hoodie')}</option>
                  <option value="apparel_totebag">{t('print_area.products.clothes.totebag')}</option>
                </optgroup>
              </select>
            </div>

            {/* STEP 2: UPLOAD OR LIB */}
            {productType && (
              <div className="input-group animate-in" style={{ padding: '1rem', background: 'rgba(255,255,255,0.03)', borderRadius: '8px' }}>
                <label className="input-label" style={{ fontWeight: 'bold' }}>{t('live_preview.step_2_label')}</label>
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
                  <button className="btn btn-outline" onClick={() => setPickerType('bases')} style={{ padding: '0 0.75rem' }}>{t('live_preview.lib_button')}</button>
                  {mockupPreviewUrl && (
                    <button className="btn btn-outline" onClick={() => setShowEditor(true)} style={{ padding: '0 0.75rem' }}>
                      {t('live_preview.edit_button')}
                    </button>
                  )}
                </div>
                
                {mockupPreviewUrl && printArea ? (
                  <div style={{ marginTop: '0.5rem', fontSize: '0.8rem', color: '#10b981' }}>
                    {t('live_preview.preview.configured')}
                  </div>
                ) : mockupPreviewUrl && (
                  <div style={{ marginTop: '0.5rem', fontSize: '0.8rem', color: 'var(--accent-color)' }}>
                    {t('live_preview.preview.needs_calibrate')}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        <div style={{ marginTop: 'auto', paddingTop: '1.5rem' }}>
          <button 
            className="btn btn-primary" 
            style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '0.5rem', padding: '1rem' }}
            onClick={handleRender}
            disabled={loading || !isValid}
          >
            {loading ? t('live_preview.gen.loading') : t('live_preview.gen.button')}
          </button>
        </div>
      </div>

      {/* Preview Area */}
      <div className="glass-panel" style={{ flex: 1, padding: '1.5rem', display: 'flex', flexDirection: 'column' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
          <h2 style={{ margin: 0, fontSize: '1.25rem' }}>{t('live_preview.result.title')}</h2>
          {renderTime !== null && (
            <div style={{ background: 'rgba(59, 130, 246, 0.12)', color: 'var(--accent-color)', padding: '0.25rem 0.75rem', borderRadius: '100px', fontSize: '0.875rem', fontWeight: 600 }}>
              {t('live_preview.preview.time_label', { ms: renderTime })}
            </div>
          )}
        </div>
        
        <div className="preview-container" style={{ flex: 1, maxWidth: 'none', background: 'rgba(0,0,0,0.1)' }}>
          {(previewUrl || resolvedTemplatePreviewUrl) ? (
            <img src={previewUrl || resolvedTemplatePreviewUrl || ''} alt={t('live_preview.result.alt')} className="preview-image" />
          ) : loading ? (
            <div className="empty-state">
              <div style={{ fontSize: '3rem', marginBottom: '1rem', opacity: 0.5 }}>â³</div>
              <div>{t('live_preview.result.loading_hint')}</div>
            </div>
          ) : (
            <div className="empty-state">
              <div style={{ fontSize: '3rem', marginBottom: '1rem', opacity: 0.5 }}>🎉</div>
              <div>{t('live_preview.gen.empty_hint')}</div>
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
                  {t('live_preview.modal.save_template')}
                </button>
                <h3 style={{ margin: 0 }}>{t('live_preview.modal.editor_title')}</h3>
              </div>
              <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
                <label className="btn btn-outline" style={{ display: 'inline-flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                  {t('live_preview.modal.upload_artwork')}
                  <input type="file" accept="image/png, image/jpeg" style={{ display: 'none' }} onChange={e => {
                    const f = e.target.files?.[0] || null;
                    setDesignFile(f);
                    setLibraryDesignUrl(null);
                  }} />
                </label>
                <label className="btn btn-outline" style={{ display: 'inline-flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                  {t('live_preview.modal.change_mockup')}
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
                  {loading ? t('live_preview.gen.loading') : t('live_preview.modal.gen_test')}
                </button>
                <button className="btn btn-outline" onClick={() => setShowEditor(false)}>{t('live_preview.modal.close_editor')}</button>
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
                  onLiveSnapshotChange={(snapshot) => {
                    liveEditorSnapshotRef.current = snapshot
                    setLiveEditorSnapshot(snapshot)
                  }}
                />
              </div>
              <aside style={{ width: '400px', borderLeft: '1px solid var(--border-color)', padding: '1rem', display: 'flex', flexDirection: 'column' }}>
                <h4>{t('live_preview.modal.quick_preview_title')}</h4>
                <div style={{ flex: 1, background: '#000', borderRadius: '8px', overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  {(previewUrl || resolvedTemplatePreviewUrl) ? <img src={previewUrl || resolvedTemplatePreviewUrl || ''} style={{ width: '100%', height: '100%', objectFit: 'contain' }} /> : <small>{t('live_preview.modal.quick_preview_hint')}</small>}
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
