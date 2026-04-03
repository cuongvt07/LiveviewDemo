import { useState, useEffect } from 'react';
import { Image as ImageIcon, X } from 'lucide-react';
import t from '../i18n/translate'

interface LibraryItem {
  name: string;
  url: string;
  size_bytes: number;
  modified_at: string;
}

interface LibraryPickerProps {
  type: 'bases' | 'artworks';
  onSelect: (item: LibraryItem) => void;
  onClose: () => void;
}

export default function LibraryPicker({ type, onSelect, onClose }: LibraryPickerProps) {
  const [items, setItems] = useState<LibraryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('');

  useEffect(() => {
    fetch(`/admin/library/${type}`)
      .then(res => res.json())
      .then(data => {
        setItems(Array.isArray(data) ? data : []);
        setLoading(false);
      })
      .catch(err => {
        console.error('Failed to fetch library:', err);
        setLoading(false);
      });
  }, [type]);

  const filteredItems = items.filter(item => 
    item.name.toLowerCase().includes(filter.toLowerCase())
  );

  return (
    <div className="modal-overlay" style={{ zIndex: 1100 }}>
      <div className="glass-panel" style={{ 
        width: '800px', 
        height: '600px', 
        display: 'flex', 
        flexDirection: 'column', 
        padding: '1.5rem',
        background: 'var(--bg-color)',
        boxShadow: '0 20px 50px rgba(0,0,0,0.5)'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
            <h3 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <ImageIcon size={20} />
              {type === 'bases' ? t('library_picker.title_bases') : t('library_picker.title_artworks')}
          </h3>
          <button className="btn-icon" onClick={onClose} style={{ background: 'transparent', border: 'none', color: 'var(--text-secondary)' }}>
            <X size={24} />
          </button>
        </div>

        <div style={{ marginBottom: '1rem' }}>
          <input 
            type="text" 
            placeholder={t('library_picker.search_placeholder')} 
            className="input-field" 
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          />
        </div>

        <div style={{ flex: 1, overflowY: 'auto', display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: '1rem', paddingRight: '0.5rem' }}>
          {loading ? (
            <div style={{ gridColumn: '1/-1', textAlign: 'center', padding: '2rem' }}>{t('library_picker.loading')}</div>
          ) : filteredItems.length === 0 ? (
            <div style={{ gridColumn: '1/-1', textAlign: 'center', padding: '2rem', opacity: 0.5 }}>
              {t('library_picker.no_items')}
            </div>
          ) : filteredItems.map(item => (
            <div 
              key={item.url} 
              className="library-item glass-panel" 
              style={{ padding: '0.5rem', cursor: 'pointer', transition: 'transform 0.2s' }}
              onClick={() => onSelect(item)}
            >
              <div style={{ width: '100%', aspectRatio: '1', background: 'rgba(255,255,255,0.05)', borderRadius: '4px', overflow: 'hidden', marginBottom: '0.5rem' }}>
                <img src={item.url} alt={item.name} style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
              </div>
              <div style={{ fontSize: '0.75rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={item.name}>
                {item.name}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
