import { useState, useEffect, useCallback, useRef } from 'react';
import { useLayoutContext } from './Layout';
import { listGuests, createGuest, updateGuest, deleteGuest } from '../api';

const RISK_COLORS = {
  high:    { bg: '#ffeaea', color: '#c0392b', label: 'Yüksek' },
  medium:  { bg: '#fff7e6', color: '#e67e22', label: 'Orta'   },
  low:     { bg: '#eafaf1', color: '#27ae60', label: 'Düşük'  },
};

const EMPTY_FORM = {
  first_name: '', last_name: '', email: '', phone: '',
  nationality: '', gender: '', vip_status: false, notes: '',
  hotel: 'Resort Hotel', lead_time: 30, deposit_type: 'No Deposit',
  market_segment: 'Online TA', adults: 2, children: 0, babies: 0,
  stays_in_week_nights: 3, stays_in_weekend_nights: 1,
  is_repeated_guest: 0, previous_cancellations: 0, adr: 100.0,
};

function RiskBadge({ label }) {
  const cfg = RISK_COLORS[label] || { bg: '#f0f0f0', color: '#666', label: label || '—' };
  return (
    <span style={{
      background: cfg.bg, color: cfg.color,
      padding: '2px 8px', borderRadius: 10, fontSize: 11, fontWeight: 600,
    }}>
      {cfg.label}
    </span>
  );
}

function GuestForm({ initial, onSave, onCancel, saving }) {
  const [form, setForm] = useState(initial || EMPTY_FORM);

  function set(key, val) {
    setForm(prev => ({ ...prev, [key]: val }));
  }

  function handleSubmit(e) {
    e.preventDefault();
    onSave(form);
  }

  return (
    <form onSubmit={handleSubmit} style={{ display: 'grid', gap: 10 }}>
      <div className="small" style={{ fontWeight: 600, marginBottom: 4 }}>Kişisel Bilgiler</div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
        <label>
          <div className="small">Ad *</div>
          <input className="inputField" value={form.first_name}
            onChange={e => set('first_name', e.target.value)} required minLength={1} />
        </label>
        <label>
          <div className="small">Soyad *</div>
          <input className="inputField" value={form.last_name}
            onChange={e => set('last_name', e.target.value)} required minLength={1} />
        </label>
        <label>
          <div className="small">E-posta</div>
          <input className="inputField" type="email" value={form.email || ''}
            onChange={e => set('email', e.target.value)} />
        </label>
        <label>
          <div className="small">Telefon</div>
          <input className="inputField" value={form.phone || ''}
            onChange={e => set('phone', e.target.value)} />
        </label>
        <label>
          <div className="small">Milliyet (ISO-3)</div>
          <input className="inputField" maxLength={3} value={form.nationality || ''}
            onChange={e => set('nationality', e.target.value.toUpperCase())} placeholder="TUR" />
        </label>
        <label>
          <div className="small">Cinsiyet</div>
          <select className="inputField" value={form.gender || ''}
            onChange={e => set('gender', e.target.value)}>
            <option value="">—</option>
            <option value="M">Erkek</option>
            <option value="F">Kadın</option>
            <option value="other">Diğer</option>
          </select>
        </label>
      </div>
      <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
        <input type="checkbox" checked={!!form.vip_status}
          onChange={e => set('vip_status', e.target.checked)} />
        <span>VIP Misafir</span>
      </label>

      <div className="small" style={{ fontWeight: 600, marginTop: 8, marginBottom: 4 }}>Rezervasyon Bilgileri</div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
        <label>
          <div className="small">Otel</div>
          <select className="inputField" value={form.hotel}
            onChange={e => set('hotel', e.target.value)}>
            <option value="Resort Hotel">Resort Hotel</option>
            <option value="City Hotel">City Hotel</option>
          </select>
        </label>
        <label>
          <div className="small">Lead Time (gün)</div>
          <input className="inputField" type="number" min={0} value={form.lead_time}
            onChange={e => set('lead_time', Number(e.target.value))} />
        </label>
        <label>
          <div className="small">Depozito Türü</div>
          <select className="inputField" value={form.deposit_type}
            onChange={e => set('deposit_type', e.target.value)}>
            <option value="No Deposit">No Deposit</option>
            <option value="Non Refund">Non Refund</option>
            <option value="Refundable">Refundable</option>
          </select>
        </label>
        <label>
          <div className="small">Market Segmenti</div>
          <select className="inputField" value={form.market_segment}
            onChange={e => set('market_segment', e.target.value)}>
            <option value="Online TA">Online TA</option>
            <option value="Offline TA/TO">Offline TA/TO</option>
            <option value="Direct">Direct</option>
            <option value="Corporate">Corporate</option>
            <option value="Groups">Groups</option>
          </select>
        </label>
        <label>
          <div className="small">Yetişkin</div>
          <input className="inputField" type="number" min={0} value={form.adults}
            onChange={e => set('adults', Number(e.target.value))} />
        </label>
        <label>
          <div className="small">Çocuk</div>
          <input className="inputField" type="number" min={0} value={form.children}
            onChange={e => set('children', Number(e.target.value))} />
        </label>
        <label>
          <div className="small">Hafta içi gece</div>
          <input className="inputField" type="number" min={0} value={form.stays_in_week_nights}
            onChange={e => set('stays_in_week_nights', Number(e.target.value))} />
        </label>
        <label>
          <div className="small">Hafta sonu gece</div>
          <input className="inputField" type="number" min={0} value={form.stays_in_weekend_nights}
            onChange={e => set('stays_in_weekend_nights', Number(e.target.value))} />
        </label>
        <label>
          <div className="small">Önceki İptal</div>
          <input className="inputField" type="number" min={0} value={form.previous_cancellations}
            onChange={e => set('previous_cancellations', Number(e.target.value))} />
        </label>
        <label>
          <div className="small">ADR (€)</div>
          <input className="inputField" type="number" min={0} step={0.01} value={form.adr}
            onChange={e => set('adr', Number(e.target.value))} />
        </label>
      </div>
      <label>
        <div className="small">Notlar</div>
        <textarea className="inputField" rows={2} value={form.notes || ''}
          onChange={e => set('notes', e.target.value)}
          style={{ resize: 'vertical', width: '100%' }} />
      </label>

      <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
        <button type="submit" className="btn btn-primary" disabled={saving}>
          {saving ? 'Kaydediliyor…' : 'Kaydet'}
        </button>
        <button type="button" className="btn" onClick={onCancel}>İptal</button>
      </div>
    </form>
  );
}

function GuestDetail({ guest, apiKey, onUpdated, onDeleted }) {
  const [editing, setEditing]   = useState(false);
  const [saving, setSaving]     = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [error, setError]       = useState(null);

  async function handleSave(data) {
    setSaving(true);
    setError(null);
    try {
      await updateGuest(guest.id, data, apiKey);
      setEditing(false);
      onUpdated();
    } catch (err) {
      setError(err.message || 'Güncelleme başarısız.');
    } finally {
      setSaving(false);
    }
  }

  async function handleDelete() {
    if (!window.confirm(`${guest.first_name} ${guest.last_name} silinsin mi? Bu işlem geri alınamaz.`)) return;
    setDeleting(true);
    try {
      await deleteGuest(guest.id, apiKey);
      onDeleted();
    } catch (err) {
      setError(err.message || 'Silme başarısız.');
      setDeleting(false);
    }
  }

  const riskScore = guest.risk_score != null ? `${(guest.risk_score * 100).toFixed(1)}%` : '—';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      {!editing ? (
        <>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <div style={{ fontSize: 18, fontWeight: 700 }}>
                {guest.vip_status && <span title="VIP">⭐ </span>}
                {guest.first_name} {guest.last_name}
              </div>
              {guest.email && <div className="small" style={{ color: 'var(--c-text-muted, #888)' }}>{guest.email}</div>}
            </div>
            {guest.risk_label && <RiskBadge label={guest.risk_label} />}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, fontSize: 13 }}>
            <div><strong>Risk Skoru:</strong> {riskScore}</div>
            <div><strong>Otel:</strong> {guest.hotel || '—'}</div>
            <div><strong>Lead Time:</strong> {guest.lead_time != null ? `${guest.lead_time} gün` : '—'}</div>
            <div><strong>Depozito:</strong> {guest.deposit_type || '—'}</div>
            <div><strong>Telefon:</strong> {guest.phone || '—'}</div>
            <div><strong>Milliyet:</strong> {guest.nationality || '—'}</div>
            <div><strong>Market:</strong> {guest.market_segment || '—'}</div>
            <div><strong>ADR:</strong> {guest.adr != null ? `€${guest.adr.toFixed(2)}` : '—'}</div>
            <div><strong>Yetişkin:</strong> {guest.adults ?? '—'}</div>
            <div><strong>Önceki İptal:</strong> {guest.previous_cancellations ?? '—'}</div>
          </div>
          {guest.notes && (
            <div style={{ fontSize: 12, borderTop: '1px solid var(--c-border, #ddd)', paddingTop: 8 }}>
              <strong>Notlar:</strong> {guest.notes}
            </div>
          )}
          {error && <div style={{ color: '#c0392b', fontSize: 12 }}>{error}</div>}
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="btn btn-primary" style={{ fontSize: 12 }} onClick={() => setEditing(true)}>Düzenle</button>
            <button className="btn" style={{ fontSize: 12, color: '#c0392b' }} onClick={handleDelete} disabled={deleting}>
              {deleting ? 'Siliniyor…' : 'Sil'}
            </button>
          </div>
        </>
      ) : (
        <GuestForm initial={guest} onSave={handleSave} onCancel={() => setEditing(false)} saving={saving} />
      )}
    </div>
  );
}

export default function GuestPage() {
  const { runs } = useLayoutContext();
  const apiKey = runs.apiKey;

  const [guests, setGuests]           = useState([]);
  const [total, setTotal]             = useState(0);
  const [loading, setLoading]         = useState(false);
  const [error, setError]             = useState(null);
  const [search, setSearch]           = useState('');
  const [page, setPage]               = useState(0);
  const [selected, setSelected]       = useState(null);
  const [creating, setCreating]       = useState(false);
  const [saving, setSaving]           = useState(false);
  const abortRef = useRef(null);

  const PAGE_SIZE = 20;

  const fetchGuests = useCallback(async (q = search, pg = page) => {
    if (abortRef.current) abortRef.current.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    setLoading(true);
    setError(null);
    try {
      const data = await listGuests(
        { search: q, limit: PAGE_SIZE, offset: pg * PAGE_SIZE },
        apiKey,
        { signal: ctrl.signal },
      );
      setGuests(data.items ?? data ?? []);
      setTotal(data.total ?? (data.items ?? data ?? []).length);
    } catch (err) {
      if (err.name !== 'AbortError') setError(err.message || 'Misafirler yüklenemedi.');
    } finally {
      setLoading(false);
    }
  }, [apiKey, search, page]);

  useEffect(() => {
    fetchGuests(search, page);
    return () => abortRef.current?.abort();
  }, [apiKey, search, page]); // eslint-disable-line react-hooks/exhaustive-deps

  function handleSearch(e) {
    setSearch(e.target.value);
    setPage(0);
    setSelected(null);
  }

  async function handleCreate(data) {
    setSaving(true);
    setError(null);
    try {
      const created = await createGuest(data, apiKey);
      setCreating(false);
      fetchGuests(search, page);
      setSelected(created);
    } catch (err) {
      setError(err.message || 'Misafir oluşturulamadı.');
    } finally {
      setSaving(false);
    }
  }

  const totalPages = Math.ceil(total / PAGE_SIZE);

  return (
    <>
      <header className="pageHeader">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 12 }}>
          <div>
            <h1>👥 Misafir Yönetimi</h1>
            <p className="subtitle">
              Misafirleri kaydet, ara, düzenle ve iptal riski hesapla.
              {total > 0 && ` Toplam ${total} kayıt.`}
            </p>
          </div>
          <button
            className="btn btn-primary"
            onClick={() => { setCreating(true); setSelected(null); }}
          >
            + Yeni Misafir
          </button>
        </div>
      </header>

      {/* Search bar */}
      <div className="card" style={{ padding: '10px 16px' }}>
        <input
          className="inputField"
          placeholder="Ad, soyad veya e-posta ile ara…"
          value={search}
          onChange={handleSearch}
          style={{ maxWidth: 380 }}
          aria-label="Misafir ara"
        />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: selected || creating ? '1fr 380px' : '1fr', gap: 16, alignItems: 'start' }}>
        {/* Guest list */}
        <section className="card" style={{ padding: 0, overflow: 'hidden' }}>
          {loading && <div style={{ padding: 16 }}>Yükleniyor…</div>}
          {error  && <div style={{ padding: 16, color: '#c0392b' }}>{error}</div>}
          {!loading && guests.length === 0 && !error && (
            <div style={{ padding: 24, textAlign: 'center', color: 'var(--c-text-muted, #999)' }}>
              {search ? `"${search}" için sonuç bulunamadı.` : 'Henüz misafir kaydı yok.'}
            </div>
          )}
          {guests.length > 0 && (
            <div className="tableWrap">
              <table>
                <thead>
                  <tr>
                    <th>Ad Soyad</th>
                    <th>E-posta</th>
                    <th>Otel</th>
                    <th>Risk</th>
                    <th>Skor</th>
                    <th>VIP</th>
                  </tr>
                </thead>
                <tbody>
                  {guests.map(g => {
                    const isActive = selected?.id === g.id;
                    return (
                      <tr
                        key={g.id}
                        style={{
                          cursor: 'pointer',
                          background: isActive ? 'var(--c-accent-bg, #e0f0ff)' : undefined,
                          fontWeight: isActive ? 600 : 400,
                        }}
                        onClick={() => { setSelected(g); setCreating(false); }}
                        tabIndex={0}
                        onKeyDown={e => e.key === 'Enter' && (setSelected(g), setCreating(false))}
                        role="button"
                        aria-label={`${g.first_name} ${g.last_name} detayını göster`}
                        aria-pressed={isActive}
                      >
                        <td>{g.first_name} {g.last_name}</td>
                        <td style={{ fontSize: 12, color: 'var(--c-text-muted, #666)' }}>{g.email || '—'}</td>
                        <td style={{ fontSize: 12 }}>{g.hotel || '—'}</td>
                        <td>{g.risk_label ? <RiskBadge label={g.risk_label} /> : <span style={{ color: '#aaa' }}>—</span>}</td>
                        <td style={{ fontFamily: 'Consolas', fontSize: 12 }}>
                          {g.risk_score != null ? `${(g.risk_score * 100).toFixed(1)}%` : '—'}
                        </td>
                        <td style={{ textAlign: 'center' }}>{g.vip_status ? '⭐' : ''}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}

          {/* Pagination */}
          {totalPages > 1 && (
            <div style={{ display: 'flex', gap: 8, padding: 12, justifyContent: 'center' }}>
              <button className="btn" disabled={page === 0} onClick={() => setPage(p => p - 1)}>← Önceki</button>
              <span style={{ lineHeight: '28px', fontSize: 12 }}>{page + 1} / {totalPages}</span>
              <button className="btn" disabled={page >= totalPages - 1} onClick={() => setPage(p => p + 1)}>Sonraki →</button>
            </div>
          )}
        </section>

        {/* Detail / Create panel */}
        {(selected || creating) && (
          <aside className="card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
              <div className="small" style={{ fontWeight: 700 }}>
                {creating ? 'Yeni Misafir' : 'Misafir Detayı'}
              </div>
              <button
                style={{ background: 'none', border: 'none', cursor: 'pointer', fontSize: 14, color: 'var(--c-text-muted, #999)' }}
                onClick={() => { setSelected(null); setCreating(false); }}
                aria-label="Kapat"
              >✕</button>
            </div>

            {creating ? (
              <GuestForm
                onSave={handleCreate}
                onCancel={() => setCreating(false)}
                saving={saving}
              />
            ) : selected ? (
              <GuestDetail
                key={selected.id}
                guest={selected}
                apiKey={apiKey}
                onUpdated={() => { fetchGuests(search, page); setSelected(null); }}
                onDeleted={() => { fetchGuests(search, page); setSelected(null); }}
              />
            ) : null}
          </aside>
        )}
      </div>
    </>
  );
}
