import { useCallback, useEffect, useState } from 'react';
import { createGuest, listGuests, updateGuest } from '../api';
import { useLayoutContext } from './Layout';

const PAGE_SIZE = 12;

const HOTELS = ['City Hotel', 'Resort Hotel'];
const DEPOSIT_TYPES = ['No Deposit', 'Non Refund', 'Refundable'];
const SEGMENTS = ['Online TA', 'Direct', 'Corporate', 'Groups', 'Offline TA/TO'];

const INITIAL_FORM = {
  first_name: '',
  last_name: '',
  email: '',
  phone: '',
  nationality: '',
  identity_no: '',
  birth_date: '',
  gender: '',
  vip_status: false,
  notes: '',
  hotel: 'City Hotel',
  lead_time: 30,
  deposit_type: 'No Deposit',
  market_segment: 'Online TA',
  adults: 2,
  children: 0,
  babies: 0,
  stays_in_week_nights: 2,
  stays_in_weekend_nights: 1,
  is_repeated_guest: 0,
  previous_cancellations: 0,
  adr: '',
};

function mapGuestToForm(guest) {
  const merged = { ...INITIAL_FORM, ...guest };
  return {
    ...merged,
    birth_date: merged.birth_date || '',
    adr: merged.adr == null ? '' : String(merged.adr),
  };
}

function toPayload(form) {
  return {
    first_name: String(form.first_name || '').trim(),
    last_name: String(form.last_name || '').trim(),
    email: form.email || null,
    phone: form.phone || null,
    nationality: form.nationality || null,
    identity_no: form.identity_no || null,
    birth_date: form.birth_date || null,
    gender: form.gender || null,
    vip_status: !!form.vip_status,
    notes: form.notes || null,
    hotel: form.hotel,
    lead_time: Number(form.lead_time || 0),
    deposit_type: form.deposit_type,
    market_segment: form.market_segment,
    adults: Number(form.adults || 1),
    children: Number(form.children || 0),
    babies: Number(form.babies || 0),
    stays_in_week_nights: Number(form.stays_in_week_nights || 0),
    stays_in_weekend_nights: Number(form.stays_in_weekend_nights || 0),
    is_repeated_guest: Number(form.is_repeated_guest || 0),
    previous_cancellations: Number(form.previous_cancellations || 0),
    adr: form.adr === '' ? null : Number(form.adr),
  };
}

function RiskBadge({ label, score }) {
  if (!label) return <span className="textMuted">-</span>;
  const mod = label === 'high' ? 'riskHigh' : label === 'medium' ? 'riskMed' : 'riskLow';
  const text = label === 'high' ? 'YUKSEK' : label === 'medium' ? 'ORTA' : 'DUSUK';
  const suffix = score == null ? '' : ` %${Math.round(score * 100)}`;
  return <span className={`riskBadge ${mod}`}>{`${text}${suffix}`}</span>;
}

const thS = 'guestTh';
const tdS = 'guestTd';

export default function GuestsPage() {
  const { runs, auth } = useLayoutContext();
  const apiKey = runs.apiKey;

  const [guests, setGuests] = useState([]);
  const [total, setTotal] = useState(0);
  const [search, setSearch] = useState('');
  const [offset, setOffset] = useState(0);
  const [listLoading, setListLoading] = useState(false);
  const [listError, setListError] = useState('');

  const [editingId, setEditingId] = useState(null);
  const [form, setForm] = useState(INITIAL_FORM);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState('');
  const [saveOk, setSaveOk] = useState('');

  const handleApiError = useCallback((err) => {
    if (err?.status === 401) {
      auth.handleAuthFailure?.(err);
      return true;
    }
    return false;
  }, [auth]);

  const loadGuests = useCallback(async (query, nextOffset) => {
    if (!apiKey) return;
    setListLoading(true);
    setListError('');
    try {
      const result = await listGuests(
        {
          search: query || undefined,
          limit: PAGE_SIZE,
          offset: nextOffset,
        },
        apiKey,
      );
      setGuests(result?.items || []);
      setTotal(Number(result?.total || 0));
    } catch (err) {
      if (!handleApiError(err)) {
        setListError(err.message || 'Misafirler yuklenemedi.');
      }
    } finally {
      setListLoading(false);
    }
  }, [apiKey, handleApiError]);

  useEffect(() => {
    setOffset(0);
    loadGuests('', 0);
  }, [apiKey, loadGuests]);

  function handleSearchChange(event) {
    const query = event.target.value;
    setSearch(query);
    setOffset(0);
    loadGuests(query, 0);
  }

  function resetForm() {
    setEditingId(null);
    setForm(INITIAL_FORM);
    setSaveError('');
    setSaveOk('');
  }

  function startEdit(guest) {
    setEditingId(guest.id);
    setForm(mapGuestToForm(guest));
    setSaveError('');
    setSaveOk('');
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setSaveError('');
    setSaveOk('');

    const payload = toPayload(form);
    if (!payload.first_name || !payload.last_name) {
      setSaveError('Ad ve soyad zorunludur.');
      return;
    }

    setSaving(true);
    try {
      if (editingId) {
        const updated = await updateGuest(editingId, payload, apiKey);
        setSaveOk(`#${updated.id} kaydi guncellendi.`);
      } else {
        const created = await createGuest(payload, apiKey);
        setSaveOk(`#${created.id} kaydi olusturuldu.`);
      }

      await loadGuests(search, offset);
      if (!editingId) {
        resetForm();
      }
    } catch (err) {
      if (!handleApiError(err)) {
        setSaveError(err.message || 'Kayit islemi basarisiz.');
      }
    } finally {
      setSaving(false);
    }
  }

  function prevPage() {
    const next = Math.max(0, offset - PAGE_SIZE);
    setOffset(next);
    loadGuests(search, next);
  }

  function nextPage() {
    if (offset + PAGE_SIZE >= total) return;
    const next = offset + PAGE_SIZE;
    setOffset(next);
    loadGuests(search, next);
  }

  return (
    <>
      <header className="pageHeader">
        <h1>Misafir Yonetimi</h1>
        <p className="subtitle">Listele, ara, yeni kayit ac ve mevcut kayitlari guncelle.</p>
      </header>

      <section className="card" style={{ marginBottom: 20 }}>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap', marginBottom: 12 }}>
          <div className="small" style={{ flex: 1 }}>Kayitli Misafirler ({total})</div>
          <input
            value={search}
            onChange={handleSearchChange}
            placeholder="Isim veya e-posta ile ara..."
            style={{ width: 240 }}
          />
          <button onClick={resetForm}>Yeni Kayit</button>
        </div>

        {listError && <div className="error" style={{ marginBottom: 8 }}>{listError}</div>}
        {listLoading && <div className="textMuted" style={{ padding: 12 }}>Yukleniyor...</div>}

        {!listLoading && guests.length === 0 && (
          <div className="textMuted" style={{ padding: 12 }}>
            {search ? 'Arama sonucu bulunamadi.' : 'Henuz misafir kaydi yok.'}
          </div>
        )}

        {!listLoading && guests.length > 0 && (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th className={thS}>Ad Soyad</th>
                  <th className={thS}>Iletisim</th>
                  <th className={thS}>Otel</th>
                  <th className={thS}>Segment</th>
                  <th className={thS}>Risk</th>
                  <th className={thS}>VIP</th>
                  <th className={thS}>Islem</th>
                </tr>
              </thead>
              <tbody>
                {guests.map((guest) => (
                  <tr key={guest.id} className={editingId === guest.id ? 'guestRowActive' : undefined}>
                    <td className={tdS}>
                      <strong>{guest.first_name} {guest.last_name}</strong>
                    </td>
                    <td className={tdS}>
                      <div>{guest.email || <span className="textMuted">-</span>}</div>
                      <div className="guestSecondary">{guest.phone || ''}</div>
                    </td>
                    <td className={tdS}>{guest.hotel}</td>
                    <td className={tdS}>{guest.market_segment}</td>
                    <td className={tdS}><RiskBadge label={guest.risk_label} score={guest.risk_score} /></td>
                    <td className={tdS} style={{ textAlign: 'center' }}>{guest.vip_status ? 'Yildiz' : '-'}</td>
                    <td className={tdS}>
                      <button
                        className={editingId === guest.id ? 'btnPrimary' : undefined}
                        style={{ padding: '3px 12px', fontSize: 12 }}
                        onClick={() => startEdit(guest)}
                      >
                        {editingId === guest.id ? 'Secili' : 'Duzenle'}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {total > PAGE_SIZE && (
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingTop: 10 }}>
            <button onClick={prevPage} disabled={offset === 0} style={{ padding: '4px 12px' }}>Onceki</button>
            <span className="textMuted">{offset + 1}-{Math.min(offset + PAGE_SIZE, total)} / {total}</span>
            <button onClick={nextPage} disabled={offset + PAGE_SIZE >= total} style={{ padding: '4px 12px' }}>Sonraki</button>
          </div>
        )}
      </section>

      <section className="card">
        <div className="small" style={{ marginBottom: 10 }}>
          {editingId ? `Kayit #${editingId} guncelle` : 'Yeni Misafir Kaydi'}
        </div>

        <form onSubmit={handleSubmit}>
          <div className="chatFormGrid">
            <div>
              <label>Ad *</label>
              <input value={form.first_name} onChange={(e) => setForm((p) => ({ ...p, first_name: e.target.value }))} required />
            </div>
            <div>
              <label>Soyad *</label>
              <input value={form.last_name} onChange={(e) => setForm((p) => ({ ...p, last_name: e.target.value }))} required />
            </div>
            <div>
              <label>E-posta</label>
              <input type="email" value={form.email} onChange={(e) => setForm((p) => ({ ...p, email: e.target.value }))} />
            </div>
            <div>
              <label>Telefon</label>
              <input value={form.phone} onChange={(e) => setForm((p) => ({ ...p, phone: e.target.value }))} />
            </div>
            <div>
              <label>Uyruk</label>
              <input value={form.nationality} onChange={(e) => setForm((p) => ({ ...p, nationality: e.target.value.toUpperCase() }))} maxLength={3} />
            </div>
            <div>
              <label>Kimlik No</label>
              <input value={form.identity_no} onChange={(e) => setForm((p) => ({ ...p, identity_no: e.target.value }))} />
            </div>
            <div>
              <label>Dogum Tarihi</label>
              <input type="date" value={form.birth_date} onChange={(e) => setForm((p) => ({ ...p, birth_date: e.target.value }))} />
            </div>
            <div>
              <label>Cinsiyet</label>
              <select value={form.gender} onChange={(e) => setForm((p) => ({ ...p, gender: e.target.value }))}>
                <option value="">Belirtilmedi</option>
                <option value="M">Erkek</option>
                <option value="F">Kadin</option>
                <option value="other">Diger</option>
              </select>
            </div>
            <div>
              <label>Otel</label>
              <select value={form.hotel} onChange={(e) => setForm((p) => ({ ...p, hotel: e.target.value }))}>
                {HOTELS.map((hotel) => <option key={hotel} value={hotel}>{hotel}</option>)}
              </select>
            </div>
            <div>
              <label>Lead Time</label>
              <input type="number" min="0" value={form.lead_time} onChange={(e) => setForm((p) => ({ ...p, lead_time: e.target.value }))} />
            </div>
            <div>
              <label>Depozito</label>
              <select value={form.deposit_type} onChange={(e) => setForm((p) => ({ ...p, deposit_type: e.target.value }))}>
                {DEPOSIT_TYPES.map((item) => <option key={item} value={item}>{item}</option>)}
              </select>
            </div>
            <div>
              <label>Segment</label>
              <select value={form.market_segment} onChange={(e) => setForm((p) => ({ ...p, market_segment: e.target.value }))}>
                {SEGMENTS.map((item) => <option key={item} value={item}>{item}</option>)}
              </select>
            </div>
            <div>
              <label>Yetiskin</label>
              <input type="number" min="1" value={form.adults} onChange={(e) => setForm((p) => ({ ...p, adults: e.target.value }))} />
            </div>
            <div>
              <label>Cocuk</label>
              <input type="number" min="0" value={form.children} onChange={(e) => setForm((p) => ({ ...p, children: e.target.value }))} />
            </div>
            <div>
              <label>Bebek</label>
              <input type="number" min="0" value={form.babies} onChange={(e) => setForm((p) => ({ ...p, babies: e.target.value }))} />
            </div>
            <div>
              <label>Hafta Ici Gece</label>
              <input type="number" min="0" value={form.stays_in_week_nights} onChange={(e) => setForm((p) => ({ ...p, stays_in_week_nights: e.target.value }))} />
            </div>
            <div>
              <label>Hafta Sonu Gece</label>
              <input type="number" min="0" value={form.stays_in_weekend_nights} onChange={(e) => setForm((p) => ({ ...p, stays_in_weekend_nights: e.target.value }))} />
            </div>
            <div>
              <label>Tekrar Eden</label>
              <select value={form.is_repeated_guest} onChange={(e) => setForm((p) => ({ ...p, is_repeated_guest: e.target.value }))}>
                <option value={0}>Hayir</option>
                <option value={1}>Evet</option>
              </select>
            </div>
            <div>
              <label>Gecmis Iptal</label>
              <input type="number" min="0" value={form.previous_cancellations} onChange={(e) => setForm((p) => ({ ...p, previous_cancellations: e.target.value }))} />
            </div>
            <div>
              <label>ADR</label>
              <input type="number" min="0" step="0.01" value={form.adr} onChange={(e) => setForm((p) => ({ ...p, adr: e.target.value }))} />
            </div>
          </div>

          <div style={{ marginTop: 8 }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <input
                type="checkbox"
                checked={form.vip_status}
                onChange={(e) => setForm((p) => ({ ...p, vip_status: e.target.checked }))}
              />
              VIP Misafir
            </label>
          </div>

          <div style={{ marginTop: 8 }}>
            <label>Notlar</label>
            <textarea
              rows={2}
              style={{ width: '100%', resize: 'vertical', boxSizing: 'border-box' }}
              value={form.notes}
              onChange={(e) => setForm((p) => ({ ...p, notes: e.target.value }))}
            />
          </div>

          <div style={{ marginTop: 12, display: 'flex', gap: 8 }}>
            <button type="submit" disabled={saving}>{saving ? 'Kaydediliyor...' : (editingId ? 'PATCH ile Guncelle' : 'Kayit Olustur')}</button>
            <button type="button" className="btnGhost" onClick={resetForm}>Temizle</button>
          </div>

          {saveOk && <div className="formSuccess">{saveOk}</div>}
          {saveError && <div className="error" style={{ marginTop: 8 }}>{saveError}</div>}
        </form>
      </section>
    </>
  );
}
