import { useState, useCallback, useEffect, useRef } from 'react';
import { startChatSession, getChatSummary, predictRiskScore, createGuest, streamChatMessage } from '../api';

/**
 * useChat — Chat asistanı hook'u
 *
 * Müşteri formu, oturum yönetimi ve mesajlaşma state'i.
 * AbortController ile oturum açma cancel edilebilir.
 * Form değişince debounced auto-predict çalışır (600 ms).
 *
 * @param {object}   opts
 * @param {string}   opts.apiKey          - X-API-KEY header değeri
 * @param {function} opts.onAuthFailed    - 401 alındığında çağrılır
 * @param {object}   [opts.initialCustomer={}] - Mevcut misafir verisiyle formu ön doldurur
 *
 * @returns {{
 *   sessionId:           string,
 *   messages:            Array<{role: string, content: string}>,
 *   input:               string,
 *   quickActions:        string[],
 *   summary:             object|null,
 *   busy:                boolean,
 *   error:               string,
 *   riskScore:           number|null,
 *   riskLabel:           string,
 *   predicting:          boolean,
 *   selectedModel:       string|null,
 *   guestId:             number|null,
 *   guestSaved:          boolean,
 *   customer:            object,
 *   setInput:            (v: string) => void,
 *   setSelectedModel:    (m: string|null) => void,
 *   handleCustomerChange:(key: string, value: any) => void,
 *   handleStartSession:  () => Promise<void>,
 *   handleSend:          () => Promise<void>,
 *   handleSaveGuest:     () => Promise<void>,
 * }}
 */
export function useChat({ apiKey, onAuthFailed, initialCustomer = {} }) {
  const [sessionId, setSessionId]         = useState('');
  const [messages, setMessages]           = useState([]);
  const [input, setInput]                 = useState('');
  const [quickActions, setQuickActions]   = useState([]);
  const [summary, setSummary]             = useState(null);
  const [busy, setBusy]                   = useState(false);
  const [error, setError]                 = useState('');
  const [riskScore, setRiskScore]         = useState(null);
  const [riskLabel, setRiskLabel]         = useState('unknown');
  const [predicting, setPredicting]       = useState(false);
  const [selectedModel, setSelectedModel] = useState(null); // null → varsayılan şampiyon
  const [guestId, setGuestId]             = useState(initialCustomer?.id ?? null);
  const [guestSaved, setGuestSaved]       = useState(!!initialCustomer?.id);
  const [customer, setCustomer]           = useState({
    // Kişisel bilgiler (DB'ye kaydedilir, modele gitmez)
    first_name:  '',
    last_name:   '',
    email:       '',
    phone:       '',
    nationality: '',
    identity_no: '',
    birth_date:  '',
    gender:      '',
    vip_status:  false,
    notes:       '',
    // Rezervasyon / model alanları
    hotel:                   'City Hotel',
    lead_time:               30,
    deposit_type:            'No Deposit',
    previous_cancellations:  0,
    market_segment:          'Online TA',
    adults:                  2,
    children:                0,
    stays_in_week_nights:    2,
    stays_in_weekend_nights: 1,
    is_repeated_guest:       0,
    ...initialCustomer,
  });

  const abortRef       = useRef(null);
  const predictAbort   = useRef(null);
  const debounceTimer  = useRef(null);
  const sessionRef     = useRef(sessionId);
  sessionRef.current   = sessionId;

  function handleCustomerChange(key, value) {
    setCustomer(prev => ({ ...prev, [key]: value }));
  }

  // Yalnızca rezervasyon alanları değişince risk yeniden hesaplanır (kişisel alanlar tetiklemez)
  const bookingSnapshot = JSON.stringify({
    hotel: customer.hotel, lead_time: customer.lead_time,
    deposit_type: customer.deposit_type, market_segment: customer.market_segment,
    adults: customer.adults, children: customer.children,
    stays_in_week_nights: customer.stays_in_week_nights,
    stays_in_weekend_nights: customer.stays_in_weekend_nights,
    is_repeated_guest: customer.is_repeated_guest,
    previous_cancellations: customer.previous_cancellations,
    selectedModel,
  });

  // Form değişince 600 ms debounce ile otomatik risk hesapla
  useEffect(() => {
    clearTimeout(debounceTimer.current);
    predictAbort.current?.abort();

    debounceTimer.current = setTimeout(async () => {
      const ctrl = new AbortController();
      predictAbort.current = ctrl;
      setPredicting(true);
      try {
        const result = await predictRiskScore(
          {
            hotel:                   customer.hotel,
            lead_time:               Number(customer.lead_time || 0),
            deposit_type:            customer.deposit_type,
            market_segment:          customer.market_segment,
            adults:                  Number(customer.adults || 1),
            children:                Number(customer.children || 0),
            babies:                  Number(customer.babies || 0),
            stays_in_week_nights:    Number(customer.stays_in_week_nights || 0),
            stays_in_weekend_nights: Number(customer.stays_in_weekend_nights || 0),
            previous_cancellations:  Number(customer.previous_cancellations || 0),
            is_repeated_guest:       Number(customer.is_repeated_guest || 0),
            adr:                     Number(customer.adr || 0),
          },
          apiKey,
          { signal: ctrl.signal, modelName: selectedModel || undefined },
        );
        setRiskScore(result.risk_score);
        setRiskLabel(result.risk_label);
      } catch (e) {
        if (e.name !== 'AbortError') {
          setRiskScore(null);
          setRiskLabel('unknown');
        }
      } finally {
        setPredicting(false);
      }
    }, 600);

    return () => {
      clearTimeout(debounceTimer.current);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bookingSnapshot, apiKey]);

  const openSession = useCallback(async () => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setError('');
    setBusy(true);
    try {
      // Misafir adı girilmişse ve henüz kaydedilmemişse DB'ye kaydet
      if (!guestId && customer.first_name) {
        try {
          const guest = await createGuest({
            first_name:              customer.first_name,
            last_name:               customer.last_name   || '-',
            email:                   customer.email       || null,
            phone:                   customer.phone       || null,
            nationality:             customer.nationality || null,
            identity_no:             customer.identity_no || null,
            birth_date:              customer.birth_date  || null,
            gender:                  customer.gender      || null,
            vip_status:              !!customer.vip_status,
            notes:                   customer.notes       || null,
            hotel:                   customer.hotel,
            lead_time:               Number(customer.lead_time || 0),
            deposit_type:            customer.deposit_type,
            market_segment:          customer.market_segment,
            adults:                  Number(customer.adults || 1),
            children:                Number(customer.children || 0),
            babies:                  0,
            stays_in_week_nights:    Number(customer.stays_in_week_nights || 0),
            stays_in_weekend_nights: Number(customer.stays_in_weekend_nights || 0),
            is_repeated_guest:       Number(customer.is_repeated_guest || 0),
            previous_cancellations:  Number(customer.previous_cancellations || 0),
          }, apiKey);
          setGuestId(guest.id);
          setGuestSaved(true);
        } catch (guestErr) {
          console.warn('Misafir kaydı oluşturulamadı:', guestErr);
        }
      }

      // Mevcut hesaplanmış risk skoru kullan (auto-predict ile geldi)
      // Eğer henüz hesaplanmadıysa bekle / fallback kullan
      const computedScore = riskScore ?? 0.5;
      const computedLabel = riskLabel !== 'unknown' ? riskLabel : 'medium';

      const payload = {
        customer_data: {
          ...customer,
          lead_time:               Number(customer.lead_time || 0),
          previous_cancellations:  Number(customer.previous_cancellations || 0),
          adults:                  Number(customer.adults || 1),
          children:                Number(customer.children || 0),
          stays_in_week_nights:    Number(customer.stays_in_week_nights || 0),
          stays_in_weekend_nights: Number(customer.stays_in_weekend_nights || 0),
          is_repeated_guest:       Number(customer.is_repeated_guest || 0),
        },
        risk_score: computedScore,
        risk_label: computedLabel,
      };
      const created = await startChatSession(payload, apiKey, { signal: controller.signal });
      setSessionId(created.session_id);
      setQuickActions(created.quick_actions || []);
      setMessages([{ role: 'assistant', content: created.bot_message || 'Oturum açıldı.' }]);

      const s = await getChatSummary(created.session_id, apiKey, { signal: controller.signal });
      setSummary(s);
    } catch (err) {
      if (err.name === 'AbortError') return;
      if (err?.status === 401) { onAuthFailed?.(err); return; }
      setError(err.message || 'Chat oturumu açılamadı.');
    } finally {
      setBusy(false);
    }
  }, [customer, riskScore, riskLabel, apiKey, onAuthFailed, guestId]);

  const sendMessage = useCallback(async (text) => {
    const messageText = String(text || '').trim();
    if (!messageText || !sessionRef.current) return;

    setError('');
    setBusy(true);
    setMessages(prev => [...prev, { role: 'user', content: messageText }]);
    setInput('');

    // Add a streaming placeholder for the assistant response
    const streamId = `stream-${Date.now()}`;
    setMessages(prev => [...prev, { role: 'assistant', content: '', streaming: true, id: streamId }]);

    try {
      const reader = await streamChatMessage(
        { session_id: sessionRef.current, message: messageText },
        apiKey,
      );
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          try {
            const data = JSON.parse(line.slice(6));
            if (data.token) {
              setMessages(prev => prev.map(m =>
                m.id === streamId ? { ...m, content: m.content + data.token } : m
              ));
            }
            if (data.done) {
              setMessages(prev => prev.map(m =>
                m.id === streamId ? { ...m, streaming: false } : m
              ));
              setQuickActions(data.quick_actions || []);
            }
            if (data.error) throw new Error(data.error);
          } catch (parseErr) {
            if (parseErr.message && !parseErr.message.startsWith('JSON')) throw parseErr;
          }
        }
      }
      // Ensure streaming flag is cleared even if 'done' event was missed
      setMessages(prev => prev.map(m =>
        m.id === streamId ? { ...m, streaming: false } : m
      ));

      const s = await getChatSummary(sessionRef.current, apiKey);
      setSummary(s);
    } catch (err) {
      if (err?.status === 401) { onAuthFailed?.(err); return; }
      setError(err.message || 'Mesaj gönderilemedi.');
      setMessages(prev => prev.filter(m => m.id !== streamId));
    } finally {
      setBusy(false);
    }
  }, [apiKey, onAuthFailed]);

  useEffect(() => () => abortRef.current?.abort(), []);

  return {
    sessionId, messages, input, setInput,
    quickActions, summary, busy, error,
    riskScore, riskLabel,
    predicting,
    guestId, setGuestId, guestSaved, setGuestSaved,
    customer, handleCustomerChange,
    selectedModel, setSelectedModel,
    openSession, sendMessage,
  };
}
