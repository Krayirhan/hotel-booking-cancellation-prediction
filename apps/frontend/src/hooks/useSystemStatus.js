import { useState, useCallback, useEffect, useRef } from 'react';
import { getSystemStatus } from '../api';

/**
 * useSystemStatus — Tüm backend servislerinin sağlık durumunu çeker.
 *
 * GET /dashboard/api/system → { overall, generated_at, services: { database, redis, ollama, model } }
 *
 * AbortController ile sayfa değişiminde inflight istek otomatik olarak iptal edilir.
 *
 * @param {object}   [opts={}]
 * @param {string}   opts.apiKey        - API anahtarı (VITE_DEFAULT_API_KEY varsayılan)
 * @param {function} opts.onAuthFailed  - 401 durumunda çağrılır
 *
 * @returns {{
 *   status:  {overall: string, generated_at: string, services: object}|null,
 *   loading: boolean,
 *   error:   string,
 *   refresh: () => Promise<void>,
 * }}
 */
export function useSystemStatus({ apiKey, onAuthFailed } = {}) {
  const [status, setStatus]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState('');
  const abortRef = useRef(null);

  const refresh = useCallback(async () => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setLoading(true);
    setError('');
    try {
      const data = await getSystemStatus(apiKey, { signal: controller.signal });
      setStatus(data);
    } catch (err) {
      if (err.name === 'AbortError') return;
      if (err?.status === 401 || String(err?.message || '').includes('401')) {
        onAuthFailed?.(err);
        return;
      }
      setError(err.message || 'Sistem durumu alınamadı');
    } finally {
      setLoading(false);
    }
  }, [apiKey, onAuthFailed]);

  // Unmount'ta inflight isteği iptal et
  useEffect(() => () => abortRef.current?.abort(), []);

  return { status, loading, error, refresh };
}
