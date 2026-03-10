import { useState, useCallback, useEffect } from 'react';
import { login, logout, me } from '../api';

/**
 * useAuth — Kimlik doğrulama hook'u
 *
 * JWT token'ı localStorage'dan okur ve /auth/me ile doğrular.
 * Login / logout / auth-failure aksiyonlarını yönetir.
 *
 * @returns {{
 *   authenticated: boolean,
 *   currentUser:   string,
 *   loginError:    string,
 *   checking:      boolean,
 *   handleLogin:   (username: string, password: string) => Promise<boolean>,
 *   handleLogout:  () => Promise<void>,
 *   handleAuthFailure: () => void,
 * }}
 */
export function useAuth() {
  const [authenticated, setAuthenticated] = useState(false);
  const [currentUser, setCurrentUser]     = useState('');
  const [loginError, setLoginError]       = useState('');
  const [checking, setChecking]           = useState(true);

  // İlk yüklemede mevcut token'ı doğrula
  useEffect(() => {
    const token = localStorage.getItem('dashboard_token');
    if (!token) {
      setAuthenticated(false);
      setChecking(false);
      return;
    }
    me()
      .then((p) => {
        setAuthenticated(true);
        setCurrentUser(p.username || '');
      })
      .catch(() => {
        localStorage.removeItem('dashboard_token');
        setAuthenticated(false);
      })
      .finally(() => setChecking(false));
  }, []);

  const handleLogin = useCallback(async (username, password) => {
    setLoginError('');
    try {
      const p = await login(username, password);
      localStorage.setItem('dashboard_token', p.access_token);
      setAuthenticated(true);
      setCurrentUser(p.username || username);
      return true;
    } catch (err) {
      setLoginError(err.message || 'Giriş yapılamadı.');
      return false;
    }
  }, []);

  const handleLogout = useCallback(async () => {
    try { await logout(); } catch { /* ignore */ }
    localStorage.removeItem('dashboard_token');
    setAuthenticated(false);
    setCurrentUser('');
  }, []);

  const handleAuthFailure = useCallback(() => {
    localStorage.removeItem('dashboard_token');
    setAuthenticated(false);
    setLoginError('Oturum süresi doldu. Lütfen tekrar giriş yapın.');
  }, []);

  return {
    authenticated,
    currentUser,
    loginError,
    checking,
    handleLogin,
    handleLogout,
    handleAuthFailure,
  };
}
