import { Routes, Route, Navigate } from 'react-router-dom';
import { useAuth } from './hooks/useAuth';
import { useTheme } from './hooks/useTheme';
import LoginPage from './components/LoginPage';
import Layout from './components/Layout';
import OverviewPage from './components/OverviewPage';
import ModelsPage from './components/ModelsPage';
import PipelinePage from './components/PipelinePage';
import RunsPage from './components/RunsPage';
import ChatPage from './components/ChatPage';
import GuestPage from './components/GuestPage';
import SystemPage from './components/SystemPage';
import ErrorBoundary from './components/ErrorBoundary';

function withRouteBoundary(element) {
  return <ErrorBoundary>{element}</ErrorBoundary>;
}

/**
 * App — Kök bileşen
 *
 * HashRouter main.jsx'te sarılır.
 * Kimlik doğrulama yapılana kadar LoginPage,
 * sonrasında Layout + sayfa rotaları gösterilir.
 */
export default function App() {
  const auth  = useAuth();
  const theme = useTheme();

  /* İlk token kontrolü tamamlanana kadar splash */
  if (auth.checking) {
    return (
      <div className="loginWrap" style={{ justifyContent: 'center', alignItems: 'center', display: 'flex' }}>
        <div className="card" style={{ textAlign: 'center', padding: 40 }}>
          <div style={{ fontSize: 36, marginBottom: 12 }}>⏳</div>
          <div>Oturum kontrol ediliyor…</div>
        </div>
      </div>
    );
  }

  /* Giriş yapılmamış */
  if (!auth.authenticated) {
    return <LoginPage auth={auth} theme={theme} />;
  }

  /* Ana uygulama — rotalı yapı */
  return (
    <Routes>
      <Route element={<Layout auth={auth} theme={theme} />}>
        <Route index element={withRouteBoundary(<OverviewPage />)} />
        <Route path="models"   element={withRouteBoundary(<ModelsPage />)} />
        <Route path="pipeline" element={withRouteBoundary(<PipelinePage />)} />
        <Route path="runs"     element={withRouteBoundary(<RunsPage />)} />
        <Route path="chat"     element={withRouteBoundary(<ChatPage />)} />
        <Route path="guests"   element={withRouteBoundary(<GuestPage />)} />
        <Route path="system"   element={withRouteBoundary(<SystemPage />)} />
        <Route path="*"        element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
