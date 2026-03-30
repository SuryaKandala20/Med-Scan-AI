import React, { useState, useEffect } from 'react';
import ChatPage from './components/ChatPage';
import ReportPage from './components/ReportPage';
import ReportComparePage from './components/ReportComparePage';
import DrugCheckerPage from './components/DrugCheckerPage';
import SkinPage from './components/SkinPage';
import AdminPage from './components/AdminPage';
import { getHealth } from './utils/api';

const MODES = [
  { id: 'chat', icon: '💬', label: 'Chat Assistant', badge: 'GPT-4o' },
  { id: 'skin', icon: '🔬', label: 'Skin Analysis', badge: 'CNN' },
  { id: 'report', icon: '📄', label: 'Report Explainer', badge: 'PDF' },
  { id: 'compare', icon: '📊', label: 'Report Comparison', badge: 'NEW' },
  { id: 'drugs', icon: '💊', label: 'Drug Interactions', badge: 'NEW' },
  { id: 'admin', icon: '⚙️', label: 'Admin Dashboard', badge: null },
];

const TOPBAR_INFO = {
  chat: { title: 'Chat Assistant', sub: 'GPT-4o + ChromaDB RAG • Streaming • Multi-language' },
  skin: { title: 'Skin Analysis', sub: 'EfficientNet-B0 • HAM10000 Dataset' },
  report: { title: 'Report Explainer', sub: 'GPT-4o Structured Analysis • PDF Upload' },
  compare: { title: 'Report Comparison', sub: 'Upload 2 reports • AI highlights changes' },
  drugs: { title: 'Drug Interaction Checker', sub: 'GPT-4o Pharmacology Analysis' },
  admin: { title: 'Admin Dashboard', sub: 'SQLite Audit + ChromaDB Stats' },
};

export default function App() {
  const [mode, setMode] = useState('chat');
  const [health, setHealth] = useState(null);

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth({ status: 'offline' }));
  }, []);

  const info = TOPBAR_INFO[mode];

  return (
    <div className="app-layout">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-logo">
          <h1><span className="logo-icon">🏥</span> MedScan AI</h1>
          <p>AI-Powered Medical Assistant</p>
        </div>

        <nav className="sidebar-nav">
          <div className="nav-label">Analysis Modes</div>
          {MODES.map(m => (
            <button
              key={m.id}
              className={`nav-btn ${mode === m.id ? 'active' : ''}`}
              onClick={() => setMode(m.id)}
            >
              <span className="nav-icon">{m.icon}</span>
              {m.label}
              {m.badge && <span className="nav-badge">{m.badge}</span>}
            </button>
          ))}

          <div className="nav-label">System</div>
          <div className="nav-btn" style={{ cursor: 'default' }}>
            <span className="nav-icon">🗄️</span>
            Vector DB
            <span className="nav-badge">ChromaDB</span>
          </div>
          <div className="nav-btn" style={{ cursor: 'default' }}>
            <span className="nav-icon">💾</span>
            Audit Logs
            <span className="nav-badge">SQLite</span>
          </div>
          <div className="nav-btn" style={{ cursor: 'default' }}>
            <span className="nav-icon">{health?.openai_configured ? '🟢' : '🔴'}</span>
            OpenAI
            <span className="nav-badge">{health?.model || '...'}</span>
          </div>
        </nav>

        <div className="sidebar-bottom">
          <div className="warn-box">
            ⚠️ Educational tool only — never a substitute for professional medical advice.
          </div>
        </div>
      </aside>

      {/* Main */}
      <main className="main-area">
        <div className="topbar">
          <div className="pulse-dot" />
          <div className="topbar-info">
            <div className="topbar-title">{info.title}</div>
            <div className="topbar-sub">{info.sub}</div>
          </div>
        </div>

        <div className="disclaimer">
          ⚠️ <strong>Educational Tool Only</strong> — NOT a medical device. Always consult a healthcare professional.
        </div>

        {mode === 'chat' && <ChatPage />}
        {mode === 'skin' && <SkinPage />}
        {mode === 'report' && <ReportPage />}
        {mode === 'compare' && <ReportComparePage />}
        {mode === 'drugs' && <DrugCheckerPage />}
        {mode === 'admin' && <AdminPage />}
      </main>
    </div>
  );
}
