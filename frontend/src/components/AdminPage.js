import React, { useState, useEffect } from 'react';
import { getAdminStats } from '../utils/api';

export default function AdminPage() {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getAdminStats().then(data => { setStats(data); setLoading(false); }).catch(() => setLoading(false));
  }, []);

  if (loading) return <div className="admin-page"><p>Loading dashboard...</p></div>;
  if (!stats) return <div className="admin-page"><p>Could not load stats. Is the backend running?</p></div>;

  const triageDist = stats.triage_distribution || {};
  const topConds = stats.top_conditions || [];
  const vdb = stats.vector_db || {};
  const errors = stats.recent_errors || [];
  const totalTriage = Object.values(triageDist).reduce((a, b) => a + b, 0) || 1;

  const triageColors = { Emergency: '#dc2626', Urgent: '#ea580c', 'Same-day': '#ca8a04', Routine: '#16a34a' };

  return (
    <div className="admin-page">
      <h2 style={{ fontSize: '1.2rem', marginBottom: 6 }}>📊 Admin Dashboard</h2>
      <p className="subtitle" style={{ color: '#6b7280', marginBottom: 20, fontSize: '0.86rem' }}>
        Real-time analytics from SQLite + ChromaDB
      </p>

      <div className="metrics-grid">
        <div className="metric-card"><div className="metric-val">{stats.total_sessions || 0}</div><div className="metric-label">Sessions</div></div>
        <div className="metric-card"><div className="metric-val">{stats.total_messages || 0}</div><div className="metric-label">Messages</div></div>
        <div className="metric-card"><div className="metric-val">{stats.total_assessments || 0}</div><div className="metric-label">Assessments</div></div>
        <div className="metric-card"><div className="metric-val">{stats.total_errors || 0}</div><div className="metric-label">Errors</div></div>
      </div>

      <div className="admin-grid">
        <div className="admin-card">
          <h3>📊 Triage Distribution</h3>
          {Object.entries(triageDist).map(([level, count]) => (
            <div key={level} style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '7px 0', fontSize: '0.84rem' }}>
              <div style={{ width: 10, height: 10, borderRadius: '50%', background: triageColors[level] || '#999', flexShrink: 0 }} />
              <div style={{ flex: 1 }}>{level}</div>
              <div style={{ flex: 2, height: 8, background: '#f0f0f0', borderRadius: 4, overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${(count / totalTriage) * 100}%`, background: triageColors[level], borderRadius: 4 }} />
              </div>
              <div style={{ fontWeight: 700, width: 32, textAlign: 'right', fontSize: '0.8rem' }}>{count}</div>
            </div>
          ))}
          {Object.keys(triageDist).length === 0 && <p style={{ color: '#999', fontSize: '0.82rem' }}>No assessments yet</p>}
        </div>

        <div className="admin-card">
          <h3>🏆 Top Conditions</h3>
          {topConds.map((c, i) => (
            <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 0', borderBottom: '1px solid #f5f5f5', fontSize: '0.84rem' }}>
              <span>{c.condition}</span>
              <span style={{ fontWeight: 700, color: '#2563eb' }}>{c.count}</span>
            </div>
          ))}
          {topConds.length === 0 && <p style={{ color: '#999', fontSize: '0.82rem' }}>No data yet</p>}
        </div>

        <div className="admin-card">
          <h3>🗄️ Vector Database</h3>
          <div style={{ display: 'flex', gap: 10, marginBottom: 12 }}>
            {[
              { val: vdb.total_documents || 0, lab: 'Documents' },
              { val: vdb.categories?.conditions || 0, lab: 'Conditions' },
              { val: vdb.categories?.treatments || 0, lab: 'Treatments' },
            ].map((s, i) => (
              <div key={i} style={{ flex: 1, background: '#f8fafc', borderRadius: 8, padding: 12, textAlign: 'center', border: '1px solid #e5e7eb' }}>
                <div style={{ fontSize: '1.2rem', fontWeight: 800, color: '#2563eb' }}>{s.val}</div>
                <div style={{ fontSize: '0.68rem', color: '#999' }}>{s.lab}</div>
              </div>
            ))}
          </div>
          <div style={{ fontSize: '0.76rem', color: '#888' }}>
            <strong>Model:</strong> all-MiniLM-L6-v2 (384-dim)
          </div>
        </div>

        <div className="admin-card">
          <h3>⚠️ Recent Errors</h3>
          {errors.length === 0 ? (
            <p style={{ color: '#16a34a', fontSize: '0.84rem' }}>✅ No errors logged</p>
          ) : errors.map((e, i) => (
            <div key={i} style={{ padding: '6px 0', borderBottom: '1px solid #f5f5f5', fontSize: '0.8rem' }}>
              <span style={{ color: '#999', fontSize: '0.7rem' }}>{e.time}</span>{' '}
              <strong>{e.type}</strong>: {e.message}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
