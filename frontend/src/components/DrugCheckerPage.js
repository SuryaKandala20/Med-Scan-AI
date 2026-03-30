import React, { useState } from 'react';

export default function DrugCheckerPage() {
  const [drugs, setDrugs] = useState(['', '']);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const addDrug = () => setDrugs(prev => [...prev, '']);
  const removeDrug = (i) => setDrugs(prev => prev.filter((_, idx) => idx !== i));
  const updateDrug = (i, val) => setDrugs(prev => prev.map((d, idx) => idx === i ? val : d));

  const checkInteractions = async () => {
    const validDrugs = drugs.filter(d => d.trim());
    if (validDrugs.length < 2) return;
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const res = await fetch('http://localhost:8000/api/drugs/check', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ drugs: validDrugs }),
      });
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail); }
      setResult(await res.json());
    } catch (err) { setError(err.message); }
    setLoading(false);
  };

  const severityStyle = (sev) => {
    const map = {
      Major: { bg: '#fef2f2', color: '#b91c1c', border: '#fca5a5', icon: '🔴' },
      Moderate: { bg: '#fff7ed', color: '#c2410c', border: '#fed7aa', icon: '🟠' },
      Minor: { bg: '#fef9c3', color: '#a16207', border: '#fde68a', icon: '🟡' },
      None: { bg: '#dcfce7', color: '#15803d', border: '#bbf7d0', icon: '🟢' },
    };
    return map[sev] || map.None;
  };

  const riskStyle = (risk) => {
    const map = {
      High: { bg: '#fef2f2', color: '#b91c1c' },
      Moderate: { bg: '#fff7ed', color: '#c2410c' },
      Low: { bg: '#fef9c3', color: '#a16207' },
      Safe: { bg: '#dcfce7', color: '#15803d' },
    };
    return map[risk] || map.Low;
  };

  return (
    <div style={{ flex: 1, overflowY: 'auto', padding: 30 }}>
      <h2 style={{ fontSize: '1.2rem', marginBottom: 6 }}>💊 Drug Interaction Checker</h2>
      <p style={{ color: '#6b7280', fontSize: '0.86rem', marginBottom: 20 }}>
        Enter 2 or more medications to check for dangerous interactions
      </p>

      <div style={{ maxWidth: 600 }}>
        {drugs.map((drug, i) => (
          <div key={i} style={{ display: 'flex', gap: 8, marginBottom: 8, alignItems: 'center' }}>
            <span style={{ color: '#999', fontSize: '0.8rem', width: 60 }}>Drug {i + 1}</span>
            <input
              type="text"
              value={drug}
              onChange={(e) => updateDrug(i, e.target.value)}
              placeholder="e.g., Aspirin, Ibuprofen, Metformin..."
              style={{
                flex: 1, padding: '10px 14px', border: '1.5px solid #e5e7eb',
                borderRadius: 10, fontSize: '0.88rem', fontFamily: 'inherit',
                outline: 'none',
              }}
            />
            {drugs.length > 2 && (
              <button onClick={() => removeDrug(i)} style={{
                background: 'none', border: 'none', color: '#999', cursor: 'pointer', fontSize: '1.1rem',
              }}>✕</button>
            )}
          </div>
        ))}

        <div style={{ display: 'flex', gap: 10, marginTop: 12 }}>
          <button onClick={addDrug} style={{
            padding: '8px 16px', borderRadius: 8, border: '1.5px dashed #d1d5db',
            background: 'none', cursor: 'pointer', fontSize: '0.82rem', fontFamily: 'inherit', color: '#6b7280',
          }}>+ Add another drug</button>
          <button onClick={checkInteractions} disabled={loading || drugs.filter(d => d.trim()).length < 2} style={{
            padding: '10px 22px', borderRadius: 10, border: 'none',
            background: '#2563eb', color: '#fff', fontSize: '0.86rem',
            fontWeight: 600, cursor: 'pointer', fontFamily: 'inherit',
            opacity: (loading || drugs.filter(d => d.trim()).length < 2) ? 0.5 : 1,
          }}>
            {loading ? '⏳ Checking...' : '🔍 Check Interactions'}
          </button>
        </div>
      </div>

      {error && <div style={{ color: '#dc2626', marginTop: 16, fontSize: '0.88rem' }}>⚠️ {error}</div>}

      {result && (
        <div style={{ maxWidth: 700, marginTop: 24, animation: 'fadeUp 0.3s ease-out' }}>
          {/* Overall risk */}
          {result.overall_risk && (
            <div style={{
              padding: 16, borderRadius: 12, marginBottom: 16, fontSize: '0.9rem', lineHeight: 1.6,
              background: riskStyle(result.overall_risk).bg, color: riskStyle(result.overall_risk).color,
            }}>
              <strong>Overall Risk: {result.overall_risk}</strong>
              {result.overall_summary && <> — {result.overall_summary}</>}
            </div>
          )}

          {/* Interactions */}
          {result.interactions?.map((inter, i) => {
            const s = severityStyle(inter.severity);
            return (
              <div key={i} style={{
                background: '#fff', borderRadius: 12, padding: 16, marginBottom: 10,
                boxShadow: '0 1px 3px rgba(0,0,0,0.05)',
                borderLeft: `4px solid ${s.color}`,
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                  <span>{s.icon}</span>
                  <strong style={{ fontSize: '0.9rem' }}>{inter.drug_pair}</strong>
                  <span style={{
                    marginLeft: 'auto', padding: '3px 10px', borderRadius: 12,
                    fontSize: '0.68rem', fontWeight: 700, background: s.bg, color: s.color,
                    border: `1px solid ${s.border}`,
                  }}>{inter.severity}</span>
                </div>
                <p style={{ fontSize: '0.84rem', color: '#444', lineHeight: 1.6 }}>{inter.description}</p>
                {inter.mechanism && (
                  <p style={{ fontSize: '0.78rem', color: '#888', marginTop: 6 }}>
                    <strong>Mechanism:</strong> {inter.mechanism}
                  </p>
                )}
                {inter.recommendation && (
                  <p style={{ fontSize: '0.82rem', color: '#9a3412', marginTop: 6, fontWeight: 500 }}>
                    💡 {inter.recommendation}
                  </p>
                )}
              </div>
            );
          })}

          {/* Warnings */}
          {result.warnings?.length > 0 && (
            <div style={{
              background: '#fef2f2', border: '1px solid #fca5a5', borderRadius: 10,
              padding: 14, marginTop: 14, fontSize: '0.84rem', color: '#991b1b',
            }}>
              <strong>⚠️ Warnings:</strong>
              {result.warnings.map((w, i) => <div key={i} style={{ marginTop: 4 }}>• {w}</div>)}
            </div>
          )}

          {/* Advice */}
          {result.advice && (
            <div style={{
              background: '#fff7ed', border: '1px solid #fed7aa', borderRadius: 10,
              padding: '13px 16px', marginTop: 14, fontSize: '0.82rem', color: '#9a3412', lineHeight: 1.6,
            }}>
              👨‍⚕️ {result.advice}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
