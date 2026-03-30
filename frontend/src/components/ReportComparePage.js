import React, { useState, useRef } from 'react';

export default function ReportComparePage() {
  const [text1, setText1] = useState('');
  const [text2, setText2] = useState('');
  const [file1, setFile1] = useState(null);
  const [file2, setFile2] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const fileRef1 = useRef(null);
  const fileRef2 = useRef(null);

  const handleCompare = async () => {
    if ((!text1.trim() && !file1) || (!text2.trim() && !file2)) return;
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const form = new FormData();
      form.append('text1', text1);
      form.append('text2', text2);
      if (file1) form.append('file1', file1);
      if (file2) form.append('file2', file2);
      const res = await fetch('http://localhost:8000/api/report/compare', { method: 'POST', body: form });
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail); }
      setResult(await res.json());
    } catch (err) { setError(err.message); }
    setLoading(false);
  };

  const dirColor = (dir) => {
    if (dir === 'Improved') return { bg: '#dcfce7', color: '#15803d', icon: '📈' };
    if (dir === 'Worsened') return { bg: '#fef2f2', color: '#b91c1c', icon: '📉' };
    if (dir === 'New Finding') return { bg: '#eff4ff', color: '#2563eb', icon: '🆕' };
    return { bg: '#f3f4f6', color: '#6b7280', icon: '➡️' };
  };

  const trendStyle = (trend) => {
    const map = {
      Improved: { bg: '#dcfce7', color: '#15803d' },
      Worsened: { bg: '#fef2f2', color: '#b91c1c' },
      Mixed: { bg: '#fff7ed', color: '#c2410c' },
      Stable: { bg: '#f3f4f6', color: '#6b7280' },
    };
    return map[trend] || map.Stable;
  };

  const ReportInput = ({ label, text, setText, file, setFile, fileRef }) => (
    <div style={{ flex: 1 }}>
      <h3 style={{ fontSize: '0.9rem', fontWeight: 600, marginBottom: 8 }}>{label}</h3>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Paste report text here..."
        style={{
          width: '100%', height: 130, padding: '12px 14px', border: '1.5px solid #e5e7eb',
          borderRadius: 10, fontFamily: 'inherit', fontSize: '0.84rem', resize: 'vertical',
          outline: 'none', lineHeight: 1.5,
        }}
      />
      <div style={{ marginTop: 6 }}>
        <input type="file" accept=".pdf,.txt" ref={fileRef} style={{ display: 'none' }}
          onChange={(e) => setFile(e.target.files[0])} />
        <button onClick={() => fileRef.current?.click()} style={{
          padding: '6px 14px', borderRadius: 8, border: '1.5px dashed #d1d5db',
          background: 'none', cursor: 'pointer', fontSize: '0.78rem', fontFamily: 'inherit', color: '#6b7280',
        }}>
          📎 {file ? file.name : 'Upload PDF'}
        </button>
        {file && <button onClick={() => { setFile(null); fileRef.current.value = ''; }}
          style={{ background: 'none', border: 'none', color: '#999', cursor: 'pointer', fontSize: '0.78rem', marginLeft: 6 }}>✕</button>}
      </div>
    </div>
  );

  return (
    <div style={{ flex: 1, overflowY: 'auto', padding: 30 }}>
      <h2 style={{ fontSize: '1.2rem', marginBottom: 6 }}>📊 Report Comparison</h2>
      <p style={{ color: '#6b7280', fontSize: '0.86rem', marginBottom: 20 }}>
        Upload two reports from different dates — AI highlights what changed, improved, or worsened
      </p>

      <div style={{ display: 'flex', gap: 20, maxWidth: 900 }}>
        <ReportInput label="📋 Report 1 (Older)" text={text1} setText={setText1} file={file1} setFile={setFile1} fileRef={fileRef1} />
        <ReportInput label="📋 Report 2 (Newer)" text={text2} setText={setText2} file={file2} setFile={setFile2} fileRef={fileRef2} />
      </div>

      <button onClick={handleCompare} disabled={loading || ((!text1.trim() && !file1) || (!text2.trim() && !file2))} style={{
        marginTop: 16, padding: '10px 24px', borderRadius: 10, border: 'none',
        background: '#2563eb', color: '#fff', fontSize: '0.86rem',
        fontWeight: 600, cursor: 'pointer', fontFamily: 'inherit',
        opacity: loading ? 0.5 : 1,
      }}>
        {loading ? '⏳ Comparing...' : '🔍 Compare Reports'}
      </button>

      {error && <div style={{ color: '#dc2626', marginTop: 16, fontSize: '0.88rem' }}>⚠️ {error}</div>}

      {result && (
        <div style={{ maxWidth: 800, marginTop: 24, animation: 'fadeUp 0.3s ease-out' }}>
          {/* Summary */}
          {result.summary && (
            <div style={{ padding: 16, background: '#eff4ff', borderRadius: 12, fontSize: '0.9rem', lineHeight: 1.7, marginBottom: 16 }}>
              <strong>📋 Summary:</strong> {result.summary}
            </div>
          )}

          {/* Overall trend */}
          {result.overall_trend && (
            <div style={{
              padding: 14, borderRadius: 10, marginBottom: 16, fontSize: '0.88rem',
              display: 'inline-block',
              background: trendStyle(result.overall_trend).bg,
              color: trendStyle(result.overall_trend).color,
            }}>
              <strong>Overall Trend: {result.overall_trend}</strong>
            </div>
          )}

          {/* Changes */}
          {result.changes?.length > 0 && (
            <>
              <h3 style={{ fontSize: '0.95rem', margin: '16px 0 10px' }}>🔄 Changes Detected</h3>
              {result.changes.map((c, i) => {
                const d = dirColor(c.direction);
                return (
                  <div key={i} style={{
                    background: '#fff', borderRadius: 12, padding: 14, marginBottom: 8,
                    boxShadow: '0 1px 3px rgba(0,0,0,0.05)',
                    borderLeft: `4px solid ${d.color}`,
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                      <span>{d.icon}</span>
                      <strong style={{ fontSize: '0.88rem' }}>{c.parameter}</strong>
                      <span style={{
                        marginLeft: 'auto', padding: '2px 10px', borderRadius: 10,
                        fontSize: '0.65rem', fontWeight: 700, background: d.bg, color: d.color,
                      }}>{c.direction}</span>
                    </div>
                    <div style={{ display: 'flex', gap: 16, fontSize: '0.82rem', color: '#666', margin: '6px 0' }}>
                      <span>Before: <strong>{c.old_value}</strong></span>
                      <span>→</span>
                      <span>After: <strong>{c.new_value}</strong></span>
                    </div>
                    <p style={{ fontSize: '0.82rem', color: '#555', lineHeight: 1.5 }}>{c.explanation}</p>
                  </div>
                );
              })}
            </>
          )}

          {/* Positive changes */}
          {result.positive_changes?.length > 0 && (
            <div style={{ background: '#dcfce7', border: '1px solid #bbf7d0', borderRadius: 10, padding: 14, marginTop: 14 }}>
              <strong style={{ color: '#15803d', fontSize: '0.88rem' }}>✅ Improvements:</strong>
              {result.positive_changes.map((p, i) => <div key={i} style={{ fontSize: '0.84rem', color: '#166534', marginTop: 4 }}>• {p}</div>)}
            </div>
          )}

          {/* Concerns */}
          {result.concerns?.length > 0 && (
            <div style={{ background: '#fef2f2', border: '1px solid #fca5a5', borderRadius: 10, padding: 14, marginTop: 14 }}>
              <strong style={{ color: '#b91c1c', fontSize: '0.88rem' }}>⚠️ Concerns:</strong>
              {result.concerns.map((c, i) => <div key={i} style={{ fontSize: '0.84rem', color: '#991b1b', marginTop: 4 }}>• {c}</div>)}
            </div>
          )}

          {/* Follow up */}
          {result.follow_up && (
            <div style={{
              background: '#fff7ed', border: '1px solid #fed7aa', borderRadius: 10,
              padding: '13px 16px', marginTop: 14, fontSize: '0.82rem', color: '#9a3412', lineHeight: 1.6,
            }}>
              👨‍⚕️ <strong>Next Steps:</strong> {result.follow_up}
              {result.specialist && <><br />Discuss with a <strong>{result.specialist}</strong>.</>}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
