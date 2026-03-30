import React, { useState, useRef } from 'react';

export default function SkinPage() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const fileRef = useRef(null);

  const handleFile = (e) => {
    const f = e.target.files[0];
    if (!f) return;
    setFile(f);
    setResult(null);
    setError('');
    const reader = new FileReader();
    reader.onload = (ev) => setPreview(ev.target.result);
    reader.readAsDataURL(f);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith('image/')) {
      setFile(f);
      setResult(null);
      setError('');
      const reader = new FileReader();
      reader.onload = (ev) => setPreview(ev.target.result);
      reader.readAsDataURL(f);
    }
  };

  const analyze = async () => {
    if (!file) return;
    setLoading(true);
    setError('');
    try {
      const form = new FormData();
      form.append('file', file);
      const res = await fetch('http://localhost:8000/api/skin/predict', { method: 'POST', body: form });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Analysis failed');
      }
      setResult(await res.json());
    } catch (err) {
      setError(err.message);
    }
    setLoading(false);
  };

  const triageColor = (t) => {
    if (t === 'Urgent') return { bg: '#fff7ed', color: '#c2410c', border: '#fed7aa' };
    if (t === 'Emergency') return { bg: '#fef2f2', color: '#b91c1c', border: '#fca5a5' };
    return { bg: '#dcfce7', color: '#15803d', border: '#bbf7d0' };
  };

  return (
    <div style={{ flex: 1, overflowY: 'auto', padding: 30 }}>
      <h2 style={{ fontSize: '1.2rem', marginBottom: 6 }}>🔬 Skin Lesion Analysis</h2>
      <p style={{ color: '#6b7280', fontSize: '0.86rem', marginBottom: 20 }}>
        Upload a photo — EfficientNet-B0 classifies 7 types of skin lesions
      </p>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, maxWidth: 900 }}>
        {/* Left — Upload */}
        <div>
          <div
            onDrop={handleDrop}
            onDragOver={(e) => e.preventDefault()}
            onClick={() => fileRef.current?.click()}
            style={{
              border: '2px dashed #e5e7eb', borderRadius: 16, padding: 40,
              textAlign: 'center', background: '#fafbfc', cursor: 'pointer',
              transition: '0.2s',
            }}
          >
            <input type="file" accept="image/*" ref={fileRef} style={{ display: 'none' }} onChange={handleFile} />
            {preview ? (
              <img src={preview} alt="Upload" style={{ maxWidth: '100%', maxHeight: 250, borderRadius: 12 }} />
            ) : (
              <>
                <div style={{ fontSize: '2.5rem', marginBottom: 12 }}>📸</div>
                <h3 style={{ fontSize: '0.95rem', fontWeight: 600 }}>Drop image here or click to upload</h3>
                <p style={{ fontSize: '0.8rem', color: '#6b7280', marginTop: 6 }}>JPG, PNG • Clear, well-lit photo</p>
              </>
            )}
          </div>

          {file && (
            <button
              onClick={analyze}
              disabled={loading}
              style={{
                marginTop: 14, width: '100%', padding: '12px 0', borderRadius: 12,
                border: 'none', background: '#2563eb', color: '#fff',
                fontSize: '0.9rem', fontWeight: 600, cursor: 'pointer', fontFamily: 'inherit',
              }}
            >
              {loading ? '⏳ Analyzing...' : '🔍 Analyze Skin Lesion'}
            </button>
          )}
        </div>

        {/* Right — Results */}
        <div>
          {error && (
            <div style={{ color: '#dc2626', padding: 14, background: '#fef2f2', borderRadius: 12, fontSize: '0.88rem', marginBottom: 14 }}>
              ⚠️ {error}
            </div>
          )}

          {result && (
            <div style={{ animation: 'fadeUp 0.3s ease-out' }}>
              {/* Quality */}
              {result.quality && !result.quality.passed && (
                <div style={{ background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 10, padding: 12, marginBottom: 14, fontSize: '0.82rem', color: '#92400e' }}>
                  ⚠️ Quality issues: {result.quality.issues.join(', ')}
                </div>
              )}

              {/* Predictions */}
              {result.predictions?.map((p, i) => {
                const tc = triageColor(p.triage);
                return (
                  <div key={i} style={{
                    background: '#fff', borderRadius: 12, padding: '14px 18px', marginBottom: 10,
                    boxShadow: '0 1px 3px rgba(0,0,0,0.05)',
                    borderLeft: `4px solid ${i === 0 ? '#2563eb' : '#e5e7eb'}`,
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                      <div style={{
                        width: 28, height: 28, borderRadius: '50%', background: '#eff4ff',
                        color: '#2563eb', fontSize: '0.75rem', fontWeight: 800,
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                      }}>{i + 1}</div>
                      <div style={{ flex: 1 }}>
                        <div style={{ fontWeight: 700, fontSize: '0.9rem' }}>{p.full_name}</div>
                        <div style={{ fontSize: '0.78rem', color: '#6b7280' }}>
                          {(p.confidence * 100).toFixed(1)}% confidence
                        </div>
                      </div>
                      <span style={{
                        padding: '3px 10px', borderRadius: 16, fontSize: '0.65rem',
                        fontWeight: 700, background: tc.bg, color: tc.color, border: `1px solid ${tc.border}`,
                      }}>{p.triage}</span>
                    </div>
                    <p style={{ fontSize: '0.8rem', color: '#666', marginTop: 8, lineHeight: 1.5 }}>
                      {p.description}
                    </p>
                  </div>
                );
              })}

              {result.predictions?.length > 0 && (
                <div style={{
                  background: '#fff7ed', border: '1px solid #fed7aa', borderRadius: 10,
                  padding: '13px 16px', marginTop: 14, fontSize: '0.82rem', color: '#9a3412', lineHeight: 1.6,
                }}>
                  👨‍⚕️ Please consult a <strong>{result.predictions[0].specialist || 'Dermatologist'}</strong> for professional evaluation.
                </div>
              )}
            </div>
          )}

          {!result && !error && (
            <div style={{ color: '#999', textAlign: 'center', padding: '60px 20px', fontSize: '0.88rem' }}>
              <div style={{ fontSize: '2.5rem', marginBottom: 12 }}>🔬</div>
              Upload an image and click Analyze to see results
              <div style={{
                marginTop: 20, background: '#fffbeb', border: '1px solid #fde68a',
                borderRadius: 10, padding: 12, fontSize: '0.78rem', color: '#92400e', textAlign: 'left',
              }}>
                <strong>Note:</strong> Requires trained model. Run in backend terminal:<br />
                <code>python setup_data.py</code> then <code>python train_model.py</code>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
