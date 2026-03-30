import React, { useState } from 'react';

export default function ReportPage() {
  const [text, setText] = useState('');
  const [fileName, setFileName] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [expandedTerms, setExpandedTerms] = useState({});
  const [expandedDetails, setExpandedDetails] = useState({});

  const handleFileChange = (e) => {
    const f = e.target.files[0];
    if (f) {
      setFileName(f.name);
    }
  };

  const getErrorMessage = (errData) => {
    if (!errData) return 'Analysis failed';

    if (typeof errData === 'string') return errData;
    if (typeof errData.detail === 'string') return errData.detail;
    if (errData.detail?.message) return errData.detail.message;
    if (typeof errData.message === 'string') return errData.message;

    if (Array.isArray(errData.detail)) {
      return errData.detail.map((x) => x.msg || JSON.stringify(x)).join(', ');
    }

    if (errData.detail && typeof errData.detail === 'object') {
      return JSON.stringify(errData.detail, null, 2);
    }

    return 'Analysis failed';
  };

  const handleSubmit = async () => {
    const fileInput = document.getElementById('report-file-input');
    const selectedFile = fileInput?.files?.[0];

    if (!text.trim() && !selectedFile) {
      setError('Please paste report text or upload a supported file');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const form = new FormData();
      form.append('text', text || '');
      if (selectedFile) {
        form.append('file', selectedFile);
      }

      const res = await fetch('http://localhost:8000/api/report', {
        method: 'POST',
        body: form,
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(getErrorMessage(data));
      }

      if (data.error) {
        throw new Error(data.error);
      }

      setResult(data);
      setExpandedTerms({});
      setExpandedDetails({});
    } catch (err) {
      setError(err.message || 'Analysis failed');
    } finally {
      setLoading(false);
    }
  };

  const clearFile = () => {
    const fileInput = document.getElementById('report-file-input');
    if (fileInput) fileInput.value = '';
    setFileName('');
  };

  const toggleTerm = (i) => {
    setExpandedTerms((prev) => ({ ...prev, [i]: !prev[i] }));
  };

  const toggleDetail = (i) => {
    setExpandedDetails((prev) => ({ ...prev, [i]: !prev[i] }));
  };

  const findingColor = (status) => {
    if (status === 'Normal') return '#16a34a';
    if (status === 'Abnormal') return '#dc2626';
    return '#ca8a04';
  };

  const urgencyClass = (level) => {
    if (level === 'High') return 'urgency-high';
    if (level === 'Moderate') return 'urgency-moderate';
    return 'urgency-low';
  };

  const renderBulletList = (items) => {
    if (!items?.length) return null;

    return (
      <ul style={{ margin: '8px 0 0 18px', padding: 0 }}>
        {items.map((item, i) => (
          <li key={i} style={{ marginBottom: 6, color: '#444', fontSize: '0.86rem', lineHeight: 1.5 }}>
            {item}
          </li>
        ))}
      </ul>
    );
  };

  return (
    <div className="report-page">
      <h2>📄 Medical Report Explainer</h2>
      <p className="subtitle">
        Paste a report or upload a file — GPT-4o explains it in plain language
      </p>

      <textarea
        className="report-textarea"
        placeholder="Paste your medical report here (lab results, radiology, pathology...)"
        value={text}
        onChange={(e) => setText(e.target.value)}
      />

      <div className="upload-row">
        <input
          type="file"
          id="report-file-input"
          accept=".pdf,.txt,.png,.jpg,.jpeg,.webp"
          style={{ display: 'none' }}
          onChange={handleFileChange}
        />
        <button
          className="upload-btn"
          onClick={() => document.getElementById('report-file-input')?.click()}
        >
          📎 {fileName || 'Upload PDF, TXT, or Image'}
        </button>

        {fileName && (
          <button
            style={{
              background: 'none',
              border: 'none',
              color: '#999',
              cursor: 'pointer',
              fontSize: '0.82rem',
            }}
            onClick={clearFile}
          >
            ✕ Remove
          </button>
        )}

        <button
          className="btn-primary"
          onClick={handleSubmit}
          disabled={loading || (!text.trim() && !fileName)}
        >
          {loading ? '⏳ Analyzing...' : '🔍 Explain Report'}
        </button>
      </div>

      {error && (
        <div style={{ color: '#dc2626', marginTop: 12, fontSize: '0.88rem', whiteSpace: 'pre-wrap' }}>
          ⚠️ {error}
        </div>
      )}

      {result && (
        <div className="report-result">
          {result.summary && (
            <div className="summary-box">
              <strong>📋 Summary:</strong> {result.summary}
            </div>
          )}

          {result.urgency && (
            <div className={`urgency-box ${urgencyClass(result.urgency.level)}`}>
              <strong>Urgency: {result.urgency.level}</strong> — {result.urgency.message}
            </div>
          )}

          {result.key_findings?.length > 0 && (
            <>
              <h3 style={{ fontSize: '0.95rem', margin: '20px 0 10px' }}>🔍 Key Findings</h3>
              {result.key_findings.map((f, i) => (
                <div className="finding-row" key={i}>
                  <div className="finding-dot" style={{ background: findingColor(f.status) }} />
                  <div>
                    <strong>{f.finding}</strong> ({f.status})
                    <div style={{ color: '#666', fontSize: '0.82rem', marginTop: 2 }}>
                      {f.explanation}
                    </div>
                  </div>
                </div>
              ))}
            </>
          )}

          {result.detailed_key_findings?.length > 0 && (
            <>
              <h3 style={{ fontSize: '0.95rem', margin: '22px 0 10px' }}>
                🧠 Understand Your Findings in More Detail
              </h3>

              {result.detailed_key_findings.map((item, i) => (
                <div
                  key={i}
                  style={{
                    border: '1px solid #e5e7eb',
                    borderRadius: 14,
                    padding: 14,
                    marginBottom: 12,
                    background: '#fff',
                    boxShadow: '0 1px 2px rgba(0,0,0,0.04)',
                  }}
                >
                  <div
                    onClick={() => toggleDetail(i)}
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      cursor: 'pointer',
                      fontWeight: 600,
                      fontSize: '0.95rem',
                    }}
                  >
                    <span>{item.finding}</span>
                    <span style={{ color: '#666', fontSize: '0.85rem' }}>
                      {expandedDetails[i] ? '▾' : '▸'}
                    </span>
                  </div>

                  {expandedDetails[i] && (
                    <div style={{ marginTop: 12 }}>
                      {item.report_specific_summary && (
                        <div style={{ marginBottom: 10, fontSize: '0.86rem', lineHeight: 1.6 }}>
                          <strong>What this report says:</strong> {item.report_specific_summary}
                        </div>
                      )}

                      {item.what_it_means && (
                        <div style={{ marginBottom: 10, fontSize: '0.86rem', lineHeight: 1.6 }}>
                          <strong>What it means:</strong> {item.what_it_means}
                        </div>
                      )}

                      {item.why_it_matters && (
                        <div style={{ marginBottom: 10, fontSize: '0.86rem', lineHeight: 1.6 }}>
                          <strong>Why it matters:</strong> {item.why_it_matters}
                        </div>
                      )}

                      {item.general_context && (
                        <div style={{ marginBottom: 10, fontSize: '0.86rem', lineHeight: 1.6 }}>
                          <strong>General medical context:</strong> {item.general_context}
                        </div>
                      )}

                      {item.what_to_do_next?.length > 0 && (
                        <div style={{ marginBottom: 10 }}>
                          <strong style={{ fontSize: '0.86rem' }}>
                            What people commonly do next:
                          </strong>
                          {renderBulletList(item.what_to_do_next)}
                        </div>
                      )}

                      {item.red_flags?.length > 0 && (
                        <div style={{ marginBottom: 4 }}>
                          <strong style={{ fontSize: '0.86rem' }}>
                            When to get medical advice sooner:
                          </strong>
                          {renderBulletList(item.red_flags)}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </>
          )}

          {result.medical_terms?.length > 0 && (
            <>
              <h3 style={{ fontSize: '0.95rem', margin: '20px 0 10px' }}>📖 Medical Terms Explained</h3>
              {result.medical_terms.map((t, i) => (
                <div className="term-card" key={i} onClick={() => toggleTerm(i)}>
                  <div className="term-name">
                    {t.term} {expandedTerms[i] ? '▾' : '▸'}
                  </div>
                  {expandedTerms[i] && <div className="term-def">{t.explanation}</div>}
                </div>
              ))}
            </>
          )}

          {result.questions_for_doctor?.length > 0 && (
            <>
              <h3 style={{ fontSize: '0.95rem', margin: '20px 0 10px' }}>❓ Questions for Your Doctor</h3>
              {result.questions_for_doctor.map((q, i) => (
                <div key={i} style={{ padding: '4px 0', fontSize: '0.86rem' }}>
                  • {q}
                </div>
              ))}
            </>
          )}

          {result.specialist && (
            <div className="doc-banner warn" style={{ marginTop: 18 }}>
              👨‍⚕️ Discuss these results with a <strong>{result.specialist}</strong>.
            </div>
          )}
        </div>
      )}
    </div>
  );
}