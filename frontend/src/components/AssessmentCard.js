import React from 'react';

export default function AssessmentCard({ data }) {
  if (!data) return null;

  const conditions = data.conditions || [];
  const triage = data.triage || {};
  const treatments = data.treatments || [];
  const doctor = data.doctor_referral || {};
  const emergency = data.emergency_warning;

  const triageClass = {
    Emergency: 'tb-emergency', Urgent: 'tb-urgent',
    'Same-day': 'tb-same-day', Routine: 'tb-routine',
  }[triage.level] || 'tb-routine';

  const barColors = { High: '#2563eb', Moderate: '#ea580c', Low: '#d1d5db' };
  const barWidths = { High: 85, Moderate: 55, Low: 28 };

  return (
    <div className="assess-card">
      <div className="assess-header">
        <span style={{ fontSize: '1.3rem' }}>🧠</span>
        <div style={{ flex: 1 }}>
          <h3>Symptom Assessment</h3>
          <div className="sub">GPT-4o + ChromaDB RAG</div>
        </div>
        <span className={`triage-badge ${triageClass}`}>{triage.level || 'Routine'}</span>
      </div>

      <div className="assess-body">
        {emergency && (
          <div className="doc-banner danger">🚨 {emergency}</div>
        )}

        <div style={{ fontSize: '0.68rem', fontWeight: 700, color: '#999', textTransform: 'uppercase', letterSpacing: '0.06em', margin: '12px 0 8px' }}>
          Possible Conditions
        </div>

        {conditions.map((c, i) => (
          <div className="cond-row" key={i}>
            <div className="rank">{c.rank || i + 1}</div>
            <div style={{ flex: 1 }}>
              <div className="cond-name">{c.name}</div>
              <div className="cond-meta">
                {c.likelihood} likelihood
                {c.matching_symptoms?.length > 0 && ` • ${c.matching_symptoms.join(', ')}`}
              </div>
              {c.description && (
                <div style={{ fontSize: '0.78rem', color: '#666', marginTop: 4, lineHeight: 1.5 }}>
                  {c.description}
                </div>
              )}
            </div>
            <div className="bar">
              <div className="bar-fill" style={{
                width: `${barWidths[c.likelihood] || 50}%`,
                background: barColors[c.likelihood] || '#999'
              }} />
            </div>
          </div>
        ))}

        {treatments.length > 0 && (
          <div className="tx-box">
            <h4>💊 Care & Treatment Tips</h4>
            {treatments.map((t, i) => <div className="tx-item" key={i}>{t}</div>)}
          </div>
        )}

        {doctor.specialty && (
          <div className={`doc-banner ${triage.level === 'Emergency' ? 'danger' : 'warn'}`}>
            {doctor.icon || '👨‍⚕️'} <strong>{doctor.specialty}</strong>
            <br />{doctor.message}
            {doctor.urgency && <><br /><em>{doctor.urgency}</em></>}
          </div>
        )}
      </div>
    </div>
  );
}
