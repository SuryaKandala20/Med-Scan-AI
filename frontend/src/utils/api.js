const API = 'http://localhost:8000';

export async function streamChat(message, sessionId, language, onToken, onAssessment, onDone, onError) {
  try {
    const res = await fetch(`${API}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, session_id: sessionId, language: language || 'English' }),
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let newSessionId = sessionId;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const data = JSON.parse(line.slice(6));
          if (data.type === 'session') newSessionId = data.session_id;
          else if (data.type === 'token') onToken(data.content);
          else if (data.type === 'assessment') onAssessment(data.data);
          else if (data.type === 'done') onDone(newSessionId, data.latency_ms);
          else if (data.type === 'error') onError(data.message);
        } catch {}
      }
    }
  } catch (err) {
    onError(err.message);
  }
}

export async function explainReport(text, file) {
  const form = new FormData();
  form.append('text', text || '');
  if (file) form.append('file', file);

  const res = await fetch(`${API}/api/report`, { method: 'POST', body: form });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Report analysis failed');
  }
  return res.json();
}

export async function getAdminStats() {
  const res = await fetch(`${API}/api/admin/stats`);
  return res.json();
}

export async function getHealth() {
  const res = await fetch(`${API}/api/health`);
  return res.json();
}

export async function resetChat(sessionId) {
  await fetch(`${API}/api/chat/reset`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId }),
  });
}
