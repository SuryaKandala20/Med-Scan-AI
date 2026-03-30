import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { streamChat } from '../utils/api';
import AssessmentCard from './AssessmentCard';

const GREETING = `Hi! I'm **MedScan AI**, your medical assistant powered by GPT-4o with a medical knowledge base.\n\n**Tell me what's bothering you** — describe how you're feeling in your own words.\n\n_Remember: I'm an educational tool, not a real doctor._`;

const LANGUAGES = [
  'English', 'Hindi', 'Spanish', 'French', 'German', 'Telugu',
  'Tamil', 'Bengali', 'Marathi', 'Urdu', 'Arabic', 'Chinese',
  'Japanese', 'Korean', 'Portuguese', 'Russian',
];

export default function ChatPage() {
  const [messages, setMessages] = useState([
    { role: 'bot', content: GREETING }
  ]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [sessionId, setSessionId] = useState('');
  const [language, setLanguage] = useState('English');
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const sendMessage = async () => {
    const text = input.trim();
    if (!text || isStreaming) return;

    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: text }]);
    setIsStreaming(true);

    // Add empty bot message that we'll stream into
    let botContent = '';
    let assessment = null;

    setMessages(prev => [...prev, { role: 'bot', content: '', streaming: true }]);

    await streamChat(
      text,
      sessionId,
      language,
      // onToken — append each token
      (token) => {
        botContent += token;
        setMessages(prev => {
          const updated = [...prev];
          updated[updated.length - 1] = { role: 'bot', content: botContent, streaming: true };
          return updated;
        });
      },
      // onAssessment
      (data) => {
        assessment = data;
      },
      // onDone
      (newSessionId, latencyMs) => {
        setSessionId(newSessionId);
        setMessages(prev => {
          const updated = [...prev];
          updated[updated.length - 1] = {
            role: 'bot',
            content: botContent,
            assessment: assessment,
            streaming: false,
            latency: latencyMs,
          };
          return updated;
        });
        setIsStreaming(false);
      },
      // onError
      (error) => {
        setMessages(prev => {
          const updated = [...prev];
          updated[updated.length - 1] = {
            role: 'bot',
            content: `⚠️ Error: ${error}`,
            streaming: false,
          };
          return updated;
        });
        setIsStreaming(false);
      }
    );
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-messages" ref={scrollRef}>
        {messages.map((msg, i) => (
          <React.Fragment key={i}>
            <div className={`msg-row ${msg.role === 'user' ? 'user' : 'bot'}`}>
              <div className={`avatar ${msg.role === 'user' ? 'user' : 'bot'}`}>
                {msg.role === 'user' ? '👤' : '🏥'}
              </div>
              <div className="bubble">
                {msg.streaming && msg.content === '' ? (
                  <div className="typing-dots">
                    <span /><span /><span />
                  </div>
                ) : (
                  <ReactMarkdown>{msg.content}</ReactMarkdown>
                )}
              </div>
            </div>
            {msg.assessment && <AssessmentCard data={msg.assessment} />}
          </React.Fragment>
        ))}
      </div>

      <div className="chat-input-bar">
        <select
          value={language}
          onChange={(e) => setLanguage(e.target.value)}
          style={{
            padding: '10px 8px', borderRadius: 10, border: '1.5px solid #e5e7eb',
            fontSize: '0.78rem', fontFamily: 'inherit', background: '#fafbfc',
            cursor: 'pointer', outline: 'none', color: '#555', width: 90,
          }}
        >
          {LANGUAGES.map(l => <option key={l} value={l}>{l}</option>)}
        </select>
        <input
          className="chat-input"
          type="text"
          placeholder="Describe how you're feeling..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isStreaming}
        />
        <button className="send-btn" onClick={sendMessage} disabled={isStreaming || !input.trim()}>
          ➤
        </button>
      </div>
    </div>
  );
}
