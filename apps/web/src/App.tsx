import React, { useEffect, useMemo, useRef, useState } from 'react'
import { chat, ingest, type ChatResponse } from './api'
import { Plus, Upload, Send, Copy, Check, Database, Sparkles, FolderPlus, MessageSquare } from 'lucide-react'
import clsx from 'clsx'

type Session = { id: string; name: string }

function useLocalSessions() {
  const [sessions, setSessions] = useState<Session[]>(() => {
    const raw = localStorage.getItem('sessions')
    return raw ? JSON.parse(raw) : [{ id: 'sessA', name: 'Session A' }]
  })
  useEffect(() => localStorage.setItem('sessions', JSON.stringify(sessions)), [sessions])
  return { sessions, setSessions }
}

function Sidebar({ userId, current, onSelect, onNew, onUpload }: {
  userId: string
  current: string
  onSelect: (id: string) => void
  onNew: () => void
  onUpload: () => void
}) {
  const { sessions, setSessions } = useLocalSessions()
  const handleNew = () => {
    const newId = 'sess' + Math.random().toString(36).slice(2, 6)
    const newName = `Session ${sessions.length + 1}`
    const newSessions = [...sessions, { id: newId, name: newName }]
    setSessions(newSessions)
    onSelect(newId)
  }
  return (
    <aside className="h-screen w-72 bg-panel ring-1 ring-white/5 flex flex-col fixed left-0 top-0 z-10">
      <div className="p-4 flex items-center gap-2 border-b border-white/5">
        <Sparkles className="text-accent" />
        <div className="font-semibold">AI Data Analyst</div>
      </div>
      <div className="flex items-center justify-between p-3 border-b border-white/5">
        <div className="text-xs text-white/60">User: {userId}</div>
        <button className="btn-ghost" onClick={handleNew}><Plus size={16}/> New</button>
      </div>
      <div className="flex-1 overflow-auto px-2 py-2 space-y-1">
        {sessions.map(s => (
          <button key={s.id} onClick={() => onSelect(s.id)} className={clsx('w-full px-3 py-2 rounded-md text-left flex items-center gap-2', current===s.id ? 'bg-accent/20 ring-1 ring-accent/50' : 'hover:bg-panelAlt') }>
            <MessageSquare size={16} className="text-accent"/>
            <span className="truncate">{s.name}</span>
          </button>
        ))}
      </div>
      <div className="mt-auto p-3 border-t border-white/5">
        <button className="btn w-full" onClick={onUpload}><FolderPlus size={16}/> Ingest Dataset</button>
        <p className="text-xs text-white/50 mt-2 flex items-center gap-2"><Database size={14}/> Session-aware storage</p>
      </div>
    </aside>
  )
}

function OutputPanel({ out, onCopy }: { out: ChatResponse | null, onCopy: (code: string)=>void }) {
  const [copied, setCopied] = useState(false)
  useEffect(() => setCopied(false), [out?.executed_code])
  return (
    <div className="space-y-4">
      <div className="card">
        <div className="text-sm uppercase tracking-wide text-white/60 mb-1">Summary</div>
        <div className="leading-relaxed">{out?.summary || 'No summary yet.'}</div>
      </div>

      <div className="card">
        <div className="flex items-center justify-between mb-2">
          <div className="text-sm uppercase tracking-wide text-white/60">Code</div>
          <button className="btn-ghost" onClick={() => { if(out?.executed_code){ onCopy(out.executed_code); setCopied(true); setTimeout(()=>setCopied(false), 1200);} }}>
            {copied ? <Check size={16}/> : <Copy size={16}/>} {copied ? 'Copied' : 'Copy'}
          </button>
        </div>
        <pre className="code">{out?.executed_code || '# Code will appear here'}</pre>
      </div>

      {out?.preview && out.preview.length>0 && (
        <div className="card">
          <div className="text-sm uppercase tracking-wide text-white/60 mb-2">Preview</div>
          <div className="overflow-auto">
            <table className="table">
              <thead>
                <tr>
                  {Object.keys(out.preview[0]).map(k => <th key={k}>{k}</th>)}
                </tr>
              </thead>
              <tbody>
                {out.preview.slice(0,20).map((row, i) => (
                  <tr key={i}>
                    {Object.keys(out.preview[0]).map(k => <td key={k}>{String(row[k])}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {out?.suggested_next && out.suggested_next.length>0 && (
        <div className="card">
          <div className="text-sm uppercase tracking-wide text-white/60 mb-2">Suggested next steps</div>
          <ul className="list-disc pl-6 space-y-1">
            {out.suggested_next.map((s,i)=>(<li key={i}>{s}</li>))}
          </ul>
        </div>
      )}
    </div>
  )
}

type ChatTurn = { question: string; answer: ChatResponse }

function getHistory(userId: string, sessionId: string): ChatTurn[] {
  const key = `chat_${userId}_${sessionId}`
  try {
    return JSON.parse(localStorage.getItem(key) || '[]')
  } catch {
    return []
  }
}
function saveHistory(userId: string, sessionId: string, history: ChatTurn[]) {
  const key = `chat_${userId}_${sessionId}`
  localStorage.setItem(key, JSON.stringify(history))
}

export default function App() {
  const [userId, setUserId] = useState('user1')
  const [sessionId, setSessionId] = useState('sessA')
  const [message, setMessage] = useState('')
  const [loading, setLoading] = useState(false)
  const [history, setHistory] = useState<ChatTurn[]>(() => getHistory('user1', 'sessA'))
  const fileInput = useRef<HTMLInputElement>(null)
  const chatBottom = useRef<HTMLDivElement>(null)

  // Load chat history when session/user changes
  useEffect(() => {
    setHistory(getHistory(userId, sessionId))
  }, [userId, sessionId])

  // Scroll to bottom on new message
  useEffect(() => {
    chatBottom.current?.scrollIntoView({ behavior: 'smooth' })
  }, [history, loading])

  const onUpload = () => fileInput.current?.click()
  const handleFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (!f) return
    try {
      setLoading(true)
      await ingest(userId, sessionId, f)
    } catch (e) {
      console.error(e)
      alert('Upload failed: ' + (e as Error).message)
    } finally {
      setLoading(false)
      e.target.value = ''
    }
  }

  const send = async () => {
    if (!message.trim()) return
    setLoading(true)
    const q = message
    setMessage('')
    try {
      const r = await chat({ user_id: userId, session_id: sessionId, message: q })
      const newHistory = [...history, { question: q, answer: r }]
      setHistory(newHistory)
      saveHistory(userId, sessionId, newHistory)
    } catch (e) {
      alert('Error: ' + (e as Error).message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="h-screen bg-bg">
      <Sidebar userId={userId} current={sessionId} onSelect={setSessionId} onNew={() => {}} onUpload={onUpload} />
      <main className="h-full flex flex-col ml-72">
        <header className="p-4 border-b border-white/5 flex items-center gap-3">
          <input className="input max-w-xs" placeholder="User ID" value={userId} onChange={e=>setUserId(e.target.value)} />
          <input className="input max-w-xs" placeholder="Session ID" value={sessionId} onChange={e=>setSessionId(e.target.value)} />
        </header>
        <div className="flex-1 overflow-auto p-6 flex flex-col gap-3" style={{scrollBehavior:'smooth'}}>
          {history.length === 0 && <div className="text-white/40 text-center mt-10">Start a conversation about your data.</div>}
          {history.map((turn, i) => (
            <div key={i} className="flex flex-col gap-1">
              <div className="self-end max-w-2xl bg-accent/80 text-white px-4 py-2 rounded-lg shadow-glow mb-1">
                <span className="font-semibold">You:</span> {turn.question}
              </div>
              <div className="self-start max-w-2xl bg-panelAlt text-white px-4 py-3 rounded-lg shadow-glow">
                <OutputPanel out={turn.answer} onCopy={(code)=>navigator.clipboard.writeText(code)} />
              </div>
            </div>
          ))}
          {loading && <div className="self-start text-white/60">Thinking…</div>}
          <div ref={chatBottom} />
        </div>
        <footer className="p-4 border-t border-white/5">
          <div className="flex gap-2">
            <input className="input flex-1" placeholder="Ask about the dataset…" value={message} onChange={e=>setMessage(e.target.value)} onKeyDown={e=>{ if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); send(); } }} disabled={loading} />
            <button className={clsx('btn', loading && 'opacity-70 pointer-events-none')} onClick={send}><Send size={16}/> Ask</button>
            <input type="file" ref={fileInput} onChange={handleFile} className="hidden" accept=".csv,.xls,.xlsx" />
          </div>
          <div className="text-xs text-white/40 mt-2">Outputs show: summary → code (copy) → preview → suggestions</div>
        </footer>
      </main>
    </div>
  )
}
