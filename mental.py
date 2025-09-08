""" Mental Health Assistant (Open‑Source Models) – FastAPI app

This single‑file app provides:

A JSON API for safe, supportive mental‑health Q&A using an open‑source LLM (via Ollama).

A lightweight in‑browser chat UI served from "/".

A simple recommender that suggests evidence‑based self‑help procedures.

Guardrails for safety (crisis detection & medical‑advice avoidance).


⚠️ Important: This app is for education and support only. It does not diagnose, treat, or replace professional care.

Quick start

1. Install Python 3.10+


2. Install dependencies: pip install fastapi uvicorn pydantic requests python-multipart jinja2


3. Install Ollama and pull a model (choose ONE):

https://ollama.com/download

ollama pull llama3.1:8b\n     (or e.g. ollama pull mistral:7b)



4. Run the app: uvicorn app:app --reload --port 8000


5. Open http://localhost:8000



Env vars (optional):

OLLAMA_URL (default http://localhost:11434)

OLLAMA_MODEL (default "llama3.1:8b")


"""

from future import annotations

import os import re import json from typing import List, Optional, Dict, Any

import requests from fastapi import FastAPI, HTTPException, Body from fastapi.middleware.cors import CORSMiddleware from fastapi.responses import HTMLResponse from pydantic import BaseModel, Field

---------------------------

Configuration

---------------------------

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434") OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")  # try "mistral:7b" if preferred

SYSTEM_PROMPT = ( "You are CalmCare, a supportive mental-health assistant.\n" "Your priorities: be empathetic, validate feelings, educate gently, and encourage professional help when appropriate.\n" "SAFETY RULES (strict):\n" "- You are NOT a medical professional. Do not give medical, legal, or diagnostic advice.\n" "- Do not provide step-by-step instructions for self-harm or dangerous activities.\n" "- If the user expresses intent to harm themselves or others, or mentions being in immediate danger,\n" "  respond with a brief, compassionate crisis message directing them to emergency services and crisis resources,\n" "  and avoid further content beyond grounding and staying safe.\n" "- Use inclusive, non-judgmental language.\n" "CONTENT STYLE:\n" "- Keep answers concise and clear, use bullet points where helpful.\n" "- Offer general psychoeducation and practical, low-risk self-help options (e.g., grounding, breathing, journaling, sleep hygiene).\n" "- Encourage seeking a licensed professional for diagnosis/treatment decisions.\n" )

CRISIS_REGEX = re.compile( r"(suicide|kill myself|end my life|want to die|self-harm|self harm|hurt myself|\bcan\si\sdie\b|overdose|\bkill (him|her|them|someone)\b|homicide|murder)", re.IGNORECASE, )

Minimal, non-country-specific crisis copy. Customize locally before deployment.

CRISIS_MESSAGE = ( "I'm really sorry that you're feeling this way. Your safety matters. If you might act on these thoughts or you're in immediate danger, " "please contact local emergency services right now (for many regions, dialing your local emergency number). If available in your area, " "you can also reach a suicide and crisis hotline or speak to a trusted person near you. If you can, consider going to a safe place or " "an emergency department. You're not alone—help is available." )

---------------------------

Simple procedure recommender

---------------------------

class Procedure(BaseModel): title: str steps: List[str] caution: Optional[str] = None

PROCEDURE_LIBRARY: Dict[str, Procedure] = { "anxiety": Procedure( title="Managing Anxiety (general)", steps=[ "Box breathing (4-4-4-4): inhale 4s, hold 4s, exhale 4s, hold 4s for 3–5 minutes.", "Name–Locate–Ground: name 5 things you see, 4 touch, 3 hear, 2 smell, 1 taste.", "Worry window: schedule a 10–15 min daily slot to write worries; outside that window, jot and postpone.", "Limit caffeine late day; keep a consistent sleep schedule.", "If anxiety persists or disrupts life, consider talking with a licensed mental health professional.", ], ), "low_mood": Procedure( title="Low Mood Support", steps=[ "Behavioral activation: plan 1–2 small, meaningful activities daily (walk, call a friend, light chore).", "Self-compassion check-in: write to yourself as you would to a friend—kind, specific, non-judgmental.", "Sunlight & movement: aim for 10–20 min daylight exposure and gentle movement.", "Sleep hygiene: consistent bedtime, reduce screens 1 hour before, quiet/dim room.", "If low mood lasts 2+ weeks or worsens, seek a licensed professional for assessment.", ], ), "sleep": Procedure( title="Better Sleep (insomnia-lite hygiene)", steps=[ "Fixed wake time (even weekends).", "Wind-down routine 30–60 min: dim lights, quiet activity, no news/doomscrolling.", "Bed for sleep/sex only; if awake >20 min, get up for a calm activity until drowsy.", "Limit naps to 20–30 min before 3pm; watch caffeine after midday.", ], ), "panic": Procedure( title="Panic Soothing (in the moment)", steps=[ "Remind yourself panic peaks then passes (10–20 min).", "Paced breathing: exhale slightly longer than inhale (e.g., 4 in / 6 out).", "Temperature shift: cool water on face or hold a cool object.", "Anchor to senses: describe out loud what's around you (colors, textures, sounds).", ], ), }

KEYWORD_TO_TOPIC = [ (re.compile(r"(anxious|anxiety|worry|worried|panic|overthinking)", re.I), "anxiety"), (re.compile(r"(sad|down|depressed|low mood|no motivation)", re.I), "low_mood"), (re.compile(r"(sleep|insomnia|can't sleep|cant sleep|sleeping)", re.I), "sleep"), (re.compile(r"(panic attack|heart racing|hyperventilating)", re.I), "panic"), ]

---------------------------

LLM client (Ollama chat API)

---------------------------

def ollama_chat(messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 512) -> str: url = f"{OLLAMA_URL}/api/chat" payload = { "model": OLLAMA_MODEL, "messages": messages, "options": {"temperature": temperature, "num_predict": max_tokens}, "stream": False, } try: resp = requests.post(url, json=payload, timeout=120) resp.raise_for_status() data = resp.json() # Ollama returns {"message": {"role": "assistant", "content": "..."}, ...} return data.get("message", {}).get("content", "") except requests.RequestException as e: raise HTTPException(status_code=502, detail=f"LLM backend error: {e}")

---------------------------

FastAPI schemas

---------------------------

class ChatRequest(BaseModel): user_message: str = Field(..., min_length=1, max_length=4000) context: Optional[List[Dict[str, str]]] = Field( default=None, description="Optional chat history as [{'role':'user'|'assistant','content':'...'}]", )

class ChatResponse(BaseModel): crisis: bool message: str

class RecoRequest(BaseModel): concern: Optional[str] = Field(None, description="Free text describing the issue") topic: Optional[str] = Field(None, description="One of: anxiety, low_mood, sleep, panic")

class RecoResponse(BaseModel): title: str steps: List[str] caution: Optional[str] = None

---------------------------

App setup

---------------------------

app = FastAPI(title="CalmCare – Mental Health Assistant (Open‑Source)") app.add_middleware( CORSMiddleware, allow_origins=[""], allow_credentials=True, allow_methods=[""], allow_headers=["*"], )

---------------------------

Routes

---------------------------

@app.get("/", response_class=HTMLResponse) def home() -> str: return f""" <!doctype html>

<html>
<head>
  <meta charset='utf-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1' />
  <title>CalmCare – Mental Health Assistant</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background: #0b1020; color: #e8ecf2; }}
    .wrap {{ max-width: 900px; margin: 0 auto; padding: 24px; }}
    .card {{ background: #121833; border-radius: 16px; padding: 16px; box-shadow: 0 10px 30px rgba(0,0,0,.25); }}
    .row {{ display:flex; gap:8px; margin-top:12px; }}
    textarea {{ width:100%; min-height: 90px; border-radius:12px; border:1px solid #263056; padding:12px; background:#0b1129; color:#e8ecf2; }}
    button {{ border:0; background:#3b82f6; color:white; padding:10px 16px; border-radius:12px; font-weight:600; cursor:pointer; }}
    button:disabled {{ opacity:.6; cursor:not-allowed; }}
    .msg {{ padding:12px 14px; border-radius:12px; margin:10px 0; line-height:1.5; white-space: pre-wrap; }}
    .u {{ background:#0b1129; border:1px solid #1e293b; }}
    .a {{ background:#0c142c; border:1px solid #243b69; }}
    .small {{ opacity:.8; font-size:.9em; }}
    .footer {{ margin-top:16px; font-size:.85em; opacity:.75; }}
    .badge {{ display:inline-block; padding:4px 8px; background:#1f2a4d; border-radius:999px; font-size:.75em; margin-right:6px; border:1px solid #2d3b6b; }}
  </style>
</head>
<body>
  <div class='wrap'>
    <h1>🫶 CalmCare</h1>
    <div class='card'>
      <div id='chat'></div>
      <div class='row'>
        <textarea id='input' placeholder='Ask a supportive, mental‑health related question… (This is not medical advice)'></textarea>
      </div>
      <div class='row'>
        <button id='send'>Send</button>
        <button id='reco'>Recommend a procedure</button>
        <span class='small'>Model: <span class='badge'>{OLLAMA_MODEL}</span> via Ollama</span>
      </div>
      <div class='footer'>
        ⚠️ Educational support only. Not a substitute for diagnosis or treatment. If you might act on thoughts of harming yourself or others, contact local emergency services immediately.
      </div>
    </div>
  </div>
  <script>
    const chatEl = document.getElementById('chat');
    const inputEl = document.getElementById('input');
    const sendBtn = document.getElementById('send');
    const recoBtn = document.getElementById('reco');
    let history = [];function addMsg(role, content){
  const div = document.createElement('div');
  div.className = 'msg ' + (role === 'user' ? 'u' : 'a');
  div.textContent = content;
  chatEl.appendChild(div);
  window.scrollTo(0, document.body.scrollHeight);
}

async function chat(){
  const text = inputEl.value.trim();
  if(!text) return;
  inputEl.value = '';
  addMsg('user', text);
  sendBtn.disabled = true;
  try{
    const res = await fetch('/api/chat', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ user_message: text, context: history }) });
    const data = await res.json();
    if(res.ok){
      addMsg('assistant', data.message);
      history.push({role:'user', content: text});
      history.push({role:'assistant', content: data.message});
    } else {
      addMsg('assistant', 'Error: ' + (data.detail || 'Something went wrong'));
    }
  } catch(e){
    addMsg('assistant', 'Network error: ' + e.message);
  } finally {
    sendBtn.disabled = false;
  }
}

async function recommend(){
  const text = inputEl.value.trim();
  const body = text ? { concern: text } : {};
  try{
    const res = await fetch('/api/recommend', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
    const data = await res.json();
    if(res.ok){
      const md = `**${data.title}**\n\n` + data.steps.map((s,i)=>`${i+1}. ${s}`).join('\n');
      addMsg('assistant', md);
      history.push({role:'assistant', content: md});
    } else {
      addMsg('assistant', 'Error: ' + (data.detail || 'Could not recommend'));
    }
  } catch(e){
    addMsg('assistant', 'Network error: ' + e.message);
  }
}

sendBtn.addEventListener('click', chat);
recoBtn.addEventListener('click', recommend);
inputEl.addEventListener('keydown', (e)=>{ if(e.key==='Enter' && (e.ctrlKey||e.metaKey)) chat(); });

  </script>
</body>
</html>
    """@app.post("/api/chat", response_model=ChatResponse) def api_chat(payload: ChatRequest = Body(...)) -> ChatResponse: user_text = payload.user_message.strip()

# Crisis guardrail
if CRISIS_REGEX.search(user_text):
    return ChatResponse(crisis=True, message=CRISIS_MESSAGE)

# Compose messages for the LLM
messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
if payload.context:
    # (Truncate to last 8 turns to keep prompt small)
    trimmed = payload.context[-8:]
    for m in trimmed:
        if m.get("role") in {"user", "assistant"} and isinstance(m.get("content"), str):
            messages.append({"role": m["role"], "content": m["content"]})
messages.append({"role": "user", "content": user_text})

# Ask the model
answer = ollama_chat(messages, temperature=0.2, max_tokens=512)

# Soft post-filter: keep it supportive & non-clinical
SAFETY_FOOTER = (
    "\n\n—\nThis information is educational and not a medical diagnosis. If symptoms persist, consider talking to a licensed professional."
)
clean = answer.strip()
if not clean:
    clean = "I'm here for you. Could you share a bit more about what's going on?" + SAFETY_FOOTER
elif len(clean) < 24:
    clean += SAFETY_FOOTER
else:
    # Ensure no explicit diagnosis terms like "you have X disorder".
    clean = re.sub(r"\b(you (have|are experiencing) (.+? disorder))\b", "It may help to discuss these experiences with a licensed professional who can provide an assessment", clean, flags=re.I)
    clean += SAFETY_FOOTER

return ChatResponse(crisis=False, message=clean)

@app.post("/api/recommend", response_model=RecoResponse) def api_recommend(payload: RecoRequest = Body(default=RecoRequest())) -> RecoResponse: topic = payload.topic

# Infer a topic from the free-text concern if none specified
if not topic and payload.concern:
    txt = payload.concern
    for rx, t in KEYWORD_TO_TOPIC:
        if rx.search(txt):
            topic = t
            break

# Default fallback
if not topic:
    topic = "anxiety"

proc = PROCEDURE_LIBRARY.get(topic)
if not proc:
    raise HTTPException(status_code=400, detail=f"Unknown topic '{topic}'. Try: anxiety, low_mood, sleep, panic.")

return RecoResponse(**proc.model_dump())

Health check

@app.get("/healthz") def healthz() -> Dict[str, Any]: return {"ok": True, "model": OLLAMA_MODEL, "backend": OLLAMA_URL}

if name == "main": import uvicorn uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

