from __future__ import annotations

import base64
import json
import os
import tempfile
import threading
import wave
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ZentroMart Voice Support</title>
<style>
:root{font-family:Inter,Segoe UI,Arial,sans-serif;color:#202631;background:#f3f6fa}
*{box-sizing:border-box}body{margin:0}.wrap{max-width:850px;margin:38px auto;padding:0 18px}
header{margin-bottom:22px}h1{font-size:28px;margin:0 0 6px}header p{margin:0;color:#667085}
.card{background:white;border:1px solid #e1e6ee;border-radius:18px;padding:22px;box-shadow:0 8px 28px #192b4510}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.controls{margin:16px 0}
button{border:0;border-radius:10px;padding:12px 18px;font-size:15px;font-weight:650;cursor:pointer}
#record{background:#135fce;color:white}#record.recording{background:#b42318}#reset{background:#eef2f7;color:#25324a}
button:disabled{opacity:.5;cursor:wait}#status{color:#475467;margin:12px 0;min-height:24px}
#turns{display:flex;flex-direction:column;gap:14px;margin-top:18px}.turn{border-top:1px solid #e8edf3;padding-top:14px}
.label{font-size:12px;color:#697586;text-transform:uppercase;letter-spacing:.06em;font-weight:700}
.text{white-space:pre-wrap;line-height:1.5;margin:5px 0 9px}.sources{font-size:12px;color:#667085}
audio{width:min(100%,500px);display:block;margin:8px 0}.hint{font-size:13px;color:#667085;margin-top:14px}
</style>
</head>
<body><main class="wrap">
<header><h1>ZentroMart Voice Support</h1><p>Speak one turn at a time. Gnani transcribes your audio; the same agent keeps the conversation context and speaks its reply.</p></header>
<section class="card">
<div class="row controls"><button id="record">Start speaking</button><button id="reset">New conversation</button></div>
<div id="status" role="status">Ready. Allow microphone access when prompted. Each recording is limited to 60 seconds.</div>
<div id="turns"></div>
<p class="hint">Local demo. Use synthetic order details only. Audio is sent to Gnani for speech recognition and is not saved by this interface.</p>
</section></main>
<script>
const rec=document.getElementById('record'),reset=document.getElementById('reset'),statusEl=document.getElementById('status'),turns=document.getElementById('turns');
let stream=null,ctx=null,processor=null,source=null,chunks=[],recording=false,timer=null;
function status(s){statusEl.textContent=s}
function wavBlob(parts,rate){
  let n=parts.reduce((a,p)=>a+p.length,0),buf=new ArrayBuffer(44+n*2),v=new DataView(buf),pos=44;
  function str(o,s){for(let i=0;i<s.length;i++)v.setUint8(o+i,s.charCodeAt(i))}
  str(0,'RIFF');v.setUint32(4,36+n*2,true);str(8,'WAVE');str(12,'fmt ');v.setUint32(16,16,true);
  v.setUint16(20,1,true);v.setUint16(22,1,true);v.setUint32(24,rate,true);v.setUint32(28,rate*2,true);
  v.setUint16(32,2,true);v.setUint16(34,16,true);str(36,'data');v.setUint32(40,n*2,true);
  for(const p of parts)for(let i=0;i<p.length;i++){let x=Math.max(-1,Math.min(1,p[i]));v.setInt16(pos,x<0?x*32768:x*32767,true);pos+=2}
  return new Blob([buf],{type:'audio/wav'});
}
async function start(){
  try{
    stream=await navigator.mediaDevices.getUserMedia({audio:true});
    ctx=new (window.AudioContext||window.webkitAudioContext)();source=ctx.createMediaStreamSource(stream);
    processor=ctx.createScriptProcessor(4096,1,1);chunks=[];
    processor.onaudioprocess=e=>{if(recording)chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)))};
    const mute=ctx.createGain();mute.gain.value=0;source.connect(processor);processor.connect(mute);mute.connect(ctx.destination);
    recording=true;rec.textContent='Stop and send';rec.classList.add('recording');status('Recording… speak clearly, then press Stop and send.');
    timer=setTimeout(stopAndSend,60000);
  }catch(e){status('Microphone unavailable: '+e.message)}
}
async function stopAndSend(){
  if(!recording)return;recording=false;clearTimeout(timer);rec.disabled=true;rec.textContent='Processing…';rec.classList.remove('recording');
  const rate=ctx.sampleRate;processor.disconnect();source.disconnect();stream.getTracks().forEach(t=>t.stop());await ctx.close();
  const blob=wavBlob(chunks,rate);chunks=[];
  if(blob.size<1000){rec.disabled=false;rec.textContent='Start speaking';status('I did not receive enough audio. Please try again.');return}
  status('Sending audio to Gnani STT and preparing the spoken reply…');
  try{
    const bytes=new Uint8Array(await blob.arrayBuffer());let binary='';for(let i=0;i<bytes.length;i+=0x8000)binary+=String.fromCharCode.apply(null,bytes.subarray(i,i+0x8000));
    const res=await fetch('/api/turn',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({audio: btoa(binary)})});
    const data=await res.json();if(!res.ok)throw new Error(data.error||'Voice turn failed');
    addTurn(data);status(data.transcript?'Reply ready. Record your follow-up to continue this conversation.':'I did not catch that; please try again.');
  }catch(e){status('Could not complete this voice turn: '+e.message)}
  finally{rec.disabled=false;rec.textContent='Start speaking'}
}
function addTurn(d){
  const box=document.createElement('article');box.className='turn';
  const u=document.createElement('div');u.innerHTML='<div class="label">You said</div>';const ut=document.createElement('div');ut.className='text';ut.textContent=d.transcript||'(No clear speech detected)';u.appendChild(ut);
  const a=document.createElement('div');a.innerHTML='<div class="label">AI support assistant</div>';const at=document.createElement('div');at.className='text';at.textContent=d.reply;a.appendChild(at);
  if(d.sources&&d.sources.length){const so=document.createElement('div');so.className='sources';so.textContent='Retrieved sections: '+d.sources.join(', ');a.appendChild(so)}
  const audio=document.createElement('audio');audio.controls=true;audio.src='data:audio/wav;base64,'+d.reply_audio;a.appendChild(audio);
  box.appendChild(u);box.appendChild(a);turns.prepend(box);audio.play().catch(()=>{});
}
rec.onclick=()=>recording?stopAndSend():start();
reset.onclick=async()=>{if(recording)await stopAndSend();await fetch('/api/reset',{method:'POST'});turns.replaceChildren();status('New conversation. Press Start speaking.');};
</script></body></html>"""

def run_web_ui(agent, language: str = "en-IN", port: int = 8765):
    if not os.getenv("GNANI_API_KEY"):
        raise RuntimeError("GNANI_API_KEY is missing in this terminal. Set it, reopen the terminal, and retry.")
    try:
        from gnani.stt import GnaniSTTClient
        from gnani.tts import AudioConfig, GnaniTTSClient
    except ImportError as exc:
        raise RuntimeError("Install Gnani SDK with: python -m pip install gnani-vachana") from exc
    stt, tts = GnaniSTTClient(), GnaniTTSClient()
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def send_json(self, status, payload):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/":
                self.send_error(404)
                return
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path == "/api/reset":
                with lock:
                    agent.reset()
                self.send_json(200, {"ok": True})
                return
            if self.path != "/api/turn":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 10_000_000:
                    self.send_json(413, {"error": "Recording is empty or too large; keep it under 60 seconds."})
                    return
                payload = json.loads(self.rfile.read(length))
                audio = base64.b64decode(payload.get("audio", ""), validate=True)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(audio)
                    audio_path = Path(f.name)
                try:
                    with wave.open(str(audio_path), "rb") as w:
                        duration = w.getnframes() / max(w.getframerate(), 1)
                    if duration <= 0 or duration > 60:
                        self.send_json(400, {"error": "Record one customer turn of 60 seconds or less."})
                        return
                    transcript = str(stt.transcribe(str(audio_path), language_code=language).get("transcript", "")).strip()
                finally:
                    audio_path.unlink(missing_ok=True)
                with lock:
                    if transcript:
                        result = agent.respond(transcript)
                        reply = result["reply"]
                        sources = result.get("sources", [])
                        decision = result.get("decision", "answer")
                    else:
                        reply = "Sorry, I didn't catch that. Could you say it again?"
                        sources, decision = [], "clarify"
                    wav = tts.synthesize(
                        reply, voice="Kaveri", model="timbre-v2.5", language=language,
                        audio_config=AudioConfig(container="wav", sample_rate=48000),
                    )
                self.send_json(200, {
                    "transcript": transcript, "reply": reply, "sources": sources,
                    "decision": decision, "reply_audio": base64.b64encode(wav).decode("ascii"),
                })
            except Exception as exc:
                # Keep credentials, transcripts and stack details out of the browser response.
                print(f"Voice interface error: {type(exc).__name__}")
                self.send_json(500, {"error": "The voice turn failed. Check Gnani connectivity and the terminal output."})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"VoiceOps audio interface: {url}")
    print("Use the browser microphone. Gnani handles STT/TTS; conversation context stays active until reset.")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping VoiceOps audio interface.")
    finally:
        server.server_close()
