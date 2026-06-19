import sqlite3
import os
import uuid
import json
import re
import math
import urllib.request
import urllib.parse
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, Response, stream_with_context
from openai import OpenAI
from collections import deque

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "paswan-ai-secret-2024")

# ==================================================
# GROQ CLIENT — Fast + Free
# .env mein: GROQ_API_KEY=gsk_xxxxx
# ==================================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY
) if GROQ_API_KEY else None

TEXT_MODEL   = "llama-3.3-70b-versatile"
VISION_MODEL = "llama-3.3-70b-versatile"

# ==================================================
# AI MODES — Special System Prompts
# ==================================================
AI_MODES = {
    "default": """You are paswan.ai, an advanced AI assistant created by Sourab Paswan.
Be smart, friendly, helpful, and professional like ChatGPT.
- Think step by step
- Give complete working code always
- Use markdown formatting
- Remember conversation context
- Your name is paswan.ai, created by Sourab Paswan
- NEVER say you are Llama or any other model

MATH RULES:
- Inline math: $x^2 + 1$
- Block math: $$\\int \\frac{x^2+1}{x+1}dx$$
- NEVER use \\[ \\] or \\( \\) style
- Always use $ or $$ signs""",

    "coding": """You are paswan.ai in CODING COPILOT MODE.
- You are an expert programmer in Python, JavaScript, HTML, CSS, React, Flask, and more
- Always write complete, production-ready code
- Add comments to explain complex parts
- Suggest best practices and optimizations
- Debug errors step by step
- Format all code in proper markdown code blocks with language specified
- Your name is paswan.ai""",

    "math": """You are paswan.ai in MATH TUTOR MODE.
- You are an expert mathematician
- Solve ALL math problems step by step
- Show every single step clearly
- Use $ for inline math: $x^2$
- Use $$ for block math: $$\\frac{d}{dx}$$
- Explain the concept behind each step
- Verify your answer at the end
- Your name is paswan.ai""",

    "research": """You are paswan.ai in DEEP RESEARCH MODE.
- Provide comprehensive, well-structured research
- Use headings, subheadings, bullet points
- Cite multiple perspectives on any topic
- Distinguish between facts and opinions
- Provide historical context when relevant
- Give a balanced, academic-level response
- Your name is paswan.ai""",

    "tutor": """You are paswan.ai in STUDY TUTOR MODE.
- You are a patient, encouraging teacher
- Break down complex topics into simple parts
- Use examples, analogies, and stories
- Ask questions to check understanding
- Give practice problems when appropriate
- Adapt your teaching style to the student
- Your name is paswan.ai""",

    "debate": """You are paswan.ai in DEBATE MODE.
- Present multiple sides of every argument
- Be intellectually rigorous and fair
- Use logical reasoning and evidence
- Point out logical fallacies when present
- Conclude with a balanced summary
- Your name is paswan.ai""",

    "creative": """You are paswan.ai in CREATIVE MODE.
- You are a creative writer and storyteller
- Write engaging, vivid, imaginative content
- Use metaphors, similes, and literary devices
- Adapt tone: formal, casual, poetic, humorous
- Your name is paswan.ai""",

    "planner": """You are paswan.ai in PROJECT PLANNER MODE.
- Help plan projects systematically
- Create timelines, milestones, tasks
- Identify risks and mitigation strategies
- Suggest tools and resources
- Format output as structured plans with checklists
- Your name is paswan.ai"""
}

# ==================================================
# DATABASE
# ==================================================
def get_db():
    conn = sqlite3.connect("memory.db", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS memory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT, role TEXT, content TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    c.execute("""CREATE TABLE IF NOT EXISTS chats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT, user_uid TEXT,
        user_message TEXT, ai_reply TEXT, mode TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    c.execute("""CREATE TABLE IF NOT EXISTS memory_summary (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT UNIQUE, summary TEXT,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    c.execute("""CREATE TABLE IF NOT EXISTS notes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT, title TEXT, content TEXT,
        pinned INTEGER DEFAULT 0,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    c.execute("""CREATE TABLE IF NOT EXISTS bookmarks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT, message TEXT, reply TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit()
    conn.close()

init_db()

# ==================================================
# MEMORY
# ==================================================
session_memories = {}
MAX_MEMORY = 20
SUMMARY_THRESHOLD = 16

def get_memory(sid):
    if sid not in session_memories:
        conn = get_db()
        c = conn.cursor()
        c.execute("SELECT role, content FROM memory WHERE session_id=? ORDER BY id DESC LIMIT ?",
                  (sid, MAX_MEMORY))
        rows = c.fetchall()
        conn.close()
        session_memories[sid] = deque(
            [{"role": r["role"], "content": r["content"]} for r in reversed(rows)],
            maxlen=MAX_MEMORY
        )
    return session_memories[sid]

def save_memory(sid, role, content):
    get_memory(sid).append({"role": role, "content": content})
    conn = get_db()
    c = conn.cursor()
    c.execute("INSERT INTO memory (session_id, role, content) VALUES (?,?,?)",
              (sid, role, content))
    conn.commit()
    conn.close()

def get_summary(sid):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT summary FROM memory_summary WHERE session_id=?", (sid,))
    row = c.fetchone()
    conn.close()
    return row["summary"] if row else None

def save_summary(sid, summary):
    conn = get_db()
    c = conn.cursor()
    c.execute("""INSERT INTO memory_summary (session_id, summary) VALUES (?,?)
                 ON CONFLICT(session_id) DO UPDATE SET summary=excluded.summary,
                 updated_at=CURRENT_TIMESTAMP""", (sid, summary))
    conn.commit()
    conn.close()

def auto_summarize(sid):
    if not client:
        return
    mem = get_memory(sid)
    if len(mem) < SUMMARY_THRESHOLD:
        return
    try:
        text = "\n".join([f"{m['role']}: {m['content']}" for m in list(mem)])
        resp = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": "Summarize this conversation in 3-5 bullet points. Include: user name, preferences, key topics discussed."},
                {"role": "user", "content": text}
            ],
            max_tokens=250, temperature=0.3
        )
        save_summary(sid, resp.choices[0].message.content)
    except Exception as e:
        print("Summary error:", e)

def save_chat_db(sid, uid, user_msg, reply, mode="default"):
    conn = get_db()
    c = conn.cursor()
    c.execute("INSERT INTO chats (session_id, user_uid, user_message, ai_reply, mode) VALUES (?,?,?,?,?)",
              (sid, uid, user_msg, reply, mode))
    conn.commit()
    conn.close()

# ==================================================
# WEB SEARCH (DuckDuckGo — No API key needed!)
# ==================================================
def web_search(query, max_results=5):
    try:
        encoded = urllib.parse.quote(query)
        url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1&skip_disambig=1"
        req = urllib.request.Request(url, headers={"User-Agent": "paswan.ai/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())

        results = []

        # Abstract result
        if data.get("Abstract"):
            results.append({
                "title": data.get("Heading", "Result"),
                "snippet": data["Abstract"],
                "url": data.get("AbstractURL", "")
            })

        # Related topics
        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append({
                    "title": topic.get("FirstURL", "").split("/")[-1].replace("_", " "),
                    "snippet": topic["Text"],
                    "url": topic.get("FirstURL", "")
                })

        return results[:max_results]
    except Exception as e:
        print("Search error:", e)
        return []

def format_search_results(results, query):
    if not results:
        return f"No search results found for: {query}"
    text = f"**Web Search Results for: '{query}'**\n\n"
    for i, r in enumerate(results, 1):
        text += f"**{i}. {r['title']}**\n"
        text += f"{r['snippet']}\n"
        if r.get('url'):
            text += f"Source: {r['url']}\n"
        text += "\n"
    return text

# ==================================================
# YOUTUBE SUMMARIZER
# ==================================================
def extract_youtube_id(url):
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com/embed/([a-zA-Z0-9_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_youtube_info(video_id):
    try:
        url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        req = urllib.request.Request(url, headers={"User-Agent": "paswan.ai/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            return {
                "title": data.get("title", "Unknown"),
                "author": data.get("author_name", "Unknown"),
                "thumbnail": data.get("thumbnail_url", "")
            }
    except:
        return None

# ==================================================
# AI TOOLS
# ==================================================
def tool_calc(expr):
    try:
        safe = re.sub(r'[^0-9+\-*/().%\s]', '', expr)
        result = eval(safe, {"__builtins__": {}, "math": math})
        return f"Result: {expr} = {result}"
    except Exception as e:
        return f"Error: {e}"

TOOLS = {"calculator": tool_calc}

def process_tools(text):
    for name, inp in re.findall(r'\[TOOL:(\w+)\](.*?)\[/TOOL\]', text, re.DOTALL):
        if name in TOOLS:
            text = text.replace(f"[TOOL:{name}]{inp}[/TOOL]",
                                f"\n**Result:** {TOOLS[name](inp.strip())}\n")
    return text

# ==================================================
# AGENT
# ==================================================
def run_agent(user_message, sid, sys_prompt):
    if not client:
        return "❌ AI not configured."
    agent_sys = sys_prompt + "\nAGENT: Think step by step. End with 'Final Answer:'"
    mem = list(get_memory(sid))[-8:]
    messages = [{"role": "system", "content": agent_sys}] + mem + \
               [{"role": "user", "content": user_message}]
    try:
        for _ in range(4):
            resp = client.chat.completions.create(
                model=TEXT_MODEL, messages=messages,
                temperature=0.5, max_tokens=1500
            )
            out = process_tools(resp.choices[0].message.content)
            if "Final Answer:" in out:
                return out.split("Final Answer:", 1)[1].strip()
            messages.append({"role": "assistant", "content": out})
            messages.append({"role": "user", "content": "Continue to final answer."})
    except Exception as e:
        return f"❌ Agent error: {e}"
    return out or "Could not complete."

# ==================================================
# ROUTES
# ==================================================
@app.route("/")
def home():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok", "ai": bool(client), "model": TEXT_MODEL})

# ==================================================
# CHAT
# ==================================================
@app.route("/chat", methods=["POST"])
def chat():
    if not client:
        return jsonify({"reply": "❌ AI not configured. Set GROQ_API_KEY in .env file."}), 503

    data       = request.get_json(force=True) or {}
    user_msg   = (data.get("message") or "").strip()
    uid        = data.get("uid", "anonymous")
    img_data   = data.get("image_data") or None
    use_agent  = bool(data.get("agent", False))
    do_stream  = bool(data.get("stream", True))
    mode       = data.get("mode", "default")
    web_search_on = bool(data.get("web_search", False))

    if not user_msg and not img_data:
        return jsonify({"reply": "Please send a message."})

    sid = session.get("session_id", str(uuid.uuid4()))
    auto_summarize(sid)

    # System prompt based on mode
    sys_prompt = AI_MODES.get(mode, AI_MODES["default"])
    summary = get_summary(sid)
    if summary:
        sys_prompt += f"\n\nCONVERSATION CONTEXT:\n{summary}"

    # Web search
    search_context = ""
    if web_search_on and user_msg:
        results = web_search(user_msg)
        if results:
            search_context = format_search_results(results, user_msg)
            sys_prompt += f"\n\nWEB SEARCH RESULTS (use these to answer):\n{search_context}"

    # YouTube summarize
    yt_id = extract_youtube_id(user_msg) if user_msg else None
    if yt_id:
        info = get_youtube_info(yt_id)
        if info:
            sys_prompt += f"\n\nYOUTUBE VIDEO INFO:\nTitle: {info['title']}\nChannel: {info['author']}\nPlease summarize what this video is likely about based on the title and provide key insights."

    save_memory(sid, "user", user_msg or "[Image]")

    # Agent mode
    if use_agent and not img_data:
        reply = run_agent(user_msg, sid, sys_prompt)
        save_memory(sid, "assistant", reply)
        save_chat_db(sid, uid, user_msg, reply, mode)
        return jsonify({"reply": reply})

    # Build messages
    messages = [{"role": "system", "content": sys_prompt}]
    messages += list(get_memory(sid))[-10:]

    # Image
    is_vision = False
    if img_data:
        if not img_data.startswith("data:image"):
            return jsonify({"reply": "❌ Invalid image format."})
        is_vision = True
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": img_data, "detail": "high"}},
                {"type": "text", "text": user_msg if user_msg else "Analyze this image carefully. If it has math, equations, or code — solve step by step."}
            ]
        })

    max_tok = 512 if is_vision else 2048

    # STREAMING
    if do_stream:
        def generate():
            full = []
            try:
                resp = client.chat.completions.create(
                    model=TEXT_MODEL,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=max_tok,
                    stream=True,
                )
                for chunk in resp:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    token = delta.content if delta else None
                    if token:
                        full.append(token)
                        yield f"data: {json.dumps({'token': token})}\n\n"

                reply = "".join(full).strip()
                if reply:
                    save_memory(sid, "assistant", reply)
                    save_chat_db(sid, uid, user_msg or "[Image]", reply, mode)
                yield f"data: {json.dumps({'done': True})}\n\n"

            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )

    # NON-STREAMING
    try:
        resp  = client.chat.completions.create(
            model=TEXT_MODEL, messages=messages,
            temperature=0.7, max_tokens=max_tok
        )
        reply = resp.choices[0].message.content.strip()
        save_memory(sid, "assistant", reply)
        save_chat_db(sid, uid, user_msg or "[Image]", reply, mode)
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"reply": f"❌ Error: {e}"}), 500

# ==================================================
# NOTES
# ==================================================
@app.route("/notes", methods=["GET"])
def get_notes():
    sid = session.get("session_id", "")
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM notes WHERE session_id=? ORDER BY pinned DESC, timestamp DESC",
        (sid,)
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route("/notes", methods=["POST"])
def save_note():
    sid  = session.get("session_id", "")
    data = request.get_json() or {}
    conn = get_db()
    conn.execute(
        "INSERT INTO notes (session_id, title, content) VALUES (?,?,?)",
        (sid, data.get("title", "Note"), data.get("content", ""))
    )
    conn.commit()
    conn.close()
    return jsonify({"status": "saved"})

@app.route("/notes/<int:note_id>", methods=["DELETE"])
def delete_note(note_id):
    conn = get_db()
    conn.execute("DELETE FROM notes WHERE id=?", (note_id,))
    conn.commit()
    conn.close()
    return jsonify({"status": "deleted"})

@app.route("/notes/<int:note_id>/pin", methods=["POST"])
def pin_note(note_id):
    conn = get_db()
    conn.execute("UPDATE notes SET pinned = 1 - pinned WHERE id=?", (note_id,))
    conn.commit()
    conn.close()
    return jsonify({"status": "toggled"})

# ==================================================
# BOOKMARKS
# ==================================================
@app.route("/bookmarks", methods=["GET"])
def get_bookmarks():
    sid = session.get("session_id", "")
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM bookmarks WHERE session_id=? ORDER BY timestamp DESC",
        (sid,)
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route("/bookmarks", methods=["POST"])
def add_bookmark():
    sid  = session.get("session_id", "")
    data = request.get_json() or {}
    conn = get_db()
    conn.execute(
        "INSERT INTO bookmarks (session_id, message, reply) VALUES (?,?,?)",
        (sid, data.get("message", ""), data.get("reply", ""))
    )
    conn.commit()
    conn.close()
    return jsonify({"status": "bookmarked"})

@app.route("/bookmarks/<int:bid>", methods=["DELETE"])
def delete_bookmark(bid):
    conn = get_db()
    conn.execute("DELETE FROM bookmarks WHERE id=?", (bid,))
    conn.commit()
    conn.close()
    return jsonify({"status": "deleted"})

# ==================================================
# OTHER ROUTES
# ==================================================
@app.route("/memory_summary")
def memory_summary_api():
    sid = session.get("session_id", "")
    return jsonify({"summary": get_summary(sid) or "No summary yet."})

@app.route("/new_session", methods=["POST"])
def new_session():
    session["session_id"] = str(uuid.uuid4())
    return jsonify({"status": "ok"})

@app.route("/history")
def history():
    sid = session.get("session_id", "")
    conn = get_db()
    rows = conn.execute(
        "SELECT id, user_message, ai_reply FROM chats WHERE session_id=? ORDER BY id DESC LIMIT 20",
        (sid,)
    ).fetchall()
    conn.close()
    return jsonify([{"id": r["id"], "message": r["user_message"], "reply": r["ai_reply"]} for r in rows])

@app.route("/clear_memory", methods=["POST"])
def clear_memory():
    sid = session.get("session_id", "")
    if sid in session_memories:
        del session_memories[sid]
    conn = get_db()
    c = conn.cursor()
    c.execute("DELETE FROM memory WHERE session_id=?", (sid,))
    c.execute("DELETE FROM memory_summary WHERE session_id=?", (sid,))
    conn.commit()
    conn.close()
    session["session_id"] = str(uuid.uuid4())
    return jsonify({"status": "cleared"})

# ==================================================
# RUN
# ==================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)