#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.export_generation_for_human_review import CSV_COLUMNS, row_for_record


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_existing_scores(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        out: Dict[str, Dict[str, str]] = {}
        for row in reader:
            row_id = str(row.get("id") or "")
            if row_id:
                out[row_id] = row
        return out


def write_review_csv(path: Path, rows: List[Dict[str, Any]], scores: Dict[str, Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in rows:
            row = row_for_record(record)
            saved = scores.get(str(record.get("id") or ""))
            if saved:
                row["human_precision"] = saved.get("human_precision", row["human_precision"])
                row["human_completeness"] = saved.get("human_completeness", row["human_completeness"])
                row["human_format"] = saved.get("human_format", row["human_format"])
                row["human_comment"] = saved.get("human_comment", row["human_comment"])
            writer.writerow(row)


def validate_score(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    if text == "":
        return "-1"
    try:
        score = int(text)
    except ValueError as exc:
        raise ValueError(f"score must be -1 or integer 1..5, got {value!r}") from exc
    if score == -1 or 1 <= score <= 5:
        return str(score)
    raise ValueError(f"score must be -1 or integer 1..5, got {score}")


def generation_prompt_for_record(record: Dict[str, Any]) -> str:
    variant = (record.get("generation_metadata") or {}).get("prompt_variant") or "strict_citations"
    query = str(record.get("query") or "")
    retrieved = record.get("retrieved") or []
    context_parts = []
    for chunk in retrieved:
        text = str(chunk.get("text") or "").strip()
        if text:
            context_parts.append(f"[{chunk.get('rank')}] {text}")
    context = "\n\n---\n\n".join(context_parts) or "(контекст пуст)"

    if variant == "strict_citations":
        system = """Ты налоговый консультант.
Используй только предоставленный контекст.
Не используй внешние знания.
Не отвечай по памяти и не добавляй факты из собственных знаний.
Если в контексте нет ответа, скажи: "В предоставленных документах нет достаточной информации для ответа."
Если в контексте нет фактов, подтверждающих налоговый или правовой вывод, обязательно откажись отвечать этой фразой.
Ответ должен быть структурирован:
1. Краткий вывод
2. Обоснование по документам
3. Источники
Не выдумывай реквизиты, ссылки, сроки, суммы штрафов и номера статей."""
    elif variant == "default":
        system = """Ответь по контексту кратко и точно.
Не используй внешние знания.
Не отвечай по памяти и не добавляй факты из собственных знаний.
Если в контексте нет фактов для ответа, скажи: "В предоставленных документах нет достаточной информации для ответа." """
    else:
        system = """Если контекст не содержит ответа, обязательно откажись отвечать.
Не делай предположений.
Используй только предоставленный контекст.
Не используй внешние знания и не отвечай по памяти.
Если в контексте нет фактов, подтверждающих налоговый или правовой вывод, скажи: "В предоставленных документах нет достаточной информации для ответа." """

    return f"System prompt:\n{system}\n\nUser prompt:\nВопрос:\n{query}\n\nКонтекст:\n{context}"


INDEX_HTML = r"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Generation Review</title>
  <style>
    :root { color-scheme: light; --line:#d7dde5; --muted:#64748b; --bg:#f7f8fa; --panel:#fff; --accent:#155eef; }
    * { box-sizing: border-box; }
    body { margin:0; font:14px/1.45 system-ui,-apple-system,Segoe UI,Arial,sans-serif; background:var(--bg); color:#111827; }
    header { height:56px; display:flex; align-items:center; gap:16px; padding:0 18px; border-bottom:1px solid var(--line); background:#fff; position:sticky; top:0; z-index:5; }
    header b { font-size:16px; }
    main { display:grid; grid-template-columns: 340px 1fr; min-height:calc(100vh - 56px); }
    aside { border-right:1px solid var(--line); background:#fff; overflow:auto; height:calc(100vh - 56px); }
    .content { padding:18px; overflow:auto; height:calc(100vh - 56px); }
    .toolbar { display:flex; gap:8px; align-items:center; margin-left:auto; }
    input, select, textarea, button { font:inherit; }
    input[type=search] { width:260px; padding:8px 10px; border:1px solid var(--line); border-radius:6px; }
    button { border:1px solid var(--line); background:#fff; border-radius:6px; padding:8px 10px; cursor:pointer; }
    button.primary { background:var(--accent); color:#fff; border-color:var(--accent); }
    .list-item { padding:10px 12px; border-bottom:1px solid #edf0f4; cursor:pointer; }
    .list-item.active { background:#eef4ff; border-left:3px solid var(--accent); padding-left:9px; }
    .list-id { font-weight:650; }
    .list-q { color:#334155; margin-top:3px; max-height:40px; overflow:hidden; }
    .scores { color:var(--muted); font-size:12px; margin-top:4px; }
    .grid { display:grid; grid-template-columns: 1fr 340px; gap:16px; align-items:start; }
    section { background:var(--panel); border:1px solid var(--line); border-radius:8px; margin-bottom:14px; }
    section h2 { margin:0; padding:12px 14px; border-bottom:1px solid var(--line); font-size:15px; }
    .box { padding:14px; white-space:pre-wrap; }
    .muted { color:var(--muted); }
    details { border-top:1px solid #edf0f4; }
    details:first-child { border-top:0; }
    summary { cursor:pointer; padding:12px 14px; display:flex; gap:10px; align-items:center; }
    .chunk-text { padding:0 14px 14px; white-space:pre-wrap; }
    .tag { border-radius:999px; padding:2px 8px; font-size:12px; border:1px solid var(--line); }
    .gold { background:#ecfdf3; color:#067647; border-color:#abefc6; }
    .hard { background:#fff1f3; color:#c01048; border-color:#fecdd6; }
    .unknown { background:#f2f4f7; color:#475467; }
    .rank { font-weight:650; min-width:42px; }
    .meta { color:var(--muted); font-size:12px; margin-left:auto; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .score-row { display:grid; grid-template-columns:1fr 120px; gap:10px; align-items:center; padding:10px 14px; border-bottom:1px solid #edf0f4; }
    .score-row label { font-weight:600; }
    .score-row select { padding:7px; border:1px solid var(--line); border-radius:6px; }
    textarea { width:100%; min-height:110px; resize:vertical; border:1px solid var(--line); border-radius:6px; padding:10px; }
    .savebar { padding:14px; display:flex; gap:8px; align-items:center; }
    .status { color:var(--muted); font-size:12px; }
    pre { margin:0; white-space:pre-wrap; font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
  </style>
</head>
<body>
  <header>
    <b>Generation Review</b>
    <span id="file" class="muted"></span>
    <div class="toolbar">
      <input id="search" type="search" placeholder="id или текст вопроса">
      <button id="prev">Назад</button>
      <button id="next">Дальше</button>
    </div>
  </header>
  <main>
    <aside id="list"></aside>
    <div class="content">
      <div class="grid">
        <div>
          <section><h2 id="title">Вопрос</h2><div id="query" class="box"></div></section>
          <section><h2>Ответ модели</h2><div id="answer" class="box"></div></section>
          <section><h2>Найденные чанки</h2><div id="chunks"></div></section>
          <section><h2>Промпт генерации</h2><div class="box"><pre id="prompt"></pre></div></section>
        </div>
        <div>
          <section>
            <h2>Оценка человека</h2>
            <div class="score-row"><label>Precision</label><select id="precision"></select></div>
            <div class="score-row"><label>Completeness</label><select id="completeness"></select></div>
            <div class="score-row"><label>Format</label><select id="format"></select></div>
            <div class="box"><textarea id="comment" placeholder="Комментарий"></textarea></div>
            <div class="savebar">
              <button class="primary" id="save">Сохранить</button>
              <span id="status" class="status"></span>
            </div>
          </section>
          <section><h2>Summary</h2><div id="summary" class="box"></div></section>
          <section><h2>Шкала</h2><div class="box">-1 — не размечено
1 — плохо
2 — много проблем
3 — частично нормально
4 — хорошо, minor issues
5 — отлично</div></section>
        </div>
      </div>
    </div>
  </main>
<script>
let records = [];
let filtered = [];
let current = 0;

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function fillSelect(id) {
  const el = document.getElementById(id);
  for (const v of [-1,1,2,3,4,5]) {
    const opt = document.createElement('option');
    opt.value = String(v);
    opt.textContent = String(v);
    el.appendChild(opt);
  }
}

function scoreText(r) {
  const h = r.human || {};
  return `P:${h.precision ?? -1} C:${h.completeness ?? -1} F:${h.format ?? -1}`;
}

function applyFilter() {
  const q = document.getElementById('search').value.toLowerCase();
  filtered = records.filter(r => !q || String(r.id).toLowerCase().includes(q) || String(r.query).toLowerCase().includes(q));
  current = Math.min(current, Math.max(0, filtered.length - 1));
  renderList();
  renderCurrent();
}

function renderList() {
  const list = document.getElementById('list');
  list.innerHTML = filtered.map((r, i) => `<div class="list-item ${i===current?'active':''}" onclick="current=${i};renderList();renderCurrent()">
    <div class="list-id">${esc(r.id)}</div>
    <div class="list-q">${esc(r.query)}</div>
    <div class="scores">${esc(scoreText(r))}</div>
  </div>`).join('');
}

function tagClass(label) {
  if (label === 'gold_relevant') return 'gold';
  if (label === 'hard_negative') return 'hard';
  return 'unknown';
}

function renderCurrent() {
  if (!filtered.length) return;
  const r = filtered[current];
  document.getElementById('title').textContent = `Вопрос ${r.id} (${current+1}/${filtered.length})`;
  document.getElementById('query').textContent = r.query || '';
  document.getElementById('answer').textContent = r.answer || '';
  document.getElementById('prompt').textContent = r.generation_prompt || '';
  const h = r.human || {};
  document.getElementById('precision').value = String(h.precision ?? -1);
  document.getElementById('completeness').value = String(h.completeness ?? -1);
  document.getElementById('format').value = String(h.format ?? -1);
  document.getElementById('comment').value = h.comment || '';
  const s = r.retrieval_gold_summary || {};
  document.getElementById('summary').textContent =
    `has_gold_relevant_context: ${s.has_gold_relevant_context}\n` +
    `gold ranks: ${(s.gold_relevant_ranks || []).join(', ')}\n` +
    `gold: ${s.gold_relevant_retrieved_count ?? '-'}\n` +
    `hard negative: ${s.hard_negative_retrieved_count ?? '-'}\n` +
    `unknown: ${s.unknown_retrieved_count ?? '-'}`;
  document.getElementById('chunks').innerHTML = (r.retrieved || []).map(c => `
    <details ${c.relevance_label === 'gold_relevant' ? 'open' : ''}>
      <summary>
        <span class="rank">#${esc(c.rank)}</span>
        <span class="tag ${tagClass(c.relevance_label)}">${esc(c.relevance_label || 'unknown')}</span>
        <span class="tag">${esc(c.relevance_match_level || 'none')}</span>
        <span class="meta">${esc(c.source_code)} / ${esc(c.external_id)} / chunk_i=${esc(c.chunk_i)} / score=${esc(c.score)}</span>
      </summary>
      <div class="chunk-text">${esc(c.text)}</div>
    </details>
  `).join('');
}

async function saveCurrent() {
  const r = filtered[current];
  const payload = {
    id: r.id,
    human_precision: document.getElementById('precision').value,
    human_completeness: document.getElementById('completeness').value,
    human_format: document.getElementById('format').value,
    human_comment: document.getElementById('comment').value
  };
  const res = await fetch('/api/save', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
  if (!res.ok) {
    document.getElementById('status').textContent = await res.text();
    return;
  }
  r.human = {
    precision: Number(payload.human_precision),
    completeness: Number(payload.human_completeness),
    format: Number(payload.human_format),
    comment: payload.human_comment
  };
  document.getElementById('status').textContent = 'Сохранено';
  renderList();
}

async function init() {
  fillSelect('precision'); fillSelect('completeness'); fillSelect('format');
  const res = await fetch('/api/records');
  const data = await res.json();
  records = data.records;
  filtered = records;
  document.getElementById('file').textContent = data.input + ' -> ' + data.output;
  document.getElementById('search').addEventListener('input', applyFilter);
  document.getElementById('save').addEventListener('click', saveCurrent);
  document.getElementById('next').addEventListener('click', () => { current=Math.min(filtered.length-1,current+1); renderList(); renderCurrent(); });
  document.getElementById('prev').addEventListener('click', () => { current=Math.max(0,current-1); renderList(); renderCurrent(); });
  renderList(); renderCurrent();
}
init();
</script>
</body>
</html>
"""


class ReviewServer(BaseHTTPRequestHandler):
    rows: List[Dict[str, Any]] = []
    input_path: Path
    output_path: Path
    scores: Dict[str, Dict[str, str]] = {}

    def send_json(self, data: Any, status: int = 200) -> None:
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def send_text(self, text: str, status: int = 200, content_type: str = "text/plain; charset=utf-8") -> None:
        raw = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self.send_text(INDEX_HTML, content_type="text/html; charset=utf-8")
            return
        if path == "/api/records":
            records = []
            for record in self.rows:
                row = dict(record)
                saved = self.scores.get(str(row.get("id") or ""))
                if saved:
                    row["human"] = {
                        "precision": int(saved.get("human_precision") or -1),
                        "completeness": int(saved.get("human_completeness") or -1),
                        "format": int(saved.get("human_format") or -1),
                        "comment": saved.get("human_comment") or "",
                    }
                row["generation_prompt"] = generation_prompt_for_record(row)
                records.append(row)
            self.send_json({"input": str(self.input_path), "output": str(self.output_path), "records": records})
            return
        self.send_text("not found", status=404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/save":
            self.send_text("not found", status=404)
            return
        length = int(self.headers.get("Content-Length") or "0")
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            row_id = str(data.get("id") or "")
            if not row_id:
                raise ValueError("missing id")
            self.scores[row_id] = {
                "id": row_id,
                "human_precision": validate_score(data.get("human_precision")),
                "human_completeness": validate_score(data.get("human_completeness")),
                "human_format": validate_score(data.get("human_format")),
                "human_comment": str(data.get("human_comment") or ""),
            }
            write_review_csv(self.output_path, self.rows, self.scores)
        except Exception as exc:
            self.send_text(str(exc), status=400)
            return
        self.send_json({"ok": True})


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Local web UI for human generation review.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True, help="Review CSV output compatible with import_human_generation_scores.py")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true", help="Open browser automatically")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.out)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    rows = load_jsonl(input_path)
    scores = load_existing_scores(output_path)
    write_review_csv(output_path, rows, scores)

    ReviewServer.rows = rows
    ReviewServer.input_path = input_path
    ReviewServer.output_path = output_path
    ReviewServer.scores = scores

    server = ThreadingHTTPServer((args.host, args.port), ReviewServer)
    url = f"http://{args.host}:{args.port}"
    print(f"Review UI: {url}")
    print(f"Input: {input_path}")
    print(f"Saving CSV: {output_path}")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped")


if __name__ == "__main__":
    main()
