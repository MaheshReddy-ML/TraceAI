"""Local read-only dashboard for stored experiments."""

from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

from traceai.errors import StorageError
from traceai.storage import SQLiteRepository

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TraceAI · behavior observatory</title><style>
:root{color-scheme:light;--ink:#17312b;--muted:#5d7069;--paper:#f4f2e9;--panel:#fffef8;--line:#cad5cc;--accent:#136b55;--warm:#c66a34}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.5 system-ui,-apple-system,sans-serif}header{background:#17312b;color:#f6f4e8;padding:1.5rem max(1rem,calc((100vw - 1200px)/2))}header h1{font:700 clamp(2rem,5vw,3rem)/1.1 Georgia,serif;margin:.15rem 0}header p{margin:.3rem 0;color:#c9d9d2}.eyebrow{text-transform:uppercase;letter-spacing:.17em;font-size:.7rem;font-weight:700;color:#8cd5b9}main{max-width:1200px;margin:auto;padding:1.5rem 1rem 4rem}.layout{display:grid;grid-template-columns:230px minmax(0,1fr);gap:1.25rem}.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:1.25rem;box-shadow:0 2px 0 #dde5da}h2{font:700 1.35rem Georgia,serif;margin:0 0 .8rem}h3{font-size:1rem;margin:1.3rem 0 .5rem}.small{color:var(--muted);font-size:.86rem}button,select{font:inherit}button{border:1px solid var(--line);background:#fff;color:var(--ink);border-radius:7px;padding:.5rem .65rem;cursor:pointer;text-align:left}button:hover,button:focus-visible{border-color:var(--accent);outline:2px solid transparent;background:#eaf4ed}button.active{background:#dcf0e3;border-color:var(--accent)}.study-list{display:grid;gap:.5rem}.study-list button{width:100%;overflow-wrap:anywhere}.row{display:flex;gap:1rem;flex-wrap:wrap;align-items:center}.tag{display:inline-block;background:#e4eee5;border-radius:20px;padding:.16rem .55rem;font-size:.78rem;font-weight:650}select{padding:.45rem;border:1px solid var(--line);border-radius:6px;background:#fff;max-width:100%}.chart-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:#fff;padding:.5rem}svg{display:block;width:100%;min-width:460px;height:260px}.axis{stroke:#b9c9bd;stroke-width:1}.plot{fill:none;stroke:var(--accent);stroke-width:3}.dot{fill:var(--accent)}.chart-text{font:12px system-ui;fill:#4f655a}.table-scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;min-width:570px}th,td{text-align:left;border-bottom:1px solid var(--line);padding:.55rem .6rem;font-size:.88rem}th{color:var(--muted);font-size:.75rem;text-transform:uppercase;letter-spacing:.08em}details{border-top:1px solid var(--line);padding:.7rem 0}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f5ef;padding:.75rem;border-radius:6px;font-size:.8rem}.empty{padding:2rem;color:var(--muted)}.notice{background:#fff0dd;border-left:3px solid var(--warm);padding:.6rem .8rem;margin:1rem 0}ul{padding-left:1.3rem}main,.layout,.layout>div,.panel,.study-list{min-width:0;max-width:100%}.study-list{grid-template-columns:minmax(0,1fr)}.small,.notice,.study-list button{overflow-wrap:anywhere}@media(max-width:700px){.layout{display:block}.layout>aside{margin-bottom:1rem}header{padding:1.4rem 1rem}}
</style></head><body><header><div class="eyebrow">Local research workspace</div><h1>TraceAI</h1><p>Trace how AI models learn, behave, and change.</p></header>
<main><div class="layout"><aside class="panel"><h2>Experiments</h2><div id="studies" class="study-list" aria-label="Experiments">Loading…</div></aside><div><section class="panel"><h2 id="title">Choose an experiment</h2><div id="meta" class="small"></div><div class="notice"><strong>Observability: outputs only.</strong> Scores describe failures on fixed prompt cases. They do not identify a model's intentions or overall safety.</div><div class="row"><label for="probe">Probe</label><select id="probe" aria-label="Select probe"></select></div><h3>Checkpoint trajectory</h3><div class="chart-wrap"><svg id="chart" role="img" aria-label="Failure rate by checkpoint"></svg></div><div id="changes"></div><h3>Measurements</h3><div class="table-scroll"><table><thead><tr><th>Checkpoint</th><th>Probe</th><th>Failure rate</th><th>Cases</th><th>Approx. interval</th></tr></thead><tbody id="measurements"></tbody></table></div><h3>Raw evidence</h3><div id="evidence"></div><h3>Limitations</h3><ul id="limits"></ul></section></div></div></main>
<script>
const $=id=>document.getElementById(id);let current=null,selected=null;
const svg=(tag,attrs)=>{const el=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [key,value] of Object.entries(attrs))el.setAttribute(key,value);return el};
function clear(el){while(el.firstChild)el.removeChild(el.firstChild)}
function addText(parent,tag,value){const el=document.createElement(tag);el.textContent=value;parent.append(el);return el}
function draw(){const chart=$('chart');clear(chart);chart.setAttribute('viewBox','0 0 600 260');const data=current.observations.filter(x=>x.probe===selected);const order=current.config.checkpoints.map(x=>x.id);data.sort((a,b)=>order.indexOf(a.checkpoint)-order.indexOf(b.checkpoint));for(let i=0;i<=4;i++){const y=220-i*50;chart.append(svg('line',{x1:60,y1:y,x2:570,y2:y,class:'axis'}));const label=svg('text',{x:12,y:y+4,class:'chart-text'});label.textContent=(i*.25).toFixed(2);chart.append(label)}if(!data.length)return;const points=data.map((x,i)=>({x:60+(data.length===1?255:i*510/(data.length-1)),y:220-x.score*200,item:x}));chart.append(svg('polyline',{points:points.map(p=>`${p.x},${p.y}`).join(' '),class:'plot'}));points.forEach(p=>{chart.append(svg('circle',{cx:p.x,cy:p.y,r:5,class:'dot'}));const t=svg('text',{x:p.x,y:245,'text-anchor':'middle',class:'chart-text'});t.textContent=p.item.checkpoint;chart.append(t)});chart.setAttribute('aria-label',`${selected} failure rates: `+data.map(x=>`${x.checkpoint}: ${x.score.toFixed(2)}`).join(', '))}
function render(){if(!current)return;$('title').textContent=current.config.name;$('meta').textContent=`${current.experiment_id} · ${current.status} · ${current.config.target.runtime}/${current.config.target.model} · ${current.evidence.length} evidence records`;
const probes=[...new Set(current.observations.map(x=>x.probe))];if(!probes.includes(selected))selected=probes[0];clear($('probe'));for(const name of probes){const option=document.createElement('option');option.value=name;option.textContent=name;$('probe').append(option)}$('probe').value=selected;draw();clear($('measurements'));for(const x of current.observations.filter(x=>x.probe===selected)){const tr=document.createElement('tr');for(const value of [x.checkpoint,x.probe,x.score.toFixed(3),x.sample_count,`${x.ci_low.toFixed(2)}–${x.ci_high.toFixed(2)}`])addText(tr,'td',String(value));$('measurements').append(tr)}clear($('changes'));const changes=current.changes.filter(x=>x.probe===selected);addText($('changes'),'p',changes.length?changes.map(x=>`${x.from_checkpoint} → ${x.to_checkpoint}: ${x.delta>=0?'+':''}${x.delta.toFixed(2)} (${x.classification})`).join(' · '):'No adjacent score change reached the descriptive threshold.');clear($('evidence'));for(const x of current.evidence.filter(x=>x.probe===selected)){const detail=document.createElement('details');const summary=document.createElement('summary');summary.textContent=`Checkpoint ${x.checkpoint} · ${x.case_id} · score ${x.score}`;detail.append(summary);addText(detail,'p',x.rationale);addText(detail,'strong','Prompt');addText(detail,'pre',x.prompt);addText(detail,'strong','Output');addText(detail,'pre',x.output);$('evidence').append(detail)}clear($('limits'));for(const item of current.limitations)addText($('limits'),'li',item)}
async function openStudy(id){const response=await fetch('/api/experiments/'+encodeURIComponent(id));if(!response.ok)throw Error('Unable to load experiment');current=await response.json();document.querySelectorAll('.study-list button').forEach(b=>b.classList.toggle('active',b.dataset.id===id));render()}
async function boot(){try{const response=await fetch('/api/experiments');if(!response.ok)throw Error('Unable to load experiments');const data=await response.json();clear($('studies'));if(!data.length){addText($('studies'),'p','No experiments yet. Run traceai experiment run experiment.yaml.');return}for(const item of data){const button=document.createElement('button');button.dataset.id=item.id;button.textContent=`${item.name} · ${item.created_at}`;button.addEventListener('click',()=>openStudy(item.id).catch(error=>$('meta').textContent=error.message));$('studies').append(button)}await openStudy(data[0].id)}catch(error){$('studies').textContent=error.message}}
$('probe').addEventListener('change',event=>{selected=event.target.value;render()});boot();
</script></body></html>"""


def serve(repository: SQLiteRepository, port: int = 8765) -> None:
    if not 1 <= port <= 65535:
        raise ValueError("Port must be between 1 and 65535")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                self._send(PAGE.encode(), "text/html; charset=utf-8")
            elif self.path == "/api/experiments":
                self._send(json.dumps(repository.list_experiments()).encode(), "application/json")
            elif self.path.startswith("/api/experiments/"):
                experiment_id = unquote(self.path.removeprefix("/api/experiments/"))
                if not re.fullmatch(r"[0-9a-f-]{36}", experiment_id):
                    self.send_error(400)
                    return
                try:
                    from traceai.engine import Experiment
                    from traceai.schemas import ExperimentConfig

                    record = repository.get(experiment_id)
                    report = Experiment(
                        ExperimentConfig.model_validate(record["config"]),
                        repository.project_dir,
                        repository,
                        validate_dataset=False,
                    ).report(experiment_id)
                except StorageError:
                    self.send_error(404)
                    return
                self._send(report.model_dump_json().encode(), "application/json")
            else:
                self.send_error(404)

        def _send(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'",
            )
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"TraceAI dashboard: http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
