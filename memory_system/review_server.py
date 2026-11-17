"""Local plan-review form backed by the runner's request/response files.

The list page polls for new requests, so a reviewer can leave it open: the tab
title counts open requests, a sound and a desktop notification announce new
ones once enabled, and the newest open request can open by itself. A request
page counts down to the timeout and stops accepting replies once the request
has ended.
"""

from dataclasses import asdict
from functools import partial
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from urllib.parse import parse_qs, urlparse

from .coordination import Plan
from .intervention import may_vouch, write_json
from .repair_scopes import RepairTarget
from .review_domains import get_review_adapter


STYLE = """body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px;color:#182333;background:#fff}
table{border-collapse:collapse;width:100%;margin:14px 0}td,th{border:1px solid #d6dee8;padding:9px;text-align:left}
button,select{font:inherit;padding:9px;margin:5px}button{cursor:pointer}button:disabled{cursor:not-allowed;opacity:.5}
pre{white-space:pre-wrap;background:#f1f5f9;padding:16px}
.note{color:#536477}.card{border:1px solid #d6dee8;border-radius:10px;padding:18px;margin:16px 0}a{color:#1757a6}
.open{background:#fff4d6;font-weight:600}.banner{padding:12px 16px;border-radius:8px;margin:12px 0;background:#e8eef6}
.alert{background:#fde2e1;color:#8a1c17}.countdown{font-size:22px;font-weight:700}"""

STATUS_NAMES = {
    "pending": "待审", "answered": "已回复，等待执行", "expired": "已超时", "timeout": "已超时",
    "accepted": "已采纳", "invalid": "复核未通过", "permission": "越权", "stale_plan": "计划已变化",
    "reject": "已跳过", "episode_ended": "本局已结束",
}

SIGNAL_NAMES = {
    "choice": "要做取舍，有几个 Agent 都能帮忙，请决定让谁去", "unknown": "拿不准，系统缺少信息",
    "stalled": "团队卡住，已经很久没有新进展", "stage": "处于关键阶段", "all": "练习模式",
}

LIST_SCRIPT = """
const KEY_AUTO = 'review-auto-open';
let known = null, audio = null;
const auto = document.getElementById('auto');
try { auto.checked = localStorage.getItem(KEY_AUTO) !== 'off'; } catch (e) { auto.checked = true; }
auto.addEventListener('change', () => { try { localStorage.setItem(KEY_AUTO, auto.checked ? 'on' : 'off'); } catch (e) {} });
document.getElementById('enable').addEventListener('click', async () => {
  audio = audio || new AudioContext();
  await audio.resume();
  if ('Notification' in window && Notification.permission === 'default') { await Notification.requestPermission(); }
  document.getElementById('enable').textContent = '提醒已开启';
});
function beep() {
  if (!audio) return;
  const osc = audio.createOscillator(), gain = audio.createGain();
  osc.frequency.value = 880; gain.gain.value = 0.2;
  osc.connect(gain); gain.connect(audio.destination);
  osc.start(); osc.stop(audio.currentTime + 0.35);
}
function cell(row, text, cls) { const td = row.insertCell(); td.textContent = text; if (cls) td.className = cls; return td; }
async function poll() {
  let data;
  try { data = await (await fetch('/api/requests', {cache: 'no-store'})).json(); }
  catch (e) { document.getElementById('state').textContent = '暂时连不上运行程序（可能在两局之间），正在重试…'; return; }
  document.getElementById('state').textContent = '当前一局：' + data.run + '。每 2 秒自动刷新。';
  // Request ids restart at review_0001 in every episode, so key them by episode too.
  const requests = data.requests.map(r => Object.assign(r, {key: data.run + '/' + r.id}));
  const open = requests.filter(r => r.status === 'pending');
  document.title = (open.length ? '(' + open.length + ') 待审 · ' : '') + '计划审阅';
  const body = document.getElementById('rows');
  body.replaceChildren();
  for (const r of requests.slice().reverse()) {
    const row = body.insertRow();
    if (r.status === 'pending') row.className = 'open';
    const link = document.createElement('a');
    link.href = '/review?id=' + encodeURIComponent(r.id); link.textContent = r.id;
    row.insertCell().appendChild(link);
    cell(row, r.step); cell(row, r.edit_agents.join(', '));
    cell(row, r.status_name);
    cell(row, r.status === 'pending' && r.remaining_seconds !== null ? Math.ceil(r.remaining_seconds) + ' 秒' : '');
  }
  const fresh = open.filter(r => known !== null && !known.has(r.key));
  if (fresh.length) {
    beep();
    if ('Notification' in window && Notification.permission === 'granted') {
      new Notification('有新的计划审阅请求', {body: fresh.map(r => r.id + ' · 第 ' + r.step + ' 步').join('\\n')});
    }
    if (auto.checked) { window.location = '/review?id=' + encodeURIComponent(fresh[fresh.length - 1].id); return; }
  }
  known = new Set(requests.map(r => r.key));
}
poll(); setInterval(poll, 2000);
"""

REQUEST_SCRIPT = """
const id = document.body.dataset.request, run = document.body.dataset.run;
let remaining = Number(document.body.dataset.remaining);
const clock = document.getElementById('countdown');
function endRequest(message) {
  document.querySelectorAll('#reply button').forEach(button => { button.disabled = true; });
  const banner = document.getElementById('ended');
  banner.textContent = message; banner.hidden = false;
  clock.textContent = '0 秒';
}
const timer = setInterval(() => {
  remaining = Math.max(0, remaining - 1);
  clock.textContent = Math.ceil(remaining) + ' 秒';
  if (remaining <= 0) { clearInterval(timer); endRequest('已超时：这次回复不会被采用，系统已改用自动修复。'); }
}, 1000);
setInterval(async () => {
  let data;
  try { data = await (await fetch('/api/requests', {cache: 'no-store'})).json(); }
  catch (e) { return; }
  if (data.run !== run) { clearInterval(timer); endRequest('这一局已经结束。'); return; }
  const current = data.requests.find(r => r.id === id);
  if (current && current.status !== 'pending') { clearInterval(timer); endRequest('此请求已结束：' + current.status_name + '。'); }
}, 2000);
"""


def _label(plan):
    return f"{plan['operation']} | target={plan['material']} | share={plan['item']} → {plan['recipient']}"


def request_state(directory, request_id, now=None):
    """Where one request stands: pending, answered, expired, or the runner's final status."""
    directory = Path(directory)
    request = json.loads((directory / f"{request_id}.json").read_text(encoding="utf-8"))
    status_path = directory / f"{request_id}.status.json"
    remaining = None
    if request.get("issued_at") is not None:
        now = time.time() if now is None else now
        remaining = max(0.0, request["issued_at"] + request["timeout_seconds"] - now)
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))["status"]
    elif (directory / f"{request_id}.response.json").exists():
        status = "answered"
    elif remaining == 0:
        status = "expired"
    else:
        status = "pending"
    return {"id": request_id, "step": request["step"], "edit_agents": request["edit_agents"],
            "status": status, "status_name": STATUS_NAMES.get(status, status),
            "remaining_seconds": remaining}


class ReviewHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, directory, **kwargs):
        self.directory = Path(directory)
        # The episode folder; a batch starts a new server per episode on the same port.
        self.run = self.directory.resolve().parent.name
        super().__init__(*args, **kwargs)

    def log_message(self, *args):
        pass

    def send(self, content, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def page(self, body, status=200, title="计划审阅", attributes="", script=""):
        content = (f'<!doctype html><html lang="zh"><meta charset="utf-8"><title>{escape(title)}</title>'
                   f'<style>{STYLE}</style><body{attributes}><a href="/">审阅列表</a>{body}'
                   + (f'<script>{script}</script>' if script else '') + '</body></html>').encode("utf-8")
        self.send(content, "text/html; charset=utf-8", status)

    def requests(self):
        return [request_state(self.directory, path.stem)
                for path in sorted(self.directory.glob("review_[0-9][0-9][0-9][0-9].json"))]

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/api/requests":
            return self.send(json.dumps({"run": self.run, "requests": self.requests()},
                                        ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
        if url.path == "/":
            return self.page(
                '<h1>计划审阅</h1>'
                '<p>保持本页打开即可：新请求到达时，标签页标题会显示待审数量，并发出提示。</p>'
                '<p><button id="enable" type="button">开启声音和桌面提醒</button>'
                '<label><input id="auto" type="checkbox"> 新请求到达时自动打开</label></p>'
                '<p class="note">每次打开页面后需要点一次“开启声音和桌面提醒”，浏览器才允许发声。'
                '阅读时间从打开请求页面开始计算。</p><p id="state" class="note"></p>'
                '<table><thead><tr><th>请求</th><th>步数</th><th>可编辑 Agent</th><th>状态</th><th>剩余时间</th></tr></thead>'
                '<tbody id="rows"></tbody></table>', script=LIST_SCRIPT)
        if url.path != "/review":
            return self.page('<h1>页面不存在</h1>', status=404)
        request_id = parse_qs(url.query)["id"][0]
        request = json.loads((self.directory / f"{request_id}.json").read_text(encoding="utf-8"))
        state = request_state(self.directory, request_id)
        is_open = state["status"] == "pending"
        if is_open:
            opened = self.directory / f"{request_id}.opened.json"
            if not opened.exists():
                write_json(opened, {"opened_at": time.time()})
        proposed = {plan["agent_id"]: plan for plan in request["proposal"]}
        originals = {plan["agent_id"]: plan for plan in request["plans"]}
        fields = []
        for agent_id in request["edit_agents"]:
            options = request["domains"][str(agent_id)]
            selected = proposed.get(agent_id, originals[agent_id])
            choices = ''.join(f'<option value="{index}" {"selected" if option == selected else ""}>'
                              f'{escape(_label(option))}</option>' for index, option in enumerate(options))
            fields.append(f'<tr><td>{agent_id}</td><td>{escape(_label(originals[agent_id]))}</td>'
                          f'<td><select name="agent_{agent_id}">{choices}</select></td>'
                          f'<td>{escape(", ".join(request["allowed_fields"][str(agent_id)]))}</td></tr>')
        signals = request.get("trigger_signals", {})

        def reason(target):
            names = [SIGNAL_NAMES.get(signal, signal) for signal in signals.get(target["target_id"], [])]
            return f'<span class="note">（为什么问你：{escape("、".join(names))}）</span>' if names else ''

        goals = ''.join(f'<li>Agent {target["agent_id"]}: {target["kind"]} {escape(target["subject"])} '
                        f'{reason(target)}</li>' for target in request["targets"])
        evidence = request["evidence"]

        def display_fact(key, value):
            if key.startswith('view:') and value is not None:
                return {position: evidence['material_names'][str(material)] or 'unknown'
                        for position, material in value.items()}
            return value

        facts = ''.join(f'<details class="card" open><summary>{escape(key)}</summary>'
                        f'<pre>{escape(json.dumps(display_fact(key, value), ensure_ascii=False, indent=2))}</pre></details>'
                        for key, value in evidence["facts"].items())
        if request['proposal']:
            proposal_note = '推荐方案已在请求时的快照中通过联合验证；提交时还会复核。'
        elif request.get('accept_unknown'):
            proposal_note = ('系统缺少信息，无法验证任何修改，所以没有推荐方案。请根据下方依据自己判断：'
                             '检查结果为 unknown 时也可以提交，系统会按你的判断执行；'
                             '检查发现某个问题肯定修不好（infeasible）的修改仍会被拒绝。')
        else:
            proposal_note = '当前授权范围内没有已验证的联合修复建议；可以自行调整、预检或跳过。'
        remaining = state["remaining_seconds"]
        clock = (f'<p>剩余时间：<span id="countdown" class="countdown">{int(remaining + 0.999)} 秒</span>'
                 f'（超时后自动改用自动修复）</p>') if is_open and remaining is not None else ''
        ended = '' if is_open else f'此请求已结束：{state["status_name"]}。'
        disabled = '' if is_open else ' disabled'
        self.page(
            f'<h1>{escape(request_id)} · 第 {request["step"]} 步</h1>{clock}'
            f'<p id="ended" class="banner alert"{"" if ended else " hidden"}>{escape(ended)}</p>'
            f'<ul>{goals}</ul>'
            + (f'<p>剩余请求额度：{request["remaining_requests"]}。</p>'
               if request.get("remaining_requests") is not None else '')
            + f'<p>等待的 Agent：{request["pause_agents"]}。</p>'
            '<p class="note">可以查看关联证据；只能修改下表授权范围。提交后的实际执行结果会单独记录。</p>'
            f'<p>{proposal_note}</p>'
            f'<form id="reply" method="post" action="/respond"><input type="hidden" name="id" value="{escape(request_id)}">'
            '<table><tr><th>Agent</th><th>原计划</th><th>选择计划</th><th>允许修改字段</th></tr>'
            + ''.join(fields) + f'</table><button name="decision" value="edit"{disabled}>提交所选计划</button>'
            f'<button name="decision" value="preview"{disabled}>检查所选计划</button>'
            f'<button name="decision" value="approve"{disabled}>采用推荐方案</button>'
            f'<button name="decision" value="reject"{disabled}>跳过此次审阅</button></form>'
            f'<h2>判断依据</h2><p class="note">{evidence["fact_groups_after"]} 个库存/观测证据组。</p>{facts}',
            title=f'{request_id} · 计划审阅',
            attributes=(f' data-request="{escape(request_id)}" data-run="{escape(self.run)}"'
                        f' data-remaining="{remaining}"'
                        if is_open and remaining is not None else ''),
            script=REQUEST_SCRIPT if is_open and remaining is not None else '')

    def do_POST(self):
        fields = parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode("utf-8"))
        request_id, decision = fields["id"][0], fields["decision"][0]
        request = json.loads((self.directory / f"{request_id}.json").read_text(encoding="utf-8"))
        state = request_state(self.directory, request_id)
        if state["status"] != "pending":
            # A late reply would be ignored by the runner anyway; say so instead of storing it.
            return self.page(f'<h1>此请求已结束：{escape(state["status_name"])}</h1>'
                             '<p>这次回复没有被采用。</p><a href="/">返回审阅列表</a>', status=409)
        edits = request["proposal"] if decision == "approve" else [
            request["domains"][str(agent_id)][int(fields[f"agent_{agent_id}"][0])]
            for agent_id in request["edit_agents"]
        ] if decision in ("edit", "preview") else []
        if decision == "preview":
            adapter = get_review_adapter(request["environment"])

            def check(proposed):
                return adapter.verify(
                    [Plan(**plan) for plan in request["plans"]], adapter.restore(request["snapshot"]),
                    [RepairTarget(**target) for target in request["targets"]], [Plan(**edit) for edit in proposed],
                    {int(key): tuple(Plan(**plan) for plan in values) for key, values in request["domains"].items()},
                    {int(key): value for key, value in request["evidence"]["navigator_actions"].items()},
                )

            result = check(edits)
            if result.status == 'unknown' and request.get('accept_unknown'):
                note = ('<p class="banner">系统无法确认这样改能否修好。若你根据依据判断可行，可以返回提交。</p>'
                        if may_vouch(check([]), result) else
                        '<p class="banner alert">这样改帮不上任何一个问题，提交会被拒绝。</p>')
            else:
                note = ''
            return self.page(f'<h1>检查结果：{result.status}</h1>{note}<pre>'
                             + escape(json.dumps(asdict(result), ensure_ascii=False, indent=2))
                             + f'</pre><a href="/review?id={escape(request_id)}">返回修改</a>')
        opened = self.directory / f"{request_id}.opened.json"
        started = json.loads(opened.read_text(encoding="utf-8"))["opened_at"] if opened.exists() else time.time()
        write_json(self.directory / f"{request_id}.response.json", {
            "request_id": request_id, "decision": decision, "edits": edits,
            "read_seconds": time.time() - started, "provider": "human_web",
        })
        self.page('<h1>回复已提交</h1><p>执行器会使用当前状态复核，并记录实际执行结果。</p>'
                  '<p><a href="/">返回审阅列表</a>，有新请求时会再次提醒。</p>')


class ReviewServer:
    def __init__(self, directory, port=8765):
        self.server = ThreadingHTTPServer(("127.0.0.1", port), partial(ReviewHandler, directory=directory))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self):
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_port}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
