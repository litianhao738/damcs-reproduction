"""Local plan-review form backed by the runner's request/response files."""

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
from .intervention import write_json
from .repair_scopes import RepairTarget
from .review_domains import get_review_adapter


STYLE = """body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px;color:#182333}
table{border-collapse:collapse;width:100%;margin:14px 0}td,th{border:1px solid #d6dee8;padding:9px;text-align:left}
button,select{font:inherit;padding:9px;margin:5px}button{cursor:pointer}pre{white-space:pre-wrap;background:#f1f5f9;padding:16px}
.note{color:#536477}.card{border:1px solid #d6dee8;border-radius:10px;padding:18px;margin:16px 0}a{color:#1757a6}"""


def _label(plan):
    return f"{plan['operation']} | target={plan['material']} | share={plan['item']} → {plan['recipient']}"


class ReviewHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, directory, **kwargs):
        self.directory = Path(directory)
        super().__init__(*args, **kwargs)

    def log_message(self, *args):
        pass

    def page(self, body, status=200):
        content = (f'<!doctype html><html lang="zh"><meta charset="utf-8"><title>Plan review</title>'
                   f'<style>{STYLE}</style><body><a href="/">审阅列表</a>{body}</body></html>').encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        query = parse_qs(urlparse(self.path).query)
        if urlparse(self.path).path == "/":
            rows = []
            for path in sorted(self.directory.glob("review_[0-9][0-9][0-9][0-9].json")):
                request = json.loads(path.read_text(encoding="utf-8"))
                status_path = path.with_name(path.stem + ".status.json")
                status = json.loads(status_path.read_text(encoding="utf-8"))["status"] if status_path.exists() else "pending"
                rows.append(f'<tr><td><a href="/review?id={path.stem}">{path.stem}</a></td>'
                            f'<td>{request["step"]}</td><td>{request["edit_agents"]}</td><td>{status}</td></tr>')
            return self.page('<h1>计划审阅</h1><p>选择请求，查看依据并修改授权计划。</p>'
                             '<table><tr><th>请求</th><th>步数</th><th>可编辑 Agent</th><th>状态</th></tr>'
                             + ''.join(rows) + '</table>')
        request_id = query["id"][0]
        request = json.loads((self.directory / f"{request_id}.json").read_text(encoding="utf-8"))
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
        goals = ''.join(f'<li>Agent {target["agent_id"]}: {target["kind"]} {escape(target["subject"])}</li>'
                        for target in request["targets"])
        evidence = request["evidence"]
        def display_fact(key, value):
            if key.startswith('view:') and value is not None:
                return {position: evidence['material_names'][str(material)] or 'unknown'
                        for position, material in value.items()}
            return value
        facts = ''.join(f'<details class="card" open><summary>{escape(key)}</summary>'
                        f'<pre>{escape(json.dumps(display_fact(key, value), ensure_ascii=False, indent=2))}</pre></details>'
                        for key, value in evidence["facts"].items())
        proposal_note = '推荐方案已在请求时的快照中通过联合验证；提交时还会复核。' if request['proposal'] \
            else '当前授权范围内没有已验证的联合修复建议；可以自行调整、预检或跳过。'
        self.page(f'<h1>{escape(request_id)} · 第 {request["step"]} 步</h1><ul>{goals}</ul>'
                  f'<p>剩余请求额度：{request["remaining_requests"]}；等待的 Agent：{request["pause_agents"]}。</p>'
                  '<p class="note">可以查看关联证据；只能修改下表授权范围。提交后的实际执行结果会单独记录。</p>'
                  f'<p>{proposal_note}</p>'
                  f'<form method="post" action="/respond"><input type="hidden" name="id" value="{escape(request_id)}">'
                  '<table><tr><th>Agent</th><th>原计划</th><th>选择计划</th><th>允许修改字段</th></tr>'
                  + ''.join(fields) + '</table><button name="decision" value="edit">提交所选计划</button>'
                  '<button name="decision" value="preview">检查所选计划</button>'
                  '<button name="decision" value="approve">采用推荐方案</button>'
                  '<button name="decision" value="reject">跳过此次审阅</button></form>'
                  f'<h2>判断依据</h2><p class="note">{evidence["fact_groups_after"]} 个库存/观测证据组。</p>{facts}')

    def do_POST(self):
        fields = parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode("utf-8"))
        request_id, decision = fields["id"][0], fields["decision"][0]
        request = json.loads((self.directory / f"{request_id}.json").read_text(encoding="utf-8"))
        edits = request["proposal"] if decision == "approve" else [
            request["domains"][str(agent_id)][int(fields[f"agent_{agent_id}"][0])]
            for agent_id in request["edit_agents"]
        ] if decision in ("edit", "preview") else []
        if decision == "preview":
            adapter = get_review_adapter(request["environment"])
            result = adapter.verify(
                [Plan(**plan) for plan in request["plans"]], adapter.restore(request["snapshot"]),
                [RepairTarget(**target) for target in request["targets"]], [Plan(**edit) for edit in edits],
                {int(key): tuple(Plan(**plan) for plan in values) for key, values in request["domains"].items()},
                {int(key): value for key, value in request["evidence"]["navigator_actions"].items()},
            )
            return self.page(f'<h1>检查结果：{result.status}</h1><pre>'
                             + escape(json.dumps(asdict(result), ensure_ascii=False, indent=2))
                             + f'</pre><a href="/review?id={request_id}">返回修改</a>')
        opened = self.directory / f"{request_id}.opened.json"
        started = json.loads(opened.read_text(encoding="utf-8"))["opened_at"] if opened.exists() else time.time()
        write_json(self.directory / f"{request_id}.response.json", {
            "request_id": request_id, "decision": decision, "edits": edits,
            "read_seconds": time.time() - started, "provider": "human_web",
        })
        self.page('<h1>回复已提交</h1><p>执行器会使用当前状态复核，并记录实际执行结果。</p>')


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
