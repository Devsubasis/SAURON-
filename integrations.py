"""
SAURON++ external-integration sinks.

A non-blocking dispatcher fans out *already-deduplicated, FDR-passed* alerts
(the dicts returned by AlertManager.process) to external systems:

    * ELK / Elasticsearch  (bulk-friendly _doc index)
    * SIEM                 (CEF over syslog: Splunk / QRadar / ArcSight)
    * Slack                (incoming webhook)
    * Email                (SMTP + STARTTLS)
    * SMS                  (Twilio REST)
    * Ticketing            (PagerDuty Events v2  and/or  Jira REST)

Design notes
------------
* Stdlib only (urllib, smtplib, logging.handlers) -> no new dependencies.
* Sinks run on a background worker thread fed by a Queue, so blocking HTTP/SMTP
  never stalls the FastAPI event loop.
* Each sink has a severity floor, so paging/SMS only fire on serious alerts
  while ELK/SIEM get everything. Reuse of AlertManager's dedup means external
  systems are not spammed; an extra per-sink token bucket is a second guard.
* Every sink call is wrapped in try/except: one failing sink cannot break others.

Wiring (backend/sauron.py, in SauronEngine)
-------------------------------------------
    import integrations as _intg                       # near the other imports
    self.dispatcher = _intg.build_dispatcher_from_env() # in __init__
    ...
    alert = self.alerts.process(ev)
    if alert is not None:
        ev["alert_id"] = alert["alert_id"]; ...
        self.dispatcher.dispatch(ev)                    # <-- the one added line
"""

from __future__ import annotations
import base64
import json
import os
import queue
import smtplib
import ssl
import threading
import time
import urllib.request
import urllib.error
import urllib.parse
from email.mime.text import MIMEText
from logging.handlers import SysLogHandler
import logging
from typing import Dict, List, Optional

SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def _sev(alert: Dict) -> int:
    return SEVERITY_ORDER.get(str(alert.get("severity", "low")).lower(), 0)


def _post(url: str, data: bytes, headers: Dict[str, str], timeout: float = 5.0) -> int:
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def _post_json(url: str, payload: dict, headers: Optional[Dict[str, str]] = None, timeout: float = 5.0) -> int:
    h = {"Content-Type": "application/json"}
    if headers:
        h.update(headers)
    return _post(url, json.dumps(payload).encode(), h, timeout)


class _TokenBucket:
    """Second-line flood guard, per sink."""
    def __init__(self, rate_per_min: float, burst: int):
        self.rate = rate_per_min / 60.0
        self.tokens = float(burst)
        self.burst = float(burst)
        self.t = time.time()

    def allow(self) -> bool:
        now = time.time()
        self.tokens = min(self.burst, self.tokens + (now - self.t) * self.rate)
        self.t = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


# --------------------------------------------------------------------------- #
# Sinks                                                                        #
# --------------------------------------------------------------------------- #
class Sink:
    name = "sink"

    def __init__(self, min_sev: str = "low", rate_per_min: float = 120, burst: int = 30):
        self.min_sev = SEVERITY_ORDER.get(min_sev, 0)
        self.bucket = _TokenBucket(rate_per_min, burst)

    def matches(self, alert: Dict) -> bool:
        return _sev(alert) >= self.min_sev and self.bucket.allow()

    def send(self, alert: Dict) -> None:  # pragma: no cover - network
        raise NotImplementedError


def _summary(a: Dict) -> str:
    return (f"[SAURON++] {a.get('severity','?').upper()} {a.get('action','?')} "
            f"src={a.get('src_ip','?')} dst={a.get('dst_ip','?')} "
            f"score={a.get('aggregate', a.get('raw_aggregate','?'))} "
            f"reason={a.get('reason','')}")


class ElasticSink(Sink):
    """POST each alert as a document to Elasticsearch / OpenSearch."""
    name = "elastic"

    def __init__(self, url: str, index: str = "sauronpp-alerts", api_key: str = "",
                 user: str = "", password: str = "", **kw):
        super().__init__(**kw)
        self.endpoint = f"{url.rstrip('/')}/{index}/_doc"
        self.headers: Dict[str, str] = {}
        if api_key:
            self.headers["Authorization"] = f"ApiKey {api_key}"
        elif user:
            tok = base64.b64encode(f"{user}:{password}".encode()).decode()
            self.headers["Authorization"] = f"Basic {tok}"

    def send(self, alert: Dict) -> None:
        doc = dict(alert)
        doc.setdefault("@timestamp", int(time.time() * 1000))
        _post_json(self.endpoint, doc, self.headers)


class SiemCEFSink(Sink):
    """Emit ArcSight CEF over syslog (Splunk/QRadar/ArcSight all parse CEF)."""
    name = "siem"

    def __init__(self, host: str, port: int = 514, proto: str = "udp", **kw):
        super().__init__(**kw)
        socktype = None
        import socket as _s
        socktype = _s.SOCK_DGRAM if proto == "udp" else _s.SOCK_STREAM
        self.handler = SysLogHandler(address=(host, port), socktype=socktype)
        self.logger = logging.getLogger("sauronpp.siem")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(self.handler)
        self.logger.propagate = False

    @staticmethod
    def _cef(a: Dict) -> str:
        sev = {"low": 3, "medium": 6, "high": 8, "critical": 10}.get(str(a.get("severity", "low")).lower(), 3)
        ext = (f"src={a.get('src_ip','')} dst={a.get('dst_ip','')} "
               f"act={a.get('action','')} cs1Label=reason cs1={a.get('reason','')} "
               f"cn1Label=aggregate cn1={a.get('aggregate', a.get('raw_aggregate',0))}")
        sig = a.get("action", "ALERT")
        name = a.get("reason", "SAURON++ alert")[:60]
        return f"CEF:0|Anthropic|SAURON++|2.0|{sig}|{name}|{sev}|{ext}"

    def send(self, alert: Dict) -> None:
        self.logger.info(self._cef(alert))


class SlackSink(Sink):
    """POST a message to a Slack incoming webhook."""
    name = "slack"

    def __init__(self, webhook_url: str, **kw):
        super().__init__(**kw)
        self.url = webhook_url

    def send(self, alert: Dict) -> None:
        color = {"critical": "#b30000", "high": "#e06c00", "medium": "#d1a300"}.get(
            str(alert.get("severity", "low")).lower(), "#888888")
        _post_json(self.url, {"attachments": [{"color": color, "text": _summary(alert)}]})


class EmailSink(Sink):
    """Send an alert email via SMTP + STARTTLS."""
    name = "email"

    def __init__(self, host: str, port: int, user: str, password: str,
                 sender: str, to: str, **kw):
        super().__init__(**kw)
        self.host, self.port, self.user, self.password = host, port, user, password
        self.sender, self.to = sender, [x.strip() for x in to.split(",") if x.strip()]

    def send(self, alert: Dict) -> None:
        msg = MIMEText(json.dumps(alert, indent=2))
        msg["Subject"] = _summary(alert)[:120]
        msg["From"] = self.sender
        msg["To"] = ", ".join(self.to)
        ctx = ssl.create_default_context()
        with smtplib.SMTP(self.host, self.port, timeout=8) as s:
            s.starttls(context=ctx)
            if self.user:
                s.login(self.user, self.password)
            s.sendmail(self.sender, self.to, msg.as_string())


class SMSSink(Sink):
    """Send an SMS via Twilio REST (basic auth)."""
    name = "sms"

    def __init__(self, sid: str, token: str, from_: str, to: str, **kw):
        kw.setdefault("min_sev", "high")
        super().__init__(**kw)
        self.url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
        self.auth = base64.b64encode(f"{sid}:{token}".encode()).decode()
        self.from_, self.to = from_, to

    def send(self, alert: Dict) -> None:
        body = urllib.parse.urlencode({"From": self.from_, "To": self.to, "Body": _summary(alert)[:300]}).encode()
        _post(self.url, body, {"Authorization": f"Basic {self.auth}",
                               "Content-Type": "application/x-www-form-urlencoded"})


class PagerDutySink(Sink):
    """Trigger a PagerDuty incident via Events API v2."""
    name = "pagerduty"

    def __init__(self, routing_key: str, **kw):
        kw.setdefault("min_sev", "high")
        super().__init__(**kw)
        self.routing_key = routing_key

    def send(self, alert: Dict) -> None:
        sev = str(alert.get("severity", "warning")).lower()
        sev = sev if sev in ("critical", "error", "warning", "info") else "warning"
        payload = {
            "routing_key": self.routing_key, "event_action": "trigger",
            "dedup_key": str(alert.get("flow_id", alert.get("alert_id", ""))),
            "payload": {"summary": _summary(alert), "severity": sev,
                        "source": alert.get("src_ip", "sauronpp"), "custom_details": alert},
        }
        _post_json("https://events.pagerduty.com/v2/enqueue", payload)


class JiraSink(Sink):
    """Open a Jira issue via REST (basic auth = email:api_token)."""
    name = "jira"

    def __init__(self, base_url: str, email: str, api_token: str, project_key: str,
                 issue_type: str = "Task", **kw):
        kw.setdefault("min_sev", "high")
        super().__init__(**kw)
        self.url = f"{base_url.rstrip('/')}/rest/api/2/issue"
        self.auth = base64.b64encode(f"{email}:{api_token}".encode()).decode()
        self.project, self.issue_type = project_key, issue_type

    def send(self, alert: Dict) -> None:
        payload = {"fields": {"project": {"key": self.project},
                              "summary": _summary(alert)[:200],
                              "description": json.dumps(alert, indent=2),
                              "issuetype": {"name": self.issue_type}}}
        _post_json(self.url, payload, {"Authorization": f"Basic {self.auth}"})


# --------------------------------------------------------------------------- #
# Dispatcher                                                                   #
# --------------------------------------------------------------------------- #
class AlertDispatcher:
    """Background fan-out to all configured sinks (non-blocking)."""

    def __init__(self, sinks: List[Sink], maxq: int = 2000):
        self.sinks = sinks
        self.q: "queue.Queue[Dict]" = queue.Queue(maxsize=maxq)
        self.log = logging.getLogger("sauronpp.dispatch")
        self._stop = threading.Event()
        self.worker = threading.Thread(target=self._run, name="sauron-dispatch", daemon=True)
        if sinks:
            self.worker.start()

    def dispatch(self, alert: Dict) -> None:
        if not self.sinks:
            return
        try:
            self.q.put_nowait(dict(alert))
        except queue.Full:
            self.log.warning("dispatch queue full; dropping alert %s", alert.get("alert_id"))

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                alert = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            for s in self.sinks:
                try:
                    if s.matches(alert):
                        s.send(alert)
                except Exception as e:  # pragma: no cover - network
                    self.log.warning("sink %s failed: %s", s.name, e)

    def stop(self) -> None:
        self._stop.set()


# --------------------------------------------------------------------------- #
# Env-driven wiring                                                            #
# --------------------------------------------------------------------------- #
def _e(k: str, default: str = "") -> str:
    return os.environ.get(k, default)


def build_dispatcher_from_env() -> AlertDispatcher:
    """Assemble whichever sinks have their env vars set. Absent config => no-op."""
    sinks: List[Sink] = []

    if _e("SAURON_ELASTIC_URL"):
        sinks.append(ElasticSink(_e("SAURON_ELASTIC_URL"),
                                 index=_e("SAURON_ELASTIC_INDEX", "sauronpp-alerts"),
                                 api_key=_e("SAURON_ELASTIC_APIKEY"),
                                 user=_e("SAURON_ELASTIC_USER"),
                                 password=_e("SAURON_ELASTIC_PASS"),
                                 min_sev=_e("SAURON_ELASTIC_MINSEV", "low")))
    if _e("SAURON_SIEM_HOST"):
        sinks.append(SiemCEFSink(_e("SAURON_SIEM_HOST"),
                                 port=int(_e("SAURON_SIEM_PORT", "514")),
                                 proto=_e("SAURON_SIEM_PROTO", "udp"),
                                 min_sev=_e("SAURON_SIEM_MINSEV", "low")))
    if _e("SAURON_SLACK_WEBHOOK"):
        sinks.append(SlackSink(_e("SAURON_SLACK_WEBHOOK"),
                               min_sev=_e("SAURON_SLACK_MINSEV", "medium")))
    if _e("SAURON_SMTP_HOST"):
        sinks.append(EmailSink(_e("SAURON_SMTP_HOST"), int(_e("SAURON_SMTP_PORT", "587")),
                               _e("SAURON_SMTP_USER"), _e("SAURON_SMTP_PASS"),
                               _e("SAURON_SMTP_FROM"), _e("SAURON_SMTP_TO"),
                               min_sev=_e("SAURON_SMTP_MINSEV", "high")))
    if _e("SAURON_TWILIO_SID"):
        sinks.append(SMSSink(_e("SAURON_TWILIO_SID"), _e("SAURON_TWILIO_TOKEN"),
                             _e("SAURON_TWILIO_FROM"), _e("SAURON_TWILIO_TO"),
                             min_sev=_e("SAURON_TWILIO_MINSEV", "high")))
    if _e("SAURON_PAGERDUTY_KEY"):
        sinks.append(PagerDutySink(_e("SAURON_PAGERDUTY_KEY"),
                                   min_sev=_e("SAURON_PAGERDUTY_MINSEV", "high")))
    if _e("SAURON_JIRA_URL"):
        sinks.append(JiraSink(_e("SAURON_JIRA_URL"), _e("SAURON_JIRA_EMAIL"),
                              _e("SAURON_JIRA_TOKEN"), _e("SAURON_JIRA_PROJECT"),
                              issue_type=_e("SAURON_JIRA_TYPE", "Task"),
                              min_sev=_e("SAURON_JIRA_MINSEV", "high")))

    logging.getLogger("sauronpp.dispatch").info(
        "external sinks enabled: %s", [s.name for s in sinks] or "none")
    return AlertDispatcher(sinks)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    d = build_dispatcher_from_env()
    d.dispatch({"alert_id": "test-1", "severity": "high", "action": "DROP",
                "src_ip": "10.0.0.7", "dst_ip": "10.0.0.1", "aggregate": 0.97,
                "reason": "self-test alert"})
    time.sleep(1.0)
    print("dispatched self-test to:", [s.name for s in d.sinks] or "no sinks (set env vars)")
