#!/usr/bin/env python3
"""Lambda Cloud GPU lifecycle manager — make cost dynamic: launch on demand,
terminate when idle.

CLI:
  python -m erosolar_agent.serving.gpu_manager status
  python -m erosolar_agent.serving.gpu_manager up      [--name erosolar-serve] [--type gpu_1x_h100_pcie]
  python -m erosolar_agent.serving.gpu_manager down     [--name erosolar-serve | --id <iid>]
  python -m erosolar_agent.serving.gpu_manager touch    # record activity (resets idle timer)
  python -m erosolar_agent.serving.gpu_manager sweep --idle-min 15   # terminate if idle too long

Auth via LAMBDA_API_KEY (erosolar_agent.secrets). State (managed instance id +
last-activity) is a small JSON file (default ~/.erosolar/gpu_state.json).

NOTE: `up` launches a *bare* instance. Serving-on-demand (auto-starting vLLM from
a persistent filesystem so cold starts don't re-download the model) is layered on
top of this at deploy time; this module is the reusable lifecycle primitive.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .. import secrets

API = "https://cloud.lambda.ai/api/v1"
DEFAULT_TYPE = "gpu_1x_h100_pcie"
DEFAULT_REGIONS = ["us-west-3", "us-south-2", "us-south-3", "us-southeast-1", "us-east-1"]
DEFAULT_SSH_KEYS = ["lambda_lab"]
STATE_PATH = Path.home() / ".erosolar" / "gpu_state.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _req(method: str, path: str, body=None):
    key = secrets.lambda_api_key(required=True)
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        API + path, data=data, method=method,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 "User-Agent": "curl/8.7.1"},  # cloud.lambda.ai's CDN 403s the default urllib UA
    )
    try:
        with urllib.request.urlopen(req, timeout=40) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace") if e.fp else ""
        raise RuntimeError(f"Lambda API {method} {path} -> HTTP {e.code}: {detail[:300]}") from e


# --- raw API ---------------------------------------------------------------

def list_instances() -> list:
    return _req("GET", "/instances").get("data", [])


def get_instance(iid: str) -> dict:
    return _req("GET", f"/instances/{iid}").get("data", {})


def find_by_name(name: str):
    for inst in list_instances():
        if inst.get("name") == name:
            return inst
    return None


def regions_with_capacity(instance_type: str) -> list:
    data = _req("GET", "/instance-types").get("data", {})
    entry = data.get(instance_type, {})
    return [r["name"] for r in entry.get("regions_with_capacity_available", [])]


def launch(name: str, instance_type: str, regions: list, ssh_keys: list) -> str:
    avail = regions_with_capacity(instance_type)
    ordered = [r for r in regions if r in avail] or avail
    if not ordered:
        raise RuntimeError(f"no capacity for {instance_type} in any region right now")
    last_err = None
    for region in ordered:
        try:
            resp = _req("POST", "/instance-operations/launch", {
                "region_name": region, "instance_type_name": instance_type,
                "ssh_key_names": ssh_keys, "name": name,
            })
            return resp["data"]["instance_ids"][0]
        except RuntimeError as e:
            last_err = e
            continue
    raise RuntimeError(f"launch failed in all candidate regions: {last_err}")


def terminate(instance_ids: list) -> None:
    _req("POST", "/instance-operations/terminate", {"instance_ids": instance_ids})


# --- higher-level lifecycle ------------------------------------------------

def _wait_active(iid: str, timeout_s: int = 600) -> str:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        inst = get_instance(iid)
        status, ip = inst.get("status"), inst.get("ip")
        if status == "active" and ip:
            return ip
        if status in ("terminated", "error"):
            raise RuntimeError(f"instance {iid} entered status {status}")
        time.sleep(15)
    raise RuntimeError(f"instance {iid} not active within {timeout_s}s")


def _wait_health(url: str, timeout_s: int = 600) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(10)
    return False


def ensure_up(name: str = "erosolar-serve", instance_type: str = DEFAULT_TYPE,
              regions: list = None, ssh_keys: list = None, health_url: str = None) -> dict:
    """Return a running instance (launching if needed). If health_url is given
    (use {ip} placeholder), wait until the model server answers."""
    regions = regions or DEFAULT_REGIONS
    ssh_keys = ssh_keys or DEFAULT_SSH_KEYS
    inst = find_by_name(name)
    if inst and inst.get("status") in ("active", "booting"):
        iid = inst["id"]
    else:
        iid = launch(name, instance_type, regions, ssh_keys)
    ip = _wait_active(iid)
    if health_url:
        _wait_health(health_url.format(ip=ip))
    save_state({"instance_id": iid, "name": name, "ip": ip, "last_activity": _now()})
    return {"instance_id": iid, "ip": ip, "name": name}


# --- state + idle ----------------------------------------------------------

def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def touch() -> None:
    st = load_state()
    st["last_activity"] = _now()
    save_state(st)


def idle_seconds() -> float:
    st = load_state()
    last = st.get("last_activity")
    if not last:
        return float("inf")
    delta = datetime.now(timezone.utc) - datetime.fromisoformat(last)
    return delta.total_seconds()


def sweep(name: str, idle_min: float) -> bool:
    """Terminate the managed instance if idle longer than idle_min. Returns True if terminated."""
    inst = find_by_name(name)
    if not inst or inst.get("status") not in ("active", "booting"):
        return False
    if idle_seconds() >= idle_min * 60:
        terminate([inst["id"]])
        st = load_state()
        st["terminated_at"] = _now()
        save_state(st)
        return True
    return False


# --- CLI -------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["status", "up", "down", "touch", "sweep"])
    ap.add_argument("--name", default="erosolar-serve")
    ap.add_argument("--type", default=DEFAULT_TYPE)
    ap.add_argument("--id", default=None)
    ap.add_argument("--idle-min", type=float, default=15)
    args = ap.parse_args(argv)

    if args.action == "status":
        for inst in list_instances():
            print(f"  {inst.get('id')}  {inst.get('name')}  {inst.get('instance_type', {}).get('name')}"
                  f"  {inst.get('status')}  {inst.get('ip') or '-'}  {inst.get('region', {}).get('name')}")
        print(f"  idle: {idle_seconds():.0f}s since last touch")
        return 0
    if args.action == "up":
        info = ensure_up(name=args.name, instance_type=args.type)
        print(f"UP: {info['name']} {info['instance_id']} ip={info['ip']}")
        return 0
    if args.action == "down":
        ids = [args.id] if args.id else ([find_by_name(args.name)["id"]] if find_by_name(args.name) else [])
        if not ids:
            print("no matching instance to terminate")
            return 0
        terminate(ids)
        print(f"TERMINATED: {ids}")
        return 0
    if args.action == "touch":
        touch()
        print("activity recorded")
        return 0
    if args.action == "sweep":
        did = sweep(args.name, args.idle_min)
        print(f"swept (terminated)={did}; idle={idle_seconds():.0f}s")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
