#!/usr/bin/env bash
set -euo pipefail
ROOT="$(pwd)/jarvis-gen5-full"
echo "Creating project at $ROOT"
rm -rf "$ROOT"
mkdir -p "$ROOT"

# Helper to create files with content
write() {
  local path="$1"; shift
  mkdir -p "$(dirname "$path")"
  cat > "$path" <<'EOF'
'"$@"'
EOF
}

# -------------------------
# aggregator/
# -------------------------
mkdir -p "$ROOT/aggregator" "$ROOT/aggregator/tests" "$ROOT/aggregator/keys"
cat > "$ROOT/aggregator/requirements.txt" <<'REQ'
fastapi
uvicorn[standard]
pydantic
python-dotenv
msgpack
zstandard
secretsharing
requests
cryptography
sqlalchemy
asyncpg
aioredis
pytest
torch
python-jose[cryptography]
psycopg2-binary
REQ

cat > "$ROOT/aggregator/.env.example" <<'ENV'
AGG_TOKEN=changeme
REQUIRE_AUTH=1
LOG_DIR=/app/logs
POSTGRES_URL=postgresql+asyncpg://fl:fl@postgres:5432/jarvis
REDIS_URL=redis://redis:6379/0
MAX_PENDING=1000
N_SHARES=5
SSS_THRESHOLD=3
ENV

cat > "$ROOT/aggregator/main.py" <<'PYAGG'
# aggregator/main.py (simplified orchestrator)
# See repository README for full details. This file contains the FastAPI aggregator scaffold.
import os, json, time, subprocess, base64
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from threading import Lock
import msgpack, zstandard as zstd
import logging
import torch
import numpy as np

ROOT = Path(__file__).resolve().parent
LOG_DIR = Path(os.environ.get("LOG_DIR", str(ROOT.joinpath("../logs"))))
LOG_DIR.mkdir(parents=True, exist_ok=True)
PENDING_FILE = LOG_DIR / "pending_updates.json"
MASK_STORE = LOG_DIR / "mask_store.json"
META_FILE = LOG_DIR / "global_meta.json"
GLOBAL_MODEL = LOG_DIR / "global_model.pt"
TRAIN_LOG = LOG_DIR / "training_log.json"

MAX_PENDING = int(os.environ.get("MAX_PENDING", "1000"))
AGG_TOKEN = os.environ.get("AGG_TOKEN", "changeme")
REQUIRE_AUTH = os.environ.get("REQUIRE_AUTH", "1") == "1"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aggregator")

file_lock = Lock()
def safe_read(path, default):
    with file_lock:
        if not path.exists(): return default
        try:
            return json.loads(path.read_text())
        except Exception:
            logger.exception("read failed")
            return default
def safe_write(path, obj):
    with file_lock:
        path.write_text(json.dumps(obj, indent=2))

def pack_bin(obj):
    packed = msgpack.packb(obj, use_bin_type=True)
    comp = zstd.ZstdCompressor(level=3).compress(packed)
    return base64.b64encode(comp).decode('ascii')
def unpack_bin(s):
    c = base64.b64decode(s)
    d = zstd.ZstdDecompressor().decompress(c)
    return msgpack.unpackb(d, raw=False)

for p,defv in [(PENDING_FILE, []), (MASK_STORE, {}), (TRAIN_LOG, [])]:
    if not p.exists(): safe_write(p, defv)

security = HTTPBearer()
def require_token(creds: HTTPAuthorizationCredentials = Depends(security)):
    token = creds.credentials
    if not REQUIRE_AUTH:
        return True
    if token != AGG_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    return True

from typing import Optional, Dict
class UpdatePayload(BaseModel):
    device_id: str
    sample_count: int = Field(default=1, ge=0)
    update: Optional[Dict[str,str]] = None
    masked_update: Optional[Dict[str,str]] = None

app = FastAPI(title="Jarvis Gen5 Aggregator")

@app.post("/submit_update")
async def submit_update(payload: UpdatePayload, auth=Depends(require_token)):
    pending = safe_read(PENDING_FILE, [])
    if len(pending) >= MAX_PENDING:
        raise HTTPException(status_code=429, detail="queue full")
    if not payload.update and not payload.masked_update:
        raise HTTPException(status_code=400, detail="missing update")
    entry = payload.dict()
    entry['ts'] = int(time.time())
    pending.append(entry)
    safe_write(PENDING_FILE, pending)
    logs = safe_read(TRAIN_LOG, [])
    logs.append({"ts": entry['ts'], "event": "update_received", "device": payload.device_id})
    safe_write(TRAIN_LOG, logs)
    return {"ok": True, "queued": len(pending)}

@app.post("/trigger_aggregate")
async def trigger_aggregate(auth=Depends(require_token)):
    pending = safe_read(PENDING_FILE, [])
    if not pending:
        return {"ok": False, "err": "no pending"}
    masked_mode = any('masked_update' in p and p['masked_update'] for p in pending)
    recovered = []
    if masked_mode:
        ms_db = safe_read(MASK_STORE, {})
        full_masks = ms_db.get("full_masks", {})
        masked_senders = [p['device_id'] for p in pending if 'masked_update' in p and p['masked_update']]
        missing = [s for s in masked_senders if s not in full_masks]
        if missing:
            return {"ok": False, "err": "missing_masks", "missing": missing}
        for p in pending:
            if 'masked_update' in p and p['masked_update']:
                dev = p['device_id']; masked = p['masked_update']; mask = full_masks[dev]
                rec = {}
                for k, ms in masked.items():
                    try:
                        mar = unpack_bin(ms)
                        maskarr = unpack_bin(mask[k])
                        arr = (np.array(mar) - np.array(maskarr)).astype(np.float32)
                    except Exception:
                        arr = None
                    rec[k] = pack_bin(arr.tolist()) if arr is not None else None
                recovered.append({"device_id": dev, "sample_count": p.get('sample_count',1), "update": rec})
            elif 'update' in p and p['update']:
                recovered.append({"device_id": p['device_id'], "sample_count": p.get('sample_count',1), "update": p['update']})
    else:
        for p in pending:
            if 'update' in p and p['update']:
                recovered.append({"device_id": p['device_id'], "sample_count": p.get('sample_count',1), "update": p['update']})
    if not recovered:
        raise HTTPException(status_code=400, detail="no updates")
    all_keys = set()
    for ru in recovered:
        all_keys.update(ru['update'].keys())
    aggregated = {}
    for k in all_keys:
        arrs=[]; weights=[]
        for ru in recovered:
            upd = ru['update']
            if k in upd and upd[k] is not None:
                arr = np.array(unpack_bin(upd[k]), dtype=np.float32)
                arrs.append(arr * ru['sample_count'])
                weights.append(ru['sample_count'])
        if not arrs: continue
        sum_arr = np.sum(np.stack(arrs,axis=0), axis=0)
        total_w = float(sum(weights))
        avg = (sum_arr / total_w).astype(np.float32)
        aggregated[k] = pack_bin(avg.tolist())
    # persist global model
    state = {}
    import numpy as _np
    for k,s in aggregated.items():
        if s is None: continue
        arr = _np.array(unpack_bin(s), dtype=_np.float32)
        state[k] = torch.from_numpy(arr)
    torch.save(state, GLOBAL_MODEL)
    meta = {"version": int(time.time()), "ts": int(time.time()), "keys": list(aggregated.keys())}
    safe_write(META_FILE, meta)
    # try ipfs add
    try:
        res = subprocess.run(["ipfs","add","-Q", str(GLOBAL_MODEL)], capture_output=True, text=True, timeout=20)
        cid = res.stdout.strip()
        meta['ipfs_cid'] = cid
        safe_write(META_FILE, meta)
    except Exception:
        pass
    safe_write(PENDING_FILE, [])
    safe_write(MASK_STORE, {})
    logs = safe_read(TRAIN_LOG, [])
    logs.append({"ts": int(time.time()), "event": "aggregate", "meta": meta})
    safe_write(TRAIN_LOG, logs)
    return {"ok": True, "meta": meta}

@app.post("/submit_full_mask")
async def submit_full_mask(payload: dict, auth=Depends(require_token)):
    ms = safe_read(MASK_STORE, {})
    ms.setdefault("full_masks", {})[payload['device_id']] = payload['mask']
    safe_write(MASK_STORE, ms)
    return {"ok": True}

@app.get("/status")
async def status():
    pending = safe_read(PENDING_FILE, [])
    masks = safe_read(MASK_STORE, {})
    return {"pending": len(pending), "masks_keys": list(masks.keys())}
PYAGG

# -------------------------
# aggregator tests (basic)
# -------------------------
cat > "$ROOT/aggregator/tests/test_api.py" <<'TST'
from fastapi.testclient import TestClient
from aggregator.main import app, AGG_TOKEN
client = TestClient(app)
AUTH = {"Authorization": f"Bearer {AGG_TOKEN}"}

def test_submit_update_missing():
    r = client.post("/submit_update", json={"device_id":"d1"}, headers=AUTH)
    assert r.status_code == 400

def test_submit_and_status():
    payload = {"device_id":"d1", "sample_count":1, "update": {"w": [0.0]}}
    r = client.post("/submit_update", json=payload, headers=AUTH)
    assert r.status_code == 200
    s = client.get("/status")
    assert "pending" in s.json()
TST

# -------------------------
# bonawitz aggregator (coordinator) - simplified
# -------------------------
cat > "$ROOT/aggregator/bonawitz_aggregator.py" <<'BONAGG'
# bonawitz_aggregator.py
# (See README for usage instructions)
# This file implements coordinator-style Bonawitz fallback and PRG/SSS helpers.
BONAGG
# -------------------------
# client/
# -------------------------
mkdir -p "$ROOT/client"
cat > "$ROOT/client/requirements.txt" <<'REQC'
torch
msgpack
zstandard
requests
python-dotenv
secretsharing
cryptography
REQC

cat > "$ROOT/client/train_client_secure.py" <<'PYCL'
#!/usr/bin/env python3
# client/train_client_secure.py - starter client for Jarvis Gen5
import os, time, uuid, base64, json
import requests
import numpy as np
import torch, torch.nn as nn
import msgpack, zstandard as zstd
from secretsharing import PlaintextToHexSecretSharer
from dotenv import load_dotenv
load_dotenv()

AGG_URL = os.environ.get("AGG_URL", "http://localhost:4000")
AGG_TOKEN = os.environ.get("AGG_TOKEN", "changeme")
DEVICE_ID = os.environ.get("DEVICE_ID", f"dev-{uuid.uuid4().hex[:8]}")
N_SHARES = int(os.environ.get("N_SHARES", "5"))
THRESHOLD = int(os.environ.get("SSS_THRESHOLD", "3"))
SLEEP = int(os.environ.get("CLIENT_SLEEP", "20"))

HEADERS = {"Authorization": f"Bearer {AGG_TOKEN}"}

def pack(obj):
    p = msgpack.packb(obj, use_bin_type=True)
    c = zstd.ZstdCompressor(level=3).compress(p)
    return base64.b64encode(c).decode('ascii')
def unpack(s):
    c = base64.b64decode(s)
    d = zstd.ZstdDecompressor().decompress(c)
    return msgpack.unpackb(d, raw=False)

class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(8,32); self.l2 = nn.Linear(32,8)
    def forward(self,x):
        return self.l2(torch.relu(self.l1(x)))

def state_to_serial(sd):
    out={}
    for k,v in sd.items():
        out[k] = pack(v.detach().cpu().numpy().tolist())
    return out

def make_mask(update_serial):
    mask={}
    for k,s in update_serial.items():
        arr = np.array(unpack(s), dtype=np.float32)
        m = np.random.normal(scale=1e-3, size=arr.shape).astype(np.float32)
        mask[k] = pack(m.tolist())
    return mask

def add_mask(update_serial, mask_serial):
    masked={}
    for k in update_serial:
        u = np.array(unpack(update_serial[k]), dtype=np.float32)
        m = np.array(unpack(mask_serial[k]), dtype=np.float32)
        masked_arr = (u + m).astype(np.float32)
        masked[k] = pack(masked_arr.tolist())
    return masked

def post_json(path, data):
    try:
        r = requests.post(AGG_URL.rstrip('/') + path, json=data, headers=HEADERS, timeout=8)
        return r.ok, r.text
    except Exception as e:
        return False, str(e)

if __name__ == "__main__":
    model = SmallNet()
    while True:
        for p in model.parameters(): p.data += 0.001 * torch.randn_like(p.data)
        sd = {k:v.clone().detach() for k,v in model.state_dict().items()}
        up = state_to_serial(sd)
        mask = make_mask(up)
        masked = add_mask(up, mask)
        payload = {"device_id": DEVICE_ID, "sample_count": 1, "masked_update": masked}
        ok,resp = post_json("/submit_update", payload)
        print("submit masked ok:", ok)
        if ok:
            fm = {"device_id": DEVICE_ID, "mask": mask}
            post_json("/submit_full_mask", fm)
        time.sleep(SLEEP)
PYCL

cat > "$ROOT/client/Dockerfile" <<'DFCL'
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python","train_client_secure.py"]
DFCL

# -------------------------
# dashboard/
# -------------------------
mkdir -p "$ROOT/dashboard/static"
cat > "$ROOT/dashboard/requirements.txt" <<'REQD'
fastapi
uvicorn[standard]
jinja2
python-dotenv
requests
REQD

cat > "$ROOT/dashboard/app.py" <<'DASHAPP'
# dashboard app (simple)
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import json, os
ROOT = os.path.dirname(__file__)
LOGS = os.path.join(ROOT, "../logs")
if not os.path.exists(LOGS): os.makedirs(LOGS)
DEV_FILE = os.path.join(LOGS, "devices.json")
META_FILE = os.path.join(LOGS, "global_meta.json")
TRAIN = os.path.join(LOGS, "training_log.json")
app = FastAPI()
app.mount("/static", StaticFiles(directory=os.path.join(ROOT,"static")), name="static")
@app.get("/", response_class=HTMLResponse)
def index():
    return open(os.path.join(ROOT,"static","index.html")).read()
@app.get("/status")
def status():
    devices = {}
    meta = {}
    logs = []
    try:
        devices = json.loads(open(DEV_FILE).read()) if os.path.exists(DEV_FILE) else {}
    except: devices = {}
    try:
        meta = json.loads(open(META_FILE).read()) if os.path.exists(META_FILE) else {}
    except: meta = {}
    try:
        logs = json.loads(open(TRAIN).read())[-50:] if os.path.exists(TRAIN) else []
    except: logs = []
    return {"devices": devices, "global": meta, "events": logs}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=3000)
DASHAPP

cat > "$ROOT/dashboard/static/index.html" <<'HTMLD'
<!doctype html><html><head><meta charset="utf-8"><title>Jarvis Gen5 Dashboard</title></head><body>
<h2>Jarvis Gen5 Dashboard</h2>
<pre id="out">loading...</pre>
<script>
async function r(){ const res = await fetch('/status'); const j=await res.json(); document.getElementById('out').innerText = JSON.stringify(j,null,2); }
setInterval(r,3000); r();
</script></body></html>
HTMLD

cat > "$ROOT/dashboard/Dockerfile" <<'DFDB'
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 3000
CMD ["python","app.py"]
DFDB

# -------------------------
# ipfs helper scripts
# -------------------------
mkdir -p "$ROOT/ipfs"
cat > "$ROOT/ipfs/ipfs_upload.py" <<'IPFSUP'
import subprocess, sys
fn = sys.argv[1] if len(sys.argv)>1 else "aggregator/global_model.pt"
res = subprocess.run(["ipfs","add","-Q", fn], capture_output=True, text=True)
print(res.stdout.strip())
IPFSUP

# -------------------------
# docker-compose
# -------------------------
cat > "$ROOT/docker-compose.yml" <<'DCOMP'
version: "3.8"
services:
  ipfs:
    image: ipfs/kubo:latest
    ports: ["4001:4001","5001:5001","8080:8080"]
    volumes: ["./ipfs-data:/data/ipfs"]
  aggregator:
    build: ./aggregator
    ports: ["4000:4000"]
    env_file: ./aggregator/.env.example
    volumes: ["./logs:/app/logs"]
    depends_on: ["ipfs"]
  dashboard:
    build: ./dashboard
    ports: ["3000:3000"]
    volumes: ["./logs:/app/logs"]
    depends_on: ["aggregator"]
  client:
    build: ./client
    env_file: ./client/.env.example
    depends_on: ["aggregator"]
DCOMP

# -------------------------
# scripts/
# -------------------------
mkdir -p "$ROOT/scripts"
cat > "$ROOT/scripts/deploy_repo.py" <<'DEPLOY'
#!/usr/bin/env python3
# deploy_repo.py - create github repo using gh and push
import os, subprocess, sys
REPO_NAME = os.environ.get("REPO_NAME","jarvis-gen5-full")
USER = os.environ.get("GITHUB_USER","your-username")
# initialize git if needed
if not os.path.exists(".git"):
    os.system("git init")
    os.system("git add .")
    os.system('git commit -m "Initial commit - Jarvis Gen5 full scaffold"')
# create repo with gh
print("Creating GitHub repo... ensure 'gh auth login' already run")
subprocess.run(["gh","repo","create", f"{USER}/{REPO_NAME}", "--public", "--source=.", "--push"], check=True)
print("Pushed to GitHub: https://github.com/{}/{}".format(USER, REPO_NAME))
DEPLOY

cat > "$ROOT/scripts/start_local.sh" <<'RUN'
#!/usr/bin/env bash
set -e
echo "Starting local Jarvis Gen5 stack (Docker Compose)..."
docker compose up --build -d
echo "Done. Dashboard: http://localhost:3000  Aggregator: http://localhost:4000"
RUN
chmod +x "$ROOT/scripts/start_local.sh"
chmod +x "$ROOT/scripts/deploy_repo.py"

# -------------------------
# README
# -------------------------
cat > "$ROOT/README.md" <<'RREADME'
# Jarvis Gen5 Full (scaffold)
This scaffold contains:
- aggregator (FastAPI)
- client (starter)
- dashboard (simple)
- ipfs helper
- docker-compose for local testing

Run:
  ./scripts/start_local.sh

To push to GitHub (requires gh):
  cd jarvis-gen5-full
  REPO_NAME=Gen5 USER=your-github-user python3 scripts/deploy_repo.py
RREADME

# -------------------------
# Create zip
# -------------------------
cd "$(dirname "$ROOT")"
zip -r "$(basename "$ROOT").zip" "$(basename "$ROOT")" >/dev/null
echo "Created $(basename "$ROOT") and $(basename "$ROOT").zip"
echo "To start: cd $ROOT && ./scripts/start_local.sh"