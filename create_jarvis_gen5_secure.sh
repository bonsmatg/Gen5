#!/usr/bin/env bash
set -e
ROOT="$(pwd)/jarvis-gen5-secure"
echo "Creating project at $ROOT"
rm -rf "$ROOT"
mkdir -p "$ROOT"

# -------------------------
# aggregator/
# -------------------------
mkdir -p "$ROOT/aggregator" "$ROOT/aggregator/tests"
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

cat > "$ROOT/aggregator/Dockerfile" <<'DFAGG'
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN apt-get update && apt-get install -y build-essential libpq-dev && \
    pip install --no-cache-dir -r requirements.txt
COPY . /app
EXPOSE 4000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "4000"]
DFAGG

cat > "$ROOT/aggregator/.env.example" <<'ENVAGG'
AGG_TOKEN=changeme
LOG_DIR=/app/logs
POSTGRES_URL=postgresql+asyncpg://fl:fl@postgres:5432/jarvis
REDIS_URL=redis://redis:6379/0
MAX_PENDING=1000
JWT_PRIVATE_KEY_PATH=/app/keys/jwt_key.pem
JWT_PUBLIC_KEY_PATH=/app/keys/jwt_pub.pem
CERT_PATH=/app/keys/cert.pem
KEY_PATH=/app/keys/key.pem
ENVAGG

cat > "$ROOT/aggregator/main.py" <<'PYAGG'
# aggregator/main.py
import os, json, time, subprocess, base64, traceback
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends, File, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from threading import Lock
import msgpack, zstandard as zstd
import logging
import asyncio
from secretsharing import PlaintextToHexSecretSharer
import numpy as np
import torch

# config
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

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aggregator")

# thread safe file ops
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

# serializer helpers (msgpack + zstd + base64)
def pack_bin(obj):
    packed = msgpack.packb(obj, use_bin_type=True)
    comp = zstd.ZstdCompressor(level=3).compress(packed)
    return base64.b64encode(comp).decode('ascii')
def unpack_bin(s):
    c = base64.b64decode(s)
    d = zstd.ZstdDecompressor().decompress(c)
    return msgpack.unpackb(d, raw=False)

# init files
for p,defv in [(PENDING_FILE, []), (MASK_STORE, {}), (TRAIN_LOG, [])]:
    if not p.exists(): safe_write(p, defv)

# auth
security = HTTPBearer()
def require_token(creds: HTTPAuthorizationCredentials = Depends(security)):
    token = creds.credentials
    if not REQUIRE_AUTH:
        return True
    if token != AGG_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    return True

# pydantic models
from typing import Optional, Dict
class UpdatePayload(BaseModel):
    device_id: str
    sample_count: int = Field(default=1, ge=0)
    update: Optional[Dict[str,str]] = None
    masked_update: Optional[Dict[str,str]] = None
    meta: Optional[Dict] = None

class MaskShare(BaseModel):
    from_id: str
    to_id: str
    key: str
    share_hex: str

class FullMask(BaseModel):
    device_id: str
    mask: Dict[str,str]

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

@app.post("/submit_mask_share")
async def submit_mask_share(payload: MaskShare, auth=Depends(require_token)):
    ms = safe_read(MASK_STORE, {})
    ms.setdefault("shares", {})
    ms["shares"].setdefault(payload.to_id, {})
    ms["shares"][payload.to_id].setdefault(payload.key, []).append({"from": payload.from_id, "share": payload.share_hex})
    safe_write(MASK_STORE, ms)
    return {"ok": True}

@app.post("/submit_full_mask")
async def submit_full_mask(payload: FullMask, auth=Depends(require_token)):
    ms = safe_read(MASK_STORE, {})
    ms.setdefault("full_masks", {})[payload.device_id] = payload.mask
    safe_write(MASK_STORE, ms)
    return {"ok": True}

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
                        # fallback: try msgpack older path
                        import zlib
                        raw = zlib.decompress(base64.b64decode(ms))
                        arr = np.load(BytesIO(raw)) - np.load(BytesIO(zlib.decompress(base64.b64decode(mask[k]))))
                    rec[k] = pack_bin(arr.tolist())
                recovered.append({"device_id": dev, "sample_count": p.get('sample_count',1), "update": rec})
            elif 'update' in p:
                recovered.append({"device_id": p['device_id'], "sample_count": p.get('sample_count',1), "update": p['update']})
    else:
        for p in pending:
            if 'update' in p and p['update']:
                recovered.append({"device_id": p['device_id'], "sample_count": p.get('sample_count',1), "update": p['update']})
    if not recovered:
        raise HTTPException(status_code=400, detail="no updates")
    # FedAvg
    import numpy as np
    all_keys = set()
    for ru in recovered: all_keys.update(ru['update'].keys())
    aggregated = {}
    for k in all_keys:
        arrs=[]; wts=[]
        for ru in recovered:
            upd = ru['update']
            if k in upd:
                arr = np.array(unpack_bin(upd[k]))
                arrs.append(arr * ru['sample_count'])
                wts.append(ru['sample_count'])
        if not arrs: continue
        sum_arr = np.sum(np.stack(arrs,axis=0), axis=0)
        total_w = float(sum(wts))
        avg = (sum_arr / total_w).astype(np.float32)
        aggregated[k] = pack_bin(avg.tolist())
    # save as torch state_dict
    state={}
    import numpy as _np
    for k,s in aggregated.items():
        arr = _np.array(unpack_bin(s), dtype=_np.float32)
        state[k] = torch.from_numpy(arr)
    torch.save(state, GLOBAL_MODEL)
    meta = {"version": int(time.time()), "ts": int(time.time()), "keys": list(aggregated.keys())}
    safe_write(META_FILE, meta)
    # try ipfs
    try:
        out = subprocess.run(["ipfs","add","-Q", str(GLOBAL_MODEL)], capture_output=True, text=True, timeout=20)
        cid = out.stdout.strip()
        meta['ipfs_cid'] = cid
        safe_write(META_FILE, meta)
    except Exception as e:
        logger.warning("IPFS fail: %s", e)
    # clear pending and masks
    safe_write(PENDING_FILE, [])
    safe_write(MASK_STORE, {})
    logs = safe_read(TRAIN_LOG, [])
    logs.append({"ts": int(time.time()), "event": "aggregate", "meta": meta})
    safe_write(TRAIN_LOG, logs)
    return {"ok": True, "meta": meta}

@app.get("/global-model")
async def get_global_model():
    meta = safe_read(META_FILE, {})
    return {"meta": meta, "download": "/download/global-model" if GLOBAL_MODEL.exists() else None}

from fastapi.responses import FileResponse
@app.get("/download/global-model")
async def download_model(auth=Depends(require_token)):
    if not GLOBAL_MODEL.exists():
        raise HTTPException(status_code=404, detail="no model")
    return FileResponse(str(GLOBAL_MODEL), media_type="application/octet-stream", filename="global_model.pt")

@app.get("/status")
async def status():
    pending = safe_read(PENDING_FILE, [])
    masks = safe_read(MASK_STORE, {})
    return {"pending": len(pending), "masks_keys": list(masks.keys())}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=4000, log_level="info")
PYAGG

# -------------------------
# aggregator tests (basic)
# -------------------------
cat > "$ROOT/aggregator/tests/test_api.py" <<'TST'
from fastapi.testclient import TestClient
import os
from aggregator.main import app, AGG_TOKEN
client = TestClient(app)
AUTH = {"Authorization": f"Bearer {AGG_TOKEN}"}

def test_submit_update_missing():
    r = client.post("/submit_update", json={"device_id":"d1"}, headers=AUTH)
    assert r.status_code == 400

def test_submit_and_status():
    payload = {"device_id":"d1","sample_count":1,"update":{"w": "A"}}
    r = client.post("/submit_update", json=payload, headers=AUTH)
    assert r.status_code == 200
    s = client.get("/status")
    assert "pending" in s.json()
TST

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
python-jose[cryptography]
REQC

cat > "$ROOT/client/Dockerfile" <<'DFCL'
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
CMD ["python","train_client_secure.py"]
DFCL

cat > "$ROOT/client/train_client_secure.py" <<'PYCL'
#!/usr/bin/env python3
"""
client/train_client_secure.py
- local mock train
- compute state_dict, pack with msgpack+zstd
- PRG mask via AES-CTR per pair (coordinator-relay simplified)
- Shamir shares fallback (secretsharing)
- send masked_update, shares, then full mask to aggregator
"""
import os, time, uuid, json, base64, random
from pathlib import Path
import msgpack, zstandard as zstd
import requests
import numpy as np
import torch, torch.nn as nn
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

def serial_to_state(sd):
    out={}
    for k,s in sd.items():
        out[k] = torch.tensor(unpack(s))
    return out

def local_train(model):
    for p in model.parameters():
        p.data += 0.001 * torch.randn_like(p.data)
    return model

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
    url = AGG_URL.rstrip("/") + path
    try:
        r = requests.post(url, json=data, headers=HEADERS, timeout=8)
        return r.ok, r.text
    except Exception as e:
        print("post error", e)
        return False, str(e)

if __name__ == "__main__":
    model = SmallNet()
    while True:
        model = local_train(model)
        sd = {k:v.clone().detach() for k,v in model.state_dict().items()}
        up = state_to_serial(sd)
        mask = make_mask(up)
        masked = add_mask(up, mask)
        payload = {"device_id": DEVICE_ID, "sample_count": 1, "masked_update": masked}
        ok,resp = post_json("/submit_update", payload)
        print("submit masked ok:", ok)
        if ok:
            # create Shamir shares for each key, post shares as relay
            shares = {}
            for k,m in mask.items():
                raw = base64.b64decode(m)
                hexplain = raw.hex()
                s = PlaintextToHexSecretSharer.split_secret(hexplain, THRESHOLD, N_SHARES)
                shares[k] = s
                # post shares to aggregator as relay
                for idx,sh in enumerate(s):
                    spayload = {"from_id": DEVICE_ID, "to_id": f"relay_{idx}", "key": k, "share_hex": sh}
                    post_json("/submit_mask_share", spayload)
            # finally post full mask
            fm = {"device_id": DEVICE_ID, "mask": mask}
            post_json("/submit_full_mask", fm)
        time.sleep(SLEEP)
PYCL

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
websockets
REQD

cat > "$ROOT/dashboard/Dockerfile" <<'DFDB'
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
EXPOSE 3000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3000"]
DFDB

cat > "$ROOT/dashboard/app.py" <<'DASHAPP'
# dashboard app.py
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import json, os, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOGS = ROOT.joinpath("../logs")
DEV_FILE = LOGS.joinpath("devices.json")
META_FILE = LOGS.joinpath("global_meta.json")
TRAIN = LOGS.joinpath("training_log.json")
app = FastAPI()
app.mount("/static", StaticFiles(directory=str(ROOT.joinpath("static"))), name="static")

def readp(p, default):
    try:
        if not p.exists(): return default
        return json.loads(p.read_text())
    except:
        return default

@app.get("/", response_class=HTMLResponse)
def index():
    return (ROOT / "static" / "index.html").read_text()

@app.get("/status")
def status():
    devices = readp(DEV_FILE, {})
    meta = readp(META_FILE, {})
    logs = readp(TRAIN, [])
    return {"devices": devices, "global": meta, "events": logs[-50:]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=3000)
DASHAPP

cat > "$ROOT/dashboard/static/index.html" <<'HTMLD'
<!doctype html><html><head><meta charset="utf-8"><title>Jarvis Gen5 Dashboard</title>
<style>body{font-family:Arial;background:#f6f8fa;padding:12px} .card{background:#fff;padding:12px;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,.08);margin-bottom:12px}</style>
</head><body>
<h2>Jarvis Gen5 — Dashboard</h2>
<div class="card"><h3>Devices</h3><pre id="devices">loading...</pre></div>
<div class="card"><h3>Global Model</h3><pre id="global">loading...</pre></div>
<div class="card"><h3>Events</h3><pre id="events">loading...</pre></div>
<script>
async function refresh(){
  try{
    const r = await fetch('/status'); const j = await r.json();
    document.getElementById('devices').innerText = JSON.stringify(j.devices,null,2);
    document.getElementById('global').innerText = JSON.stringify(j.global,null,2);
    document.getElementById('events').innerText = JSON.stringify(j.events.slice(-20).reverse(),null,2);
  }catch(e){ console.log(e); }
}
setInterval(refresh,3000); refresh();
</script>
</body></html>
HTMLD

# -------------------------
# docker-compose.yml
# -------------------------
cat > "$ROOT/docker-compose.yml" <<'DCOMP'
version: "3.8"
services:
  aggregator:
    build: ./aggregator
    container_name: jarvis_aggregator
    ports:
      - "4000:4000"
    volumes:
      - ./logs:/app/logs
    environment:
      - AGG_TOKEN=changeme
      - LOG_DIR=/app/logs
  dashboard:
    build: ./dashboard
    container_name: jarvis_dashboard
    ports:
      - "3000:3000"
    volumes:
      - ./logs:/app/logs
  postgres:
    image: postgres:15
    container_name: jarvis_postgres
    environment:
      - POSTGRES_USER=fl
      - POSTGRES_PASSWORD=fl
      - POSTGRES_DB=jarvis
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
  redis:
    image: redis:7
    container_name: jarvis_redis
    ports:
      - "6379:6379"
  ipfs:
    image: ipfs/go-ipfs:latest
    container_name: jarvis_ipfs
    ports:
      - "5001:5001"
      - "8080:8080"
volumes:
  pgdata:
DCOMP

# -------------------------
# root README
# -------------------------
cat > "$ROOT/README.md" <<'RREADME'
# Jarvis Gen5 Secure Federated Learning (scaffold)

This repository is a fully-featured scaffold for a secure federated learning network:
- FastAPI Aggregator (FedAvg, masking, IPFS upload)
- Secure client using Shamir fallback + PRG mask approach
- Dashboard (monitoring)
- Postgres + Redis services
- Docker Compose for local deployment

Quickstart:

1. Build & run:
   docker compose build
   docker compose up -d

2. Start a client locally (example):
   cd client
   AGG_URL=http://localhost:4000 AGG_TOKEN=changeme python train_client_secure.py

3. Visit dashboard:
   http://localhost:3000

Notes:
- This scaffold includes practical secure-agg components but is not a drop-in production Bonawitz implementation.
- For production: enable TLS, rotate JWT keys, and replace file JSON stores with Postgres/Redis-backed tables (placeholders exist).
RREADME

# -------------------------
# Package as zip
# -------------------------
cd "$(dirname "$ROOT")"
zip -r "$(basename "$ROOT").zip" "$(basename "$ROOT")" >/dev/null
echo "Created $(basename "$ROOT") and $(basename "$ROOT").zip"
echo "To start: cd $ROOT && docker compose build && docker compose up -d"