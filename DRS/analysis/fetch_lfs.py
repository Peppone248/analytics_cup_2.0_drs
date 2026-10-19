import json, subprocess, sys, time
from pathlib import Path

REPO = "https://github.com/Peppone248/analytics_cup_2.0_drs.git"
BATCH = REPO + "/info/lfs/objects/batch"

def pointer(p):
    d = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if line.startswith("oid sha256:"):
            d["oid"] = line.split(":", 1)[1].strip()
        elif line.startswith("size "):
            d["size"] = int(line.split()[1])
    return d

def href(oid, size):
    body = json.dumps({"operation": "download", "transfers": ["basic"],
                       "objects": [{"oid": oid, "size": size}]})
    r = subprocess.run(
        ["curl", "-s", "-X", "POST",
         "-H", "Accept: application/vnd.git-lfs+json",
         "-H", "Content-Type: application/vnd.git-lfs+json",
         "--data", body, BATCH],
        capture_output=True, text=True, check=True)
    o = json.loads(r.stdout)["objects"][0]
    if "error" in o:
        raise RuntimeError(f"{oid[:12]}: {o['error']}")
    return o["actions"]["download"]["href"]

src_root, dst_root = Path(sys.argv[1]), Path(sys.argv[2])
mids = sys.argv[3:]

for mid in mids:
    src = src_root / mid / f"{mid}_tracking_extrapolated.jsonl"
    dst = dst_root / mid / f"{mid}_tracking_extrapolated.jsonl"
    meta = pointer(src)
    if dst.exists() and dst.stat().st_size == meta["size"]:
        print(f"{mid}: già presente ({meta['size']/1e6:.0f} MB)", flush=True)
        continue
    dst.parent.mkdir(parents=True, exist_ok=True)
    t = time.time()
    subprocess.run(["curl", "-sS", "-L", "-o", str(dst), href(meta["oid"], meta["size"])],
                   check=True)
    got = dst.stat().st_size
    ok = "OK" if got == meta["size"] else f"ATTESI {meta['size']}"
    print(f"{mid}: {got/1e6:.0f} MB in {time.time()-t:.0f}s  {ok}", flush=True)
