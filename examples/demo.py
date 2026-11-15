#!/usr/bin/env python3
"""End-to-end demo against a running API (default http://localhost:8000).

Seeds the example library, queries the unknown sample, then changes the
bad-channel mask (channels 10-12, which empties STD-0006-Artifact-Only),
waits for the rebuild to switch, and queries again.

Usage:  python3 examples/demo.py [BASE_URL]
"""
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:8000"


def call(method, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method)
    if data:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read())


def show_query(label):
    _, q = call("POST", "/query", json.load(open(os.path.join(HERE, "unknown.json"))))
    print(f"\n{label} (scheme {q['scheme_id']}, mask {q['masked_channels']}):")
    for r in q["results"]:
        print(f"  {r['spectrum_id']:<28} cosine={r['score']:.6f}")


def main():
    print(f"demo against {BASE}")
    for fn in sorted(os.listdir(os.path.join(HERE, "spectra"))):
        doc = json.load(open(os.path.join(HERE, "spectra", fn)))
        _, res = call("POST", "/spectra", doc)
        print(f"ingested {res['spectrum_id']:<28} indexed={res['indexed']}")

    show_query("unknown sample, initial scheme")

    print("\nrequesting rebuild with bad channels [10, 11, 12] ...")
    _, reb = call("POST", "/scheme/rebuild", {"masked_channels": [10, 11, 12]})
    print(f"  -> candidate scheme {reb.get('scheme_id')}: {reb.get('status')}")
    for _ in range(200):
        _, schemes = call("GET", "/schemes")
        cand = [s for s in schemes["schemes"] if s["scheme_id"] == reb.get("scheme_id")]
        if cand and cand[0]["status"] != "building":
            print(f"  -> scheme {cand[0]['scheme_id']} is now {cand[0]['status']}")
            break
        time.sleep(0.2)

    show_query("unknown sample, after mask rebuild")
    _, active = call("GET", "/scheme/active")
    print(f"\nactive scheme: {active}")


if __name__ == "__main__":
    main()
