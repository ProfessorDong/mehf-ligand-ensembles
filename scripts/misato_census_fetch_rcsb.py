"""Step 2 of the MISATO MMP feasibility census: fetch ligand identity + target identity.

The local MISATO artifacts carry NO SMILES (the cache's own summary.json lists
"no SMILES is available, so the smiles index column is empty" as a fidelity gap)
and no UniProt accession. Both are required for MMP matching and for grouping by
target. This script fetches them from the RCSB Data API GraphQL endpoint, keyed
by the 4-character PDB entry ID that MISATO uses as its system id.

Network only; writes a single JSONL cache under outputs/misato_mmp_census_v1/.
Resumable: re-running skips entries already in the cache.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/misato_mmp_census_v1"
CACHE_PATH = OUT / "rcsb_entries.jsonl"
ENDPOINT = "https://data.rcsb.org/graphql"

QUERY = """
query($ids:[String!]!){
  entries(entry_ids:$ids){
    rcsb_id
    struct{ title }
    polymer_entities{
      rcsb_polymer_entity_container_identifiers{ uniprot_ids }
      rcsb_polymer_entity{ pdbx_description }
      entity_poly{ rcsb_sample_sequence_length }
    }
    nonpolymer_entities{
      rcsb_nonpolymer_entity_container_identifiers{ auth_asym_ids }
      nonpolymer_comp{
        chem_comp{ id name formula_weight type }
        rcsb_chem_comp_descriptor{ SMILES SMILES_stereo }
      }
    }
  }
}
"""


def post(ids: list[str], attempt: int = 0) -> list[dict]:
    body = json.dumps({"query": QUERY, "variables": {"ids": ids}}).encode()
    req = urllib.request.Request(
        ENDPOINT, data=body,
        headers={"Content-Type": "application/json",
                 "User-Agent": "mdwm-misato-census/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            payload = json.load(r)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        if attempt >= 5:
            raise
        time.sleep(min(60, 2 ** attempt * 3))
        return post(ids, attempt + 1)
    if payload.get("errors") and not payload.get("data"):
        if attempt >= 5:
            raise RuntimeError(str(payload["errors"])[:400])
        time.sleep(min(60, 2 ** attempt * 3))
        return post(ids, attempt + 1)
    return [e for e in (payload.get("data", {}).get("entries") or []) if e]


def main() -> None:
    ids = [x.strip() for x in
           (OUT / "ids_traces_all.txt").read_text().split() if x.strip()]
    extra = [x.strip() for x in
             (OUT / "ids_coord_cache.txt").read_text().split() if x.strip()]
    wanted = sorted(set(ids) | set(extra))

    done: set[str] = set()
    if CACHE_PATH.exists():
        for line in CACHE_PATH.read_text().splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line)["rcsb_id"].upper())
                except Exception:
                    pass
    todo = [i for i in wanted if i.upper() not in done]
    print(f"wanted={len(wanted)} cached={len(done)} todo={len(todo)}", flush=True)

    batch = 50
    t0 = time.time()
    with CACHE_PATH.open("a") as fh:
        for k in range(0, len(todo), batch):
            chunk = todo[k:k + batch]
            entries = post(chunk)
            for e in entries:
                fh.write(json.dumps(e) + "\n")
            fh.flush()
            if (k // batch) % 10 == 0:
                el = time.time() - t0
                rate = (k + batch) / max(el, 1e-9)
                print(f"  {k + len(chunk)}/{len(todo)}  {el:.0f}s  "
                      f"{rate:.1f} ids/s  eta {(len(todo) - k) / max(rate, 1e-9) / 60:.1f}min",
                      flush=True)
            time.sleep(0.15)
    print(f"DONE in {time.time() - t0:.0f}s -> {CACHE_PATH}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
