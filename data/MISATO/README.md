# MISATO molecular-dynamics data store

A permanent, self-contained copy of the MISATO protein–ligand MD archive plus
derived per-system trajectories, so that future work never has to re-download
132.8 GB or re-stream it over HTTP.

Everything in here is reproducible from `raw/MD.hdf5` alone. Nothing in here
depends on anything outside this directory except three read-only inputs from
the wider repo (the leakage exclusion list, the MISATO split id files, and the
MMP census), each of which is named and hashed in the manifest summaries.

---

## 0. What is actually in here right now

Built 2026-09-17. Raw archive 124 GiB; derived data 9.0 GiB.

| config | requested | written | rejected | excluded (leakage) | size | wall |
|---|---|---|---|---|---|---|
| `frames100_pairs_v1` | 4,768 | **4,611** | 11 | 146 | 1.94 GiB | 520 s |
| `frames100_all_v1` | 16,971 | **16,536** | 110 | 325 | 7.05 GiB | 1,961 s |

Both balance exactly (written + rejected + excluded = requested) and both were
produced by extractor `extract_system/v4`, config hash `e9ea2b3a91ff50c1`, at
100 frames / stride 1 / 8 Å pocket. Mean file 440–447 KiB. Peak worker-group RSS
1.17 GiB against a 24 GiB budget; no watchdog trip; zero transient failures.

`frames100_pairs_v1` is the MMP-pair priority corpus (union of `pdb_a`/`pdb_b`
in the census). It is a **strict subset** of `frames100_all_v1`, which duplicates
about 1.9 GiB. Keep it if a ready-made pair directory is convenient; delete it to
reclaim the space, since the pair id list is recoverable from the census at any
time and re-extraction takes 9 minutes.

**Rejections (full corpus, 110):** 82 ligand above the 300-atom cap, 26
disconnected ligand bond graph (periodic-boundary damage or multi-fragment), 2
ligand below the 6-atom minimum. Each is recorded per system with its reason.

**QC over the 16,536 written systems:**

| flag | count | share |
|---|---|---|
| clash warning (<1.2 Å) | **0** | — |
| bond-length warning | **1** (6D3Y) | 0.006 % |
| pocket truncated at the 2000-atom cap | **0** | — |
| ligand boundary independently confirmed | 15,426 | 93.3 % |
| `ligand_looks_peptidic` | 1,108 | 6.7 % |
| `periodic_image_jump_suspected` | 1,689 | 10.2 % |
| — of those, a discrete imaging jump (median step < 2 Å) | 1,510 | 9.1 % |
| — of those, genuine diffusion/unbinding (median step ≥ 2 Å) | 179 | 1.1 % |
| **clean for small-molecule dynamics** (neither peptidic nor jump-flagged) | **13,756** | 83.2 % |

`pocket_truncated_at_cap = 0` everywhere means the union-pocket superset
guarantee (§3) holds for every system in the store, so the caveat there is
currently vacuous — good, but re-check it if you re-extract with a smaller cap.

6D3Y is the one bond-length warning: its worst bond over all frames is 2.79 Å
while frames 0 and 99 (which the accept rule uses) stay at 2.53 Å, so an
intermediate frame stretches a bond past the 2.6 Å threshold. It is kept and
flagged.

**Validation actually run** (all in `validation/`):

| check | scope | result |
|---|---|---|
| publisher MD5 | whole 124 GiB file | match, plus independent `md5sum -c` = OK |
| gate, re-extracted from the local HDF5 | 40 systems, 360 windows | **0.0** on all 8 keys, both code paths |
| from stored `.npz`, pairs config | 270 systems, 2,430 windows | **0.0** on all 8 keys |
| from stored `.npz`, full config | 496 systems, 4,449 windows | **0.0** on all 8 keys |
| rejection path | 4 reason classes | 3 verbatim, 1 explained (§7) |
| resume after `kill -9` | both stages | content-identical to a clean run |
| concurrent writers on one config | 4 races, 304 writes | 0 corrupt, 0 duplicate, 0 torn |
| idempotency | both configs re-run | 0 pending, all 21,147 files sha256-verified |

The "from stored `.npz`" rows are the important ones: they read the files the
bulk run actually wrote and never reopen the archive, so a torn or raced write
would show up rather than being masked by recomputation.

---

## 1. Layout

```
data/MISATO/
  raw/
    MD.hdf5                        the publisher's file, byte-exact (see §2)
    MD.hdf5.md5                    published vs locally computed MD5, with provenance
    MD.hdf5.md5.txt                md5sum-compatible line: `md5sum -c MD.hdf5.md5.txt`
    download_state.json            chunk ledger; makes the download resumable
    zenodo_record_7711953.json     Zenodo API snapshot taken at download time
  extracted/
    <config-name>/<SYSTEM>.npz     one file per system, all frames (see §3)
  manifest/
    <config-name>/
      systems/<SYSTEM>.json        per-system status + QC + SHA-256  (durable state)
      manifest.jsonl               all per-system records, concatenated (derived)
      summary.json                 counts, QC aggregates, config hash   (derived)
      run_state.json               live heartbeat of a running extraction
  validation/
    crossvalidation_local.json             gate: re-extracted from the local HDF5
    crossvalidation_from_store_pairs.json  stored .npz vs the cache (pairs)
    crossvalidation_from_store_all.json    stored .npz vs the cache (full corpus)
    crossvalidation_remote_smoke.json      extractor checked over HTTP pre-download
    rejection_crossvalidation.json         the reject path vs the cache
    resume_test.json                       hard-kill resume + guard evidence
    pipeline_reconciliation.json           build history: why two pipelines ran
  README.md                        this file
```

`manifest/<config>/systems/*.json` is the only durable bookkeeping. `manifest.jsonl`
and `summary.json` are rebuilt from it on every run, so they can never disagree
with it.

---

## 2. Provenance and integrity

| | |
|---|---|
| Source | Zenodo record **7711953**, DOI `10.5281/zenodo.7711953`, version 1.0.0, published 2023-05-24 |
| Title | *MISATO — Machine learning dataset for structure-based drug discovery* |
| File | `MD.hdf5` |
| Size | 132,841,014,019 bytes (123.72 GiB) |
| Publisher MD5 | `9bc6446922cd80e0f2f3f69349bf88ed` (from the Zenodo API, not from a README) |
| Download URL | `https://zenodo.org/api/records/7711953/files/MD.hdf5/content` |

`7711953` is the only version of concept record `7711952`; there is no later
release to prefer.

**How integrity is established.** A size match is not evidence — a stream that
drops and restarts at the wrong offset produces a file of exactly the right
length. The downloader therefore:

1. reads the size *and* MD5 from the Zenodo API (never from a hardcoded value),
   and snapshots the whole API record next to the data;
2. writes into `MD.hdf5.part`, never into the final name;
3. computes the MD5 of the assembled file with a streaming 8 MiB buffer;
4. `os.replace`s `MD.hdf5.part` → `MD.hdf5` **only if** the digest matches.

So the presence of a file named `MD.hdf5` is itself a checksum-verified claim.
A mismatch leaves the file as `.part` and exits non-zero. To re-check at any
time:

```bash
cd data/MISATO/raw && md5sum -c MD.hdf5.md5.txt        # independent tool
python scripts/misato_store_download.py --force-reverify   # re-reads and rewrites the sidecar
```

---

## 3. What a `<SYSTEM>.npz` contains

One file per MISATO system (PDB id), holding **all stored frames**, with
**raw, unaligned** coordinates in ångström exactly as MISATO ships them.

| key | shape | dtype | meaning |
|---|---|---|---|
| `ligand_coords` | (F, L, 3) | float32 | ligand atom coordinates, every stored frame |
| `pocket_coords` | (F, P, 3) | float32 | pocket heavy-atom coordinates, every stored frame |
| `ligand_atomic_numbers` | (L,) | int16 | true atomic numbers (from `atoms_number`) |
| `pocket_atomic_numbers` | (P,) | int16 | true atomic numbers |
| `ligand_atom_indices` | (L,) | int32 | indices into the system's full atom axis |
| `pocket_atom_indices` | (P,) | int32 | indices into the system's full atom axis |
| `pocket_min_distance_over_frames` | (P,) | float32 | each pocket atom's closest approach to the ligand |
| `bond_index` | (B, 2) | int64 | ligand bonds, inferred geometrically (see §7) |
| `product_features` | (L, 6) | float32 | the project's ligand feature block |
| `edit_features` | (L, 5) | float32 | in-support no-edit block (MISATO carries no chemical edit) |
| `pocket_features` | (P, 2) | float32 | `[is_heavy, mass/200]` |
| `frame_indices` | (F,) | int32 | which archive frames these are |
| `molecules_begin_atom_index` | (M,) | int64 | MISATO's molecule boundaries, copied through |
| `n_system_atoms` | scalar | int64 | atoms in the full system |
| `n_stored_frames_in_archive` | scalar | int64 | frames the archive holds (100 for every system) |
| `ligand_atoms_element`, `pocket_atoms_element` | (L,) / (P,) | int64 | MISATO's internal element *index* — **not** an element (see §9) |
| `ligand_atoms_residue`, `pocket_atoms_residue` | (L,) / (P,) | int64 | MISATO residue code; ligand atoms are code `0` |
| `ligand_atoms_type`, `pocket_atoms_type` | (L,) / (P,) | int64 | MISATO force-field atom type index |
| `frames_bSASA` | (F,) | float32 | MISATO's buried SASA per frame |
| `frames_distance` | (F,) | float32 | MISATO's per-frame distance observable |
| `frames_interaction_energy` | (F,) | float32 | MISATO's per-frame interaction energy |
| `frames_rmsd_ligand` | (F,) | float32 | MISATO's own ligand RMSD series (see caveat below) |

The last seven groups are MISATO's own auxiliary datasets, carried through
verbatim. They are a few kilobytes per system and cannot be recomputed from
coordinates, so keeping them here means a future reader never has to reopen the
124 GiB archive for them. Any that a system happens to lack are simply absent
from its `.npz`, and `qc.auxiliary_datasets_present` lists what was found.

**Caveat on `frames_rmsd_ligand`.** It is *not* referenced to trajectory frame 0
— its frame-0 value is 0.93 Å on 5WIJ, 1.93 Å on 4K6W and 5.74 Å on 1KTI, not 0
— so it cannot be reproduced from the stored coordinates, and it correlates with
no frame-0-referenced RMSD computable here (measured |r| < 0.33 for
raw-displacement, pocket-superposed and ligand-superposed variants on those
three systems). Treat it as MISATO's own quantity against MISATO's own
reference, not as a ligand RMSD you can re-derive.

**Pocket definition.** `pocket_coords` holds every protein *heavy* atom that
comes within `pocket_radius_angstrom` (8.0 Å by default) of any ligand atom in
*any* stored frame — a union over frames, not a single-frame selection. That
makes it a superset of any per-frame pocket, so a downstream window can reselect
its own anchor-frame pocket from the stored arrays and get exactly what a
from-scratch extraction would give. If the union exceeds `--max-pocket-atoms`
(2000) the closest are kept (by minimum-over-frames distance) and
`qc.pocket_truncated_at_cap` is set.

That cap is the one place where the superset guarantee can fail: for a truncated
system, an atom that a from-scratch extraction would have selected at some
anchor frame could in principle have been dropped, if more than 2000 other atoms
were closer to the ligand across the trajectory as a whole. Count the affected
systems with `qc_flag_counts.pocket_truncated_at_cap` in `summary.json`; for
those systems, and only those, go back to `raw/MD.hdf5` if you need an exact
per-frame pocket, or re-extract that configuration with a larger
`--max-pocket-atoms`.

**Why 100 frames and not 11-frame windows.** The older cache
(`data/processed/misato_atomic_window_cache_3k_v1`) stored 6 context + 5 target
frames per window. That was a caching choice; MISATO stores 100 frames for every
system. A prior MISATO pretraining run produced an almost flat learning curve
(val MAE 0.5629 → 0.5581 over 20 epochs), and a five-frame prediction horizon is
a plausible cause. Storing every frame makes the horizon a free parameter
downstream, for about 1 MB per system.

---

## 4. Re-cutting windows without touching the HDF5

This is the point of the store: a different window length is a slice, not a
re-extraction.

**Option A — the project's existing window convention**, reproduced exactly
(pocket chosen at each window's anchor frame, all frames Kabsch-superposed onto
that anchor's pocket heavy atoms):

```python
import numpy as np, sys
sys.path.insert(0, "scripts")
from misato_store_extract import windows_from_system

with np.load("data/MISATO/extracted/frames100_pairs_v1/11GS.npz") as archive:
    stored = {key: archive[key] for key in archive.files}

# 20 context frames and a 40-frame horizon instead of 6 and 5:
windows = windows_from_system(
    stored, context_frames=20, future_frames=40, frame_stride=1, window_stride=10
)
# each window: product_init (L,3), target_delta (40,L,3), context_ligand_coords
# (20,L,3), pocket_init (P',3), pocket_features, product_features, edit_features,
# bond_index, start_frame
```

`windows_from_system(stored, context_frames=6, future_frames=5, frame_stride=1,
window_stride=10, pocket_radius_angstrom=8.0, max_pocket_atoms=600)` reproduces
the old 11-frame cache **bit for bit** — that is what
`validation/crossvalidation*.json` measures.

**Option B — your own convention.** The stored coordinates are raw, so nothing
is baked in:

```python
ligand = stored["ligand_coords"]    # (100, L, 3), raw angstrom
pocket = stored["pocket_coords"]    # (100, P, 3), raw angstrom
context, horizon = ligand[:60], ligand[60:]          # no alignment at all
# or align frames onto frame 0 using whatever atom set you prefer
```

Frames are 8 ns / 100 ≈ 80 ps apart in the MISATO MD protocol; consult the
MISATO paper before relying on the absolute time step.

---

## 5. Re-extracting from the raw file

Use a **new** `--config-name` whenever extraction settings change. Settings that
affect the output are hashed into `config_hash`; if you reuse a name with
different settings, every system is correctly treated as stale and re-extracted,
but the outputs of the two configurations would then share a directory, which
you do not want.

```bash
# the MMP-pair corpus at 100 frames (the default configuration)
CUDA_VISIBLE_DEVICES="" /home/ieb24002/anaconda3/envs/MDWM/bin/python \
  scripts/misato_store_extract.py \
  --config-name frames100_pairs_v1 --systems census-pairs --workers 12

# every system MISATO ships
CUDA_VISIBLE_DEVICES="" ... --config-name frames100_all_v1 --systems all --workers 12

# an explicit id list, a wider pocket, stride 2
CUDA_VISIBLE_DEVICES="" ... --config-name frames50_stride2_r10_v1 \
  --systems my_ids.txt --frames 50 --frame-stride 2 --pocket-radius-angstrom 10
```

`--systems` takes `all`, `census-pairs` (the union of `pdb_a`/`pdb_b` in
`outputs/misato_mmp_census_v1/mmp_pairs.csv`), or a path to a newline-separated
id list. It is never hardcoded. `--dry-run` prints the plan, the config hash and
the memory arithmetic without extracting.

### Verifying an existing store

`--dry-run` is also the whole-store integrity check: it re-hashes every output
and compares against the manifest, then reports what it would redo. A clean
store prints `N already complete {'written (sha256 verified)': N}, 0 pending`.

```bash
CUDA_VISIBLE_DEVICES="" /home/ieb24002/anaconda3/envs/MDWM/bin/python \
  scripts/misato_store_extract.py --config-name frames100_pairs_v1 \
  --systems census-pairs --dry-run
```

Anything that has been damaged since it was written shows up as `pending`, and a
normal (non-dry) run repairs it.

---

## 6. Operational properties

### Resumability after a hard kill

Both stages survive `kill -9` of the whole process group and continue rather
than restart.

* **Download.** `download_state.json` is a ledger of 512 MiB chunks. A chunk is
  marked done only *after* `os.fsync`, so the ledger can lose work but can never
  claim work it did not do; a restart re-fetches only unmarked chunks. Verified:
  the same downloader was `kill -9`-ed 12 s into a 343 MB file from the same
  Zenodo record, restarted, and produced a file whose MD5 matched the
  publisher's exactly (`md5sum -c` agreed).
* **Extraction.** Completeness is decided by **content**, never by existence.
  A system counts as done only when its manifest entry says `written`, carries
  the current `config_hash`, records the requested frame count, and its `.npz`
  still has the recorded byte size *and* SHA-256. `--trust-manifest` relaxes
  only the hash. A kill mid-write leaves `<id>.npz.tmp<pid>`, which no check
  ever looks at and which the next run sweeps; a kill after the write but before
  the manifest leaves an unclaimed `.npz` that is simply redone.

  This is deliberately unlike `scripts/run_manifest_md.py`, whose "output file
  exists and is non-empty" test was demonstrated to accept truncated 60 ps runs
  as complete 1 ns runs — a killed run is *exactly* what leaves a non-empty file
  behind. Verified here: truncating a finished `.npz` to 60 % of its length left
  it non-empty, and the next run detected it and re-extracted it.

  Verified end to end: a run was `kill -9`-ed after 13 of 76 systems, resumed,
  and the resulting store was **array-for-array identical** (max absolute
  deviation 0.0 on all 15 keys of all 76 files) to a clean single-pass
  extraction. See `validation/resume_test.json`.

  Note that the recorded SHA-256 verifies that a file has not changed since it
  was written; it is not a determinism claim about `np.savez_compressed`, whose
  zip metadata embeds a timestamp. Content determinism is checked by comparing
  arrays, as above.

### Liveness

Do not infer liveness from a non-empty log — a log keeps its last line forever
after the process dies. Check the heartbeat the process writes about itself:

```bash
# extraction
python - <<'PY'
import json, os, time, datetime
s = json.load(open("data/MISATO/manifest/frames100_pairs_v1/run_state.json"))
alive = os.path.exists(f"/proc/{s['pid']}")
age = (datetime.datetime.now(datetime.timezone.utc)
       - datetime.datetime.strptime(s["heartbeat_utc"], "%Y-%m-%dT%H:%M:%SZ")
         .replace(tzinfo=datetime.timezone.utc)).total_seconds()
print("pid", s["pid"], "alive:", alive, "heartbeat age:", round(age), "s", s["progress"])
PY

# download: same idea on raw/download_state.json (heartbeat_pid, updated_utc)
```

A fresh heartbeat (seconds, not minutes) from a pid that still exists is the
only evidence of a live run.

### Resource guards

Nothing here touches a GPU. `CUDA_VISIBLE_DEVICES` is cleared before numpy or
h5py are imported, the extractor refuses to start if `torch` is in
`sys.modules`, and neither stage opens an nvidia device node.

Extraction memory is bounded by construction rather than by hope. Per worker:

```
protein frame block   --frame-block(10) x --max-system-atoms(200k) x 3 x 4 B = 24.0 MB
distance temporary    --atom-block(4096) x --max-ligand-atoms(300) x 8 B      =  9.8 MB
ligand trajectory     --frames(100)  x 300 atoms x 3 x 4 B                    =   0.4 MB
pocket trajectory     --frames(100)  x --max-pocket-atoms(2000) x 3 x 4 B     =   2.4 MB
HDF5 chunk cache      --rdcc-mib(32)                                          =  33.6 MB
                                                                    worst case ~ 70 MB
plus a CPython + numpy interpreter                                            ~ 200 MB
=> 12 workers x ~270 MB ~= 3.2 GiB
```

Those are worst cases at the configured caps, not typicals; the measured peak
worker-group RSS is recorded in `summary.json` under
`runtime_config.peak_group_rss_bytes` for every run.

A watchdog thread samples every 5 s and **pauses** the pool, then stops it
cleanly after `--watchdog-grace-seconds` (300 s) if any threshold is still
crossed:

| guard | default | why |
|---|---|---|
| worker-group RSS | `--rss-budget-gib 24` | ~7× the arithmetic above; a leak trips it long before the box notices |
| system `MemAvailable` | `--min-available-gib 100` | this machine is shared; leave the co-tenant a large floor |
| free disk (extraction) | `--disk-floor-gib 25` | outputs are ~1 MB/system, so this is pure headroom |
| free disk (download) | `--disk-floor-gib 40`, checked against *remaining bytes* | free space here has swung 325 GB within an hour |

The download's pre-flight refuses to start unless
`free ≥ remaining_bytes + floor`, and re-checks every report interval; a
violation parks the workers (so no further bytes land) and, if it does not
recover within `--max-pause-minutes` (120), exits cleanly with the ledger
persisted. Verified: an impossible `--min-available-gib 2000` made the extractor
pause immediately, stop after its grace period, exit 3, record
`watchdog_stop_reason` in `summary.json`, and leave every written output
sha256-clean with zero orphan temporary files.

### Durability

Run both stages under `setsid` with output to `runlogs/`, so they survive the
end of a shell session:

```bash
cd /home/ieb24002/MDWM_TCBB
setsid env CUDA_VISIBLE_DEVICES="" /home/ieb24002/anaconda3/envs/MDWM/bin/python \
  scripts/misato_store_download.py --workers 8 \
  > runlogs/misato_store_download_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
```

---

## 7. Validation

`validation/crossvalidation*.json` is the correctness evidence. It samples
systems from `data/processed/misato_atomic_window_cache_3k_v1` (26,686 windows
over 2,980 systems, built by this repo's own pipeline) and re-derives the same
windows **two independent ways** from the raw HDF5:

* **reference path** — `mdwm.data.misato_atomic.extract_misato_windows`, the
  exact function that produced that cache. Disagreement here means the
  downloaded bytes or the local read path differ from what the cache was built
  from. Held to bit-equality.
* **store path** — `misato_store_extract.extract_system` (100 raw frames)
  followed by `windows_from_system`, which re-cuts the 11-frame windows from the
  stored trajectory. Disagreement here means the store or its documented window
  convention is wrong even if the download is perfect. Held to a coordinate
  tolerance, since it rounds through a float32 store and redoes the Kabsch fit.

Both must pass, on every one of the eight compared keys (`product_init`,
`target_delta`, `context_ligand_coords`, `pocket_init`, `product_features`,
`edit_features`, `pocket_features`, `bond_index`), and the set of window start
frames must match exactly. Re-run at any time:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src /home/ieb24002/anaconda3/envs/MDWM/bin/python \
  scripts/misato_store_crossvalidate.py --n-systems 40
# before the download finishes, validate the extractor against the archive over HTTP:
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src ... --remote --n-systems 3
```

**Per-system physical QC** is recorded in every manifest entry under `qc`, and
aggregated in `summary.json`:

* all coordinates finite (a non-finite system is rejected, not repaired);
* ligand bond lengths: min, max over all frames, and the first/last-frame max
  that the accept/reject rule uses; `bond_length_warning` if any frame exceeds
  `--broken-bond-angstrom`;
* closest pocket-heavy-to-ligand approach over all frames, with
  `clash_warning` below 1.2 Å (recorded, not rejected — some ligands are
  genuinely close, and silently dropping them would bias the corpus);
* atom counts (ligand, ligand heavy, pocket, whole system), bond count, molecule
  count, ligand start index, element histogram;
* ligand max displacement and last-vs-first RMSD; a system whose ligand does not
  move at all is rejected, because it carries no dynamics;
* **an independent check on the ligand boundary, and a peptide-ligand flag.**
  The boundary comes from `molecules_begin_atom_index[-1]`. Separately,
  `atoms_residue` is a residue *type* code in 0..22 (measured, not assumed — it
  is not a residue instance id), in which `0` means a non-standard residue and
  appears on no protein atom. So for a small-molecule ligand the code
  independently confirms the boundary
  (`qc.ligand_is_nonstandard_residue_only` and
  `qc.ligand_boundary_independently_confirmed`). For a **peptide or amino-acid
  ligand** the last molecule is built from standard residues and its codes
  legitimately collide with the protein's — 1FTJ's ligand is glutamate (code 9),
  1LAF's is an amino acid (code 3), 1D8E's is a hexapeptide spanning codes
  6/12/14/15/19/22. That is a chemistry class, not a boundary error, so it is
  reported as `qc.ligand_looks_peptidic` with the codes themselves in
  `qc.ligand_residue_type_codes`. **A small-molecule MMP corpus probably wants
  to exclude the peptidic ones**; `summary.json` counts them;
* **periodic-image jump detection** — see the warning below.

Rejections are recorded per system with a reason (`status: "rejected"`), never
silently dropped, and are distinguished from `failed` (a fault of the run, which
leaves no manifest entry so a re-run retries it).

`validation/rejection_crossvalidation.json` checks the *reject* path against the
same known-good cache, which the window comparison cannot cover. Three of the
four reason classes in the cache's `rejected_systems.csv` reproduce verbatim
(disconnected bond graph, ligand above the atom cap, and the 302-atom boundary
case). The fourth differs by design: the cache rejects a system when no window
*anchor* frame has ≥3 protein heavy atoms within 8 Å, whereas the store's pocket
is a union over all 100 frames and so keeps the system. Measured on 4E4A, the
one such system sampled: 27 union pocket atoms, closest approach 4.93 Å, **zero**
heavy atoms within 8 Å at each of the nine anchor frames, and
`windows_from_system` yields **0 windows** — exactly what the cache holds. So the
store is strictly more permissive at the system level and identical at the window
level, which is what a reusable store should be: keep the frames, let the
consumer decide.

---

## 8. Leakage exclusions

`data/processed/misato_leakage_exclusions_v1.json` lists **325** MISATO systems
to keep out of pretraining: EGFR (P00533), CDK1 (P06493) and CDK2 (P24941) at
the strict tier, plus six paralogs (ERBB2/3/4, CDK4/6/9) at the homolog tier.

Excluded systems are **recorded, not silently dropped**: each gets a manifest
entry with `status: "excluded_leakage"`, the reason, and the SHA-256 of the
exclusion list it came from. So the manifest always distinguishes "not extracted
because it would leak" from "not extracted yet".

The exclusion list's path and SHA-256 are also written into every
`summary.json`. If the list is ever revised, a new `--config-name` with the new
list will show exactly which systems changed status.

---

## 9. Fidelity gaps — read before making a claim from this store

These are properties of MISATO, verified against the archive rather than
assumed, and they carried over from the earlier pipeline unchanged:

* **No bond table, no formal charge, no aromaticity, no hybridization, no
  SMILES.** Bonds are inferred geometrically from covalent radii (frame 0);
  `degree` follows from those bonds; the three unavailable feature columns are
  written as **zero** rather than guessed.
* **No periodic box vectors**, so a ligand torn across the periodic boundary
  cannot be made whole. Such systems are *rejected*, not repaired, by a
  bond-graph connectivity gate plus a long-bond check. Connectivity is the test
  that does the work: because the bonds are inferred from a possibly-broken
  frame, an atom displaced by a box length simply acquires no bonds and a
  bond-length test alone never fires.
* **That connectivity gate has a hole, and this store measures it.** A
  *whole-molecule* translation by a box vector keeps every bond intact, so the
  connectivity gate cannot see it at all. Measured on 2VCJ: the ligand centroid
  moves **47.2 Å between two consecutive ~80 ps frames** while the radius of
  gyration stays at 4.68–5.04 Å and MISATO's own ligand RMSD stays at 1.4–2.7 Å
  — the ligand never left the pocket and the jump is purely an imaging artifact.
  Anything computing ligand dynamics from the raw stored coordinates of such a
  system gets tens of ångström of nonsense.

  Every system therefore carries `qc.ligand_centroid_step_max_angstrom`,
  `qc.ligand_centroid_step_median_angstrom`,
  `qc.n_frames_with_centroid_step_over_10_angstrom`,
  `qc.ligand_radius_of_gyration_rel_spread` and
  `qc.periodic_image_jump_suspected` (max step > 10 Å with a near-constant
  radius of gyration). It is a flag, **not** a rejection, because a genuinely
  unbinding ligand also takes large steps and telling the two apart needs the
  box MISATO does not ship. Use the **median vs max** step to separate them: a
  single imaging jump gives a large max with a small median (2VCJ: max 47.2 Å,
  median 0.65 Å), whereas real diffusion gives a large median too (4K7N: max
  16.6 Å, median 7.5 Å, and MISATO's own RMSD runs 42–84 Å). If you need clean
  ligand dynamics, filter on this flag.
* **`atoms_element` is a small internal index, not an element.** `atoms_number`
  carries the true atomic numbers and is what this store uses.
* **The ligand is the LAST molecule.** `molecules_begin_atom_index` lists the
  first atom of every molecule, so multi-chain systems have more than two
  entries (4K6W is `[0, 2890, 5813, 8715]`); the ligand begins at the **last**
  entry. Splitting on the second entry silently folds every chain after the
  first into the "ligand".
* **`edit_features` is a no-edit block.** MISATO systems carry no chemical edit,
  so this store cannot support an action-conditioned claim. It is a
  fixed-topology pretraining corpus.
* **MISATO's own train/val/test split is disjoint by PDB entry, not by
  protein.** Do not treat it as a protein-level generalization test without
  regrouping by UniProt. The split label is copied into each manifest entry for
  provenance; the store does not act on it.
