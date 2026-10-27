# Data

## MISATO molecular-dynamics store (not committed; 7.1 GB)

All trajectories come from MISATO:
- T. Siebenmorgen et al., *Nat. Comput. Sci.* 4, 367–378 (2024), doi:10.1038/s43588-024-00627-2;
- data on Zenodo, record [7711953](https://zenodo.org/records/7711953), licensed CC BY 4.0.

The model reads a per-system frame store built from MISATO's `MD.hdf5`, at
`data/MISATO/extracted/frames100_all_v1/` (16,536 `.npz` files, 100 frames each, with the protein
atoms within 8 Å of the ligand). To build it:

```bash
# 1. resumable download of MD.hdf5 (about 133 GB); kept only if its published MD5 matches
python scripts/misato_store_download.py
# 2. cross-validate the extractor on a known-good sample, then extract
bash scripts/misato_store_pipeline.sh
# verify the result against the store used for the paper
cd data/MISATO/extracted && sha256sum -c ../store_frames100_all_v1.sha256
```

- [`MISATO/README.md`](MISATO/README.md) documents the build: extractor version, rejection rules, QC
  counts and the `.npz` layout.
- `MISATO/manifest/*/summary.json` are the build summaries. Absolute paths in them refer to the
  machine on which the store was built.
- `MISATO/store_frames100_all_v1.sha256` is the SHA-256 of every store file used in the paper.

## What is committed

These are derived from MISATO and shared under CC BY 4.0:
- the MMP census and the paired corpus with its atom maps (`outputs/misato_mmp_census_v1/`,
  `outputs/mmp_transfer_corpus_v1/`);
- the split and fold example lists;
- every per-example result CSV, in `reference_outputs/` and `cleanrun/expected/`.

## Third-party data (not redistributed)

The published-generator experiment (paper Section IV-F) uses data that remain under their authors'
licenses:
- the 100 CrossDocked test pockets (Francoeur et al., 2020);
- DiffSBDD's released samples, `crossdocked_fullatom_cond` (Schneuing et al.);
- TargetDiff samples, regenerated with its released sampler and weights (Guan et al., 2023).

Place them under `data/external/` as described in `docs/README_original_run.md` and
`reproduce/05_published_models.sh`, `reproduce/06_targetdiff_sampling.sh`. Their hashes are in
`CHECKSUMS.sha256`.
