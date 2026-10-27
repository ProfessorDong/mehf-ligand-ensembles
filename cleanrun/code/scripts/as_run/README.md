# as_run — the exact scripts used, kept as a record

These are the lane scripts exactly as they ran on 2026-10-23/25. They are the
authoritative record of the commands and flags behind every number in
`expected/`, and the logs in `cleanrun/logs/` were produced by them.

**They are not portable and should not be run as-is.** They hardcode
`/home/ieb24002/MEHF` and, worse, they set `PYTHONPATH=src` relative to that
root. On a stock MEHF checkout that resolves to the pre-supplementary-task
source, which has no `pocket_selection` and therefore uses the LEAKY
over-all-frames pocket crop — with no error, just a wrong answer.

Use `cleanrun/reproduce/*.sh` instead. Those set PYTHONPATH to
`cleanrun/code/src` and refuse to start if the wrong loader is importable.

Read these for the exact flags; run those.
