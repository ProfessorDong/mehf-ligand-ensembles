# Redactions made for publication

Three record files were edited before publication, and only on the lines described below. The edits
remove details about jobs that belonged to other people or projects on the shared GPU server:
- another person's user name;
- a reference to an unrelated research project and its file path;
- a mention of an unrelated service.

Nothing that bears on the experiment was changed: no number, decision, timestamp, hash or command.
The pre-registration (`frozen/PREREGISTRATION.md`, sha256 `3863992a…`) is untouched and still
matches its recorded hash.

| File | Lines edited | sha256 before | sha256 after |
|---|---|---|---|
| `logs/CHAIN.log` | 11–15, 261 | `b2b902cf3b8b45b5ebd086daf6a3829c25537d7049f73eda0b9cf7f0d04498c0` | `aba43e0ceae12a4c0dfe90babd9dae47520ba10049ff16d81a8b69d715005f2c` |
| `expected/RESULT.md` | 24 | `23ddd160b26d06cb8184667765d1a579f61688461ffa44c34e3fea0e2dcc69a9` | `206cb4021b46fe87dc07fc7f010513c04cd156534708367ec123cbbc92388b54` |
| `code/reference_outputs/clean_protocol_v1/RESULT.md` | 24 | same as above | same as above |

`CHECKSUMS_cleanrun.sha256` is kept exactly as received. It therefore lists the "before" hashes for
`logs/CHAIN.log` and `expected/RESULT.md`, and `sha256sum -c` reports those two as FAILED. The third
file is not listed in it. The other 20 entries that cannot be checked are Python bytecode caches
(`__pycache__/*.pyc`), which are deliberately not committed.

The unredacted originals are held by the authors and are available on request for verification.
