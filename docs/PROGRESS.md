# DeepWukong Development Progress

## Current Epic
**E1: Modernization** — Upgrade all dependencies, Dockerize, and document data preparation

## Epics
- [x] E1: Modernization — upgrade deps, Docker, data docs

## Plans Index
| Date | Plan | Epic | Status | Notes |
|------|------|------|--------|-------|
| 2026-04-07 | modernize-deps | E1 | done | gensim 4.x, PL 2.x, networkx, requirements.txt, README. All 17 modules import, model verified. |
| 2026-04-07 | dockerize | E1 | done | Dockerfile (pytorch/pytorch runtime, 6.72GB), .dockerignore, DATA_PREPARATION.md, README updated. GPU training verified in Docker (F1=94.56%). |

## Next Steps
- Project fully operational. Possible future work:
  - Add more CWE datasets (CWE399, CWE20, etc.)
  - Migrate from `GlobalAttention` (deprecated) to `AttentionalAggregation`
  - Add `sync_dist=True` to test metrics logging for cleaner DDP output

## Known Issues
- Pretrained model checkpoints (PL 1.x) not compatible with current PL 2.x — retrain instead
- `GlobalAttention` deprecation warning from torch_geometric (cosmetic, still works)

## Key Decisions
| Category | Decision |
|----------|----------|
| Python | Target Python 3.9+ |
| PyTorch | 2.6.0+cu124 |
| Docker | `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` base, ~6.7GB |
| Data | Mounted as volume, not baked into image |
| joern | Bundled in repo + Docker image (with openjdk-11-jre-headless) |

## Session Log

### 2026-04-07
- **Focus:** Modernization + Dockerization
- **Completed:**
  - Phase 1: Fixed gensim 4.x, PL 2.x, networkx gpickle across 15 source files
  - Phase 2: Downloaded CWE119 data (1.8GB), verified CPU training (F1=0.857 epoch 1)
  - Phase 3: Installed CUDA PyTorch, resolved numpy/scipy/matplotlib/gensim compat chain
  - Phase 4: GPU training on CWE119 — 17 epochs in ~9 min, test F1=94.56%, Acc=98.30%
  - Phase 5: Dockerfile, .dockerignore, DATA_PREPARATION.md, README with Docker usage
  - Phase 6: Docker build (6.72GB) + GPU training in container verified — identical results
  - Phase 7: Chinese README (`README_zh.md`) + Chinese data prep guide (`DATA_PREPARATION_zh.md`)
  - Phase 8: Bundled joern into `joern/` directory, tested full 5-step pipeline (joern→XFG→sym→w2v→train)
  - Phase 9: Pinned all dependency versions in requirements.txt, fixed download link placeholders
  - Phase 10: Added Java + joern to Docker image, fixed joern-parse.py abspath bug for Docker path matching
  - Phase 11: Rewrote Docker README sections — complete 6-step pipeline from source code to model
  - Phase 12: Added GitHub Actions CI (import check, CPU smoke test, Docker build)
  - Phase 13: Tested full Docker pipeline end-to-end (source code → joern → XFG → train)
- **Tests:** All modules import. Full train+test on CWE119 (71K train, 9K test). Docker GPU + full pipeline verified.
- **Files:** `Dockerfile`, `.dockerignore`, `README.md`, `README_zh.md`, `docs/DATA_PREPARATION.md`, `docs/DATA_PREPARATION_zh.md`, `.github/workflows/ci.yml`, `requirements.txt`, `env.sh`, `src/joern/joern-parse.py`, `joern/` (bundled), + 12 source files
