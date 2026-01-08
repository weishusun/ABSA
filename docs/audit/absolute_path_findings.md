# Absolute Path / Cross-Platform Findings

- `.gitignore:1` — Contains `cd C:\Users\weish\ABSA` (likely left from a PowerShell append). This is an absolute Windows path and breaks portability. Fix: replace file content with a clean ignore list (see suggestion below) and avoid shell redirection inside .gitignore.
- No other source files under `scripts/`, `review_pipeline/`, or `configs/` contain hard-coded absolute paths or drive letters; prior matches were from pycache/audit artifacts and are not part of the runnable codebase.

Recommended `.gitignore` refresh (portable):
```
.venv/
venv/
.idea/
.vscode/
__pycache__/
*.pyc
*.pyo
*.pyd
.pytest_cache/
.mypy_cache/
.ruff_cache/
build/
dist/
*.egg-info/
.env
.env.*
outputs/
```

Workspace strategy: keep repo code read-only; stage outputs, models, and temp data under a sibling `workspace/ABSA/` (bind via env `ABSA_WORKSPACE` with default `repo_root/outputs`). All paths in scripts already use `Path(...)` relative to repo root; swap any future absolute references to `Path(__file__).resolve().parent` + env overrides.
