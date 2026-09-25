## Summary of Changes
Provide a concise explanation of what this pull request does and why.

## Architectural & Privacy Checklist
- [ ] Preserves the dual-channel audio invariant (ch0 = mic, ch1 = remote audio).
- [ ] Preserves mean-centered embedding space for speaker biometrics.
- [ ] Maintains the 100% offline-first default (no unexpected telemetry or cloud dependencies).
- [ ] Exposes all new tunables in `config.example.yaml` rather than hardcoding.
- [ ] Verified that no `.wav` files, private configs, or personal voiceprint databases are included in the diff.

## Testing & Verification
- [ ] Added unit or integration tests under `tests/` covering the changes.
- [ ] Verified that the test suite passes (`.venv/bin/python -m pytest`).
- [ ] Verified live or offline reprocess workflow (`./record.sh --from-file ...`).
