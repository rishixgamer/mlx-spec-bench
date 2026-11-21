# Changelog

## 1.0.0 candidate

- Added an offline Qwen2 fixture, pinned Qwen2.5 formats, target-only and speculative greedy decoding, and transactional sliding-window KV caches.
- Added versioned workload, quality, and result evidence, isolated workers, matched reports, and a 96-repetition release qualification manifest.
- The five fixture prompts use 78 input tokens and emit 16 tokens at batch size 2 and a cap of 8. The complete token baseline is packaged with the fixture. The held-out fixture task records one exact match in four examples.
- Cohorts schedule individual model forwards. Speculation supports temperature zero. Physical footprint and system pressure are sampled. Hardware qualification and publication are tracked separately from candidate builds.
- Apple Silicon CI exercised the installed wheel through the offline run and report smoke, including token and schema corruption checks. The same job verifies all four fixture weight formats and two Python API repetitions.
