# Visual curation records

Per-triplet annotations connect visual curation to the exact released assets.
Every selected core triplet has a passing annotation.

[Annotations](proxy_visual_review.jsonl) · [Coverage summary](summary.json) ·
[Binding implementation](../../construction/review.py)

## Coverage and provenance

| Item | Count or method |
| --- | --- |
| Reviewed triplets | 1,036 |
| Inspected observations | 3,108 |
| Source-unit contexts checked | 73 |
| Passing candidates | 1,013 |
| Excluded candidates | 23 |
| Selected core triplets | 996 |
| Reviewer | One Codex agent |
| Review type | Agent visual review following a manual-style inspection procedure |
| Actual human reviewers | 0 |

These are dataset quality annotations. They record agent judgments and do not
represent a human perception study, independent reviewer agreement, or model
performance measurements.

## Inspection procedure

1. Inspect matched Anchor/Causal/Nuisance target crops for state contrast,
   visibility, occlusion, image artifacts, and preservation of the nuisance cue.
2. Check each source unit's instruction, completed history, target, and action
   labels against the visible local decision.
3. Revisit ambiguous cases using full frames and native target crops. Inspect
   all 96 faucet triplets in full frames because water particles can lie outside
   the faucet instance bounding box.
4. Inspect all 40 fresh television candidates in full frames alongside
   independent exact state-control checks.
5. Accept a triplet only when all four recorded visual/context criteria pass.

Primary inspection used matched crop contact sheets, with full-frame follow-up
as described above. Selection did not use predictions from the evaluated methods.
Records preserve individual verdicts, reason codes, and cue-specific notes.

## Binding and selection

Each annotation contains SHA-256 bindings for all three images, raw metadata
files, symbolic snapshots, and the reviewed task/action context. Changing those
assets or labels invalidates the annotation.

All 996 selected core records pass review. The remaining 17 passing candidates
and 23 excluded candidates retain their annotations in the expanded collection;
the reference archive is preserved. Source review does not automatically certify
an unreviewed transformed descendant.
