# Benchmark collections

All manifest paths resolve relative to top-level `data/`.

| Collection | Units | Triplets | Observations | Predicates |
| --- | ---: | ---: | ---: | ---: |
| [paper](paper/) | 71 | 996 | 2988 | 4 |
| [extended](extended/) | 93 | 11825 | 35475 | 7 |

Triplet IDs are disjoint; units overlap because extra renders retain their originating unit ID.
Paper candidates do not yet satisfy seven-predicate coverage or verified human review of every triplet. See [requirements](paper/requirements.json).
The unchanged 996-triplet historical cohort remains in [benchmark.json](benchmark.json). It is an archive, not a certificate of compliance.
