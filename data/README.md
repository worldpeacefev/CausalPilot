# Benchmark collections

All manifest paths resolve relative to top-level `data/`.

| Collection | Units | Triplets | Observations | Predicates |
| --- | ---: | ---: | ---: | ---: |
| [paper](paper/) | 71 | 996 | 2988 | 4 |
| [extended](extended/) | 95 | 11865 | 35595 | 7 |

Triplet IDs are disjoint; units overlap because extra renders retain their originating unit ID.
Paper candidates have completed the user-requested agent proxy review for every selected triplet; seven-predicate coverage remains incomplete and actual human review is not claimed. See [requirements](paper/requirements.json).
The unchanged 996-triplet historical cohort remains in [benchmark.json](benchmark.json). It is an archive, not a certificate of compliance.
