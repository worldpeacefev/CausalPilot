# Curated core

**71 causal-state units · 996 matched triplets · 2,988 observations**

A compact collection for studying state-sensitive next-action decisions,
combining exact state-control checks with agent visual curation of every
selected triplet.

[Open the catalog](catalog.json) · [Design protocol](../../docs/paper_alignment.md) ·
[Curation annotations](../review/)

| Predicate | Units | Triplets |
| --- | ---: | ---: |
| `isDirty` | 10 | 156 |
| `isOpen` | 31 | 447 |
| `isToggled` | 25 | 336 |
| `isCooked` | 5 | 57 |
| **Total** | **71** | **996** |

Core records use `pilot-v3-strict`, `extended-target-articulation-v1`, or
`extended-small-visible-v1`. Inclusion checks raw metadata, controlled symbolic
state, intervention success, action-set separation, nuisance branch preservation,
target visibility, and image signal. Every selected triplet has a passing
agent visual-review annotation bound to its assets and labels.

The loader key is `collection="paper"`. This curated collection has revised
membership and four target predicates; the expanded collection supplies the
remaining vocabulary types under separate contracts. Exact paper-reference
requirements and review provenance are retained in [requirements.json](requirements.json).
The [dataset card](../../docs/dataset_card.md) explains cohort selection and reporting.

```python
from construction.load import iter_triplets

core = iter_triplets("data", collection="paper")
```

Browse [access](scenarios/access/), [cleaning](scenarios/cleaning/),
[device control](scenarios/device_control/), and
[food preparation](scenarios/food_preparation/).
All observation paths resolve relative to the top-level `data/` directory.
