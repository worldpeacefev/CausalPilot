# Explore the collections

Browse matched decision checkpoints by semantic aspect, predicate, and FloorPlan.
All observation paths resolve relative to the top-level `data/` directory.

| Collection | Units | Triplets | Observations | Predicates |
| --- | ---: | ---: | ---: | ---: |
| [Curated core (`paper`)](paper/) | 71 | 996 | 2,988 | 4 |
| [Expanded (`extended`)](extended/) | 95 | 11,865 | 35,595 | 7 |

Triplet IDs are disjoint; units overlap because replicas retain their source-unit
identity. Every selected core triplet has passing agent visual-review provenance.
The combined release contains **95 source units / 12,861 triplets / 38,583 observations**.

[Dataset card](../docs/dataset_card.md) · [Schema and metrics](../docs/schema.md) ·
[Core requirements](paper/requirements.json) · [Curation records](review/)

The preserved [reference archive](benchmark.json) contains the original 996-triplet
cohort. Use the collection catalogs for the curated and expanded views, with
collection-specific evidence and review contracts described in the dataset card.
