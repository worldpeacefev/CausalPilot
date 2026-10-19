# Schema and metrics

## Paths and manifests

`data/catalog.json` lists scenario/predicate/scene shards. Each line of a shard
is one triplet. Every `image`, `metadata`, and `symbolic_state_path` resolves
relative to the **top-level `data/` directory**, regardless of shard depth.
`source_plan` resolves relative to the repository root; for new data it points
to a reconstruction recipe rather than an archived AsgardBench plan.

`data/benchmark.json` and `data/causal_states.json` retain the archived schema
for 996 triplets and 71 units. They have adjusted image paths, unchanged image
bytes, and unchanged labels. Use `catalog.json` for the larger release.

## Important fields

| Field | Meaning |
| --- | --- |
| `triplet_id` | Unique observation-triplet identity |
| `causal_state_unit` | Cluster identity, shared by nuisance replicas |
| `scenario_aspect` | One of the five semantic scenario groups |
| `predicate` | Decision-critical predicate type |
| `scene`, `physical_scene_type` | FloorPlan and physical room category |
| `task_type` | Underlying task/checkpoint template |
| `instruction`, `history`, `action_interface` | Fixed context |
| `available_object_types` | Fixed role-union list for new data; loader derives the list for archived records |
| `target` | Privileged predicate target and decision object |
| `replica_profile` | Nuisance generation profile |
| `release_partition` | `paper_archive`, `nuisance_extension`, `predicate_extension`, or `bulk_extension` |
| `collection` | Curated core (`paper`) or expanded (`extended`); membership is disjoint by triplet ID |
| `certification_tier` | State-control and evidence contract |
| `review_status` | Actual review evidence level |
| `primary_visual_signal` | Primary causal target-union RMS and weak-signal flags; archived observations remain intact |
| `variants` | Exactly `anchor`, `causal`, `nuisance` |
| `variants.*.acceptable_actions` | Nonempty sets of `{action, object}` labels |
| `variants.*.symbolic_state_path` | Deduplicated symbolic snapshot; raw metadata remains separate |
| `variants.*.validation.predicate_source` | Raw simulator metadata or explicitly weaker successful-intervention trace |
| `validation` | Construction evidence and allowed/observed state differences |
| `validation.nuisance_allowed_symbolic_differences` | Explicit task-irrelevant simulator changes permitted for the named irrelevant-object profile; absent means none |

For trace-derived consumables, `validation.predicate_value` is a declared
extension label while `raw_metadata_predicate_value` and raw symbolic state
remain `false`. Never silently treat that label as a metadata observation.

Bulk replicas use `release_partition="bulk_extension"` and
`certification_tier="extension-unreviewed-render-v1"`. Original validation is
retained under `source_validation` at record and variant level. Source bounding
boxes refer to the original image dimensions, not the delivered JPEG. Bulk
records have no delivered-image visibility/bbox/RMS certificate.
`construction` records the source triplet and source image hashes, family,
severity, strengths, seed, output size and JPEG encoding parameters. The loader
also checks `source_certification_tier` when excluding trace-derived used-up
records, so re-rendering cannot silently bypass this default exclusion.

Collection-specific catalogs still resolve all data paths relative to top-level
`data/`. `iter_triplets("data", collection="paper")` is the supported API;
do not pass `data/paper` as the data root. The curated core has its own membership
and four-predicate scope. The reference archive identifies the historical cohort;
`paper/requirements.json` records exact paper-reference requirements and evidence.

## CIG metrics

Let `y_v` be a strictly parsed legal action, `V_v` indicate valid parsing and
`C_v = 1[y_v in acceptable_actions_v]`. Invalid outputs receive no agreement
credit, even when both are represented as null.

```text
Ordinary = (C_anchor + C_causal + C_nuisance) / 3
CRA      = C_anchor * C_causal * 1[y_anchor != y_causal]
NIA      = V_anchor * V_nuisance * 1[y_anchor == y_nuisance]
GTA      = C_anchor * C_causal * C_nuisance * 1[y_anchor == y_nuisance]
PPR      = V_anchor * V_causal * 1[y_anchor == y_causal]
```

NIA measures valid action agreement; it does not itself require correctness.
GTA requires correctness of all three observations plus Anchor/Nuisance
agreement. Two different acceptable actions can both be individually correct
while failing that agreement condition.

The paper pools the 996 triplets. For any larger release, report its exact
partition/tier selection and denominator. Bootstrap whole causal-state units,
carrying every accepted replica together. Unit-balanced and pooled-triplet
averages answer different questions.

## Visual curation provenance

`data/review/proxy_visual_review.jsonl` contains actual per-triplet agent
judgments: `verdict`, `reason_codes`, a cue-specific note, inspection scope,
visual/label criteria and exact context/asset SHA-256 bindings. It explicitly
states `reviewer_kind="agent"` and `actual_human_review=false`. This is dataset
quality annotation, not model inference or a benchmark test result.

Selected paper records have
`review_status="automated_controls_and_agent_triplet_visual_review"` and a
`proxy_visual_review` reference. Rejected records stay in the extended/archive
views and are never admitted merely to fill quotas. An annotation becomes stale
if its reviewed image, metadata, symbolic state or task/action labels change.
