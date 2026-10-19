# Dataset card

CausalPilot-Bench is a structured diagnostic for local next-action grounding in
AI2-THOR household scenes. Matched observations connect state-sensitive action
changes with nuisance consistency at a fixed decision checkpoint.

## Release at a glance

| Collection | Units | Triplets | Observations | Predicate types |
| --- | ---: | ---: | ---: | ---: |
| Curated core (`paper`) | 71 | 996 | 2,988 | 4 |
| Expanded (`extended`) | 95 | 11,865 | 35,595 | 7 |
| **Combined, unique** | **95** | **12,861** | **38,583** | **7** |

The release spans 48 physical scenes and five semantic aspects. Triplet IDs
are disjoint across collections; source units overlap. Complete counts appear
in [counts.json](../data/counts.json).

The core uses the original three state-control tiers and passing agent visual
annotations for all 996 triplets. Its target predicates are `isDirty`, `isOpen`,
`isToggled`, and `isCooked`. The `paper` collection key denotes the curated
paper-scale subset, whose membership differs from the archived experimental
cohort. It is not a seven-predicate paper-certified replacement; full-conformance
status and exact coverage remain in [requirements.json](../data/paper/requirements.json).
Published scores belong to their original evaluation cohort.

The expanded collection adds seven-predicate coverage through explicit
extension contracts. Its bulk suite contributes 11,040 triplets from 92 source
units, 20 nuisance families, and six severities. Bulk records inherit source
labels, are marked `unreviewed_expansion`, and may obscure decision evidence.
Source validation does not certify the delivered 256×256 JPEG image.

## Intended use

Study causal response, nuisance invariance, and jointly grounded next actions.
These samples represent local checkpoints in a procedural household simulator.
They support controlled diagnostic studies; additional validation is needed for
claims about interactive task completion or physical robot deployment.

Specify `collection`, `release_partition`, `certification_tier`, and
`review_status` when defining a reported population.

## Evidence contracts

| Tier | Permitted causal changes | Visibility |
| --- | --- | --- |
| `pilot-v3-strict` | Target Boolean only | Target bbox ≥1% |
| `extended-target-articulation-v1` | Target Boolean and the openable target's own pose | Target bbox ≥1% |
| `extended-small-visible-v1` | Original core state controls | Dirty vessel bbox ≥0.5%; cooked food bbox ≥0.15%; target-union RMS ≥1 |
| `extension-liquid-content-v1` | Target liquid Boolean and `fillLiquid` | Target bbox ≥1% |
| `extension-target-topology-v1` | Original target slicing state, its descendants, and direct reciprocal containment | Target/descendant union bbox ≥1% |
| `extension-intervention-trace-v1` | Successful controlled consumable intervention with visible local change | Target bbox ≥0.15% |
| `extension-unreviewed-render-v1` | Image-space stress transforms with inherited source labels | No delivered-image visibility certificate |

For liquid and slice sources, unrelated objects, inventory, and agent state
remain controlled. Slice topology contracts permit only the original target,
its simulator-named descendants, and direct reciprocal receptacle membership.
The fixed object-type context is the union across roles.

Consumable labels use a documented successful intervention and visible
local change. Raw metadata is preserved and reports `isUsedUp=false` for these
records; the extension label is trace-derived. The default loader excludes
this tier and its bulk descendants. Set `metadata_only=False` to include it.

Additional simulator sources use a fresh scene reset for each role, with
physics paused before setup and intervention. Source gates require target-local
causal RMS ≥5, nuisance full-image RMS ≥1, separated action branches, and
controlled nuisance symbolic state. Recipes record the reset, poses, and
commands; each observation does not require a new Unity process.

Recipes may restore movable/pickupable poses using `SetObjectPoses`. A declared
`numeric_geometry_tolerance` permits position noise ≤0.00001 m and rotation
noise ≤0.001 degrees. Boolean state, inventory, and containment remain exact.
Records without that field require exact equality. This tolerance belongs to
extension construction and does not modify the original strict tier.

The named `simulator_irrelevant_object` nuisance permits a visible, empty,
task-irrelevant Drawer's open state and own articulated pose to change. The
target, other objects, inventory, agent, and acceptable action branch remain
controlled. Allowed paths are recorded in
`nuisance_allowed_symbolic_differences`.

Screened render replicas require target-union causal RMS ≥1 and nuisance RMS
≥1. These pixel-signal gates complement visual curation. Original strict
records may retain weaker primary RMS, recorded in `primary_visual_signal`.
Bulk variants keep source validation under `source_validation`; source bounding
boxes refer to source dimensions and do not certify output-image visibility.

## Visual curation

[Per-triplet annotations](../data/review/) cover 1,036 candidates and bind each
verdict to exact images, raw metadata, symbolic states, and task/action labels.
All 996 selected core triplets have passing annotations.

The reviewer was one Codex agent using matched target crops, full-frame
follow-up for ambiguous cases, full frames for all faucet candidates, and full
frames for fresh television candidates. The records explicitly identify agent
review and zero actual human reviewers. They do not establish the human-review
provenance stated in the manuscript. Automatic checks and agent annotations are
reported separately; source review does not transfer to generated descendants.
Selection does not use predictions from evaluated methods.

## Context, labels, and information access

Acceptable actions are sets. Anchor and Causal have nonempty disjoint sets;
Nuisance retains the Anchor set. `done none` applies when the local goal is
already satisfied. The object argument of `put` is the destination receptacle.
Slice contexts hold a Knife; liquid contexts ask to empty the vessel;
consumable contexts distinguish usable from exhausted supplies.

Model inputs contain only the fixed instruction, successful history, legal
action interface, fixed object types, and one current image.
`observation_input` returns these fields. Model adapters should send image
bytes without revealing the informative filesystem path. Labels, reference
predicates, oracle facts, metadata, recipes, role names, profile IDs, and
certification fields are privileged evaluator material.

## Sampling and reporting

`causal_state_unit` identifies the intervention cluster. Replica counts do not
increase the number of source units. Units may share physical scenes and objects;
95 source-unit identities do not imply 95 independent scenes. Uncertainty and
paired comparisons should retain complete replica clusters.

The manuscript uses pooled-triplet weighting. Unit-balanced averages answer a
separate question. Optional construction partitions group whole physical scenes
and their sibling units; they are not manuscript experimental splits or evidence
about training contamination. Report the exact selected denominator for each result.

Conditional consumable tasks, paused physics, and fixed inspection poses define
the scope of this diagnostic. Renderer versions and hardware can affect pixels.

## Reproducibility and licensing

The release includes raw simulator metadata, deduplicated symbolic snapshots,
reconstruction recipes, source identities, transformation parameters, and a
SHA-256 file manifest. The reference archive preserves original triplet IDs,
action labels, and image bytes alongside the classified release.

Construction code retains the upstream MIT [license](../LICENSE).
[AI2-THOR](https://github.com/allenai/ai2thor) simulator assets and rendered
content remain subject to upstream terms; the code license does not relicense
third-party visual assets.
