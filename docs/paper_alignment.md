# Benchmark design and paper protocol

CausalPilot-Bench organizes embodied decision checkpoints around matched
Anchor / Causal / Nuisance observations. Its central question is whether a
next-action policy responds to the visual state that determines the decision
while remaining consistent under task-irrelevant appearance changes.

The design reference is the final `all-finish.pdf`: Sections 3.1–3.4,
Appendix A.1–A.2, Tables 4–5, and Appendix B. The manuscript SHA-256 is recorded
in [paper_identity.json](../data/paper_identity.json).

## The matched decision contract

A causal-state unit fixes the task goal, successful history, target object,
local checkpoint, available object types, and legal action interface. Each
triplet contains:

| Role | Controlled change | Action requirement |
| --- | --- | --- |
| Anchor | Reference state | Nonempty acceptable-action set `Y_a` |
| Causal | Decision-critical target intervention | Nonempty `Y_c`, disjoint from `Y_a` |
| Nuisance | Appearance or explicitly qualified task-irrelevant variation | Same acceptable-action set as Anchor: `Y_n = Y_a` |

Each observation is supplied to a method independently. Reference predicates,
action labels, simulator metadata, and role identity belong to the evaluator.
The [reader](../construction/load.py) provides fixed context and one current
image through `observation_input`.

## Collection scope

| Collection | Units | Triplets | Observations | Target predicates |
| --- | ---: | ---: | ---: | --- |
| [Curated core (`paper`)](../data/paper/) | 71 | 996 | 2,988 | `isDirty`, `isOpen`, `isToggled`, `isCooked` |
| [Expanded (`extended`)](../data/extended/) | 95 | 11,865 | 35,595 | All seven vocabulary predicates |

The curated core uses the paper's three original state-control tiers and
per-triplet agent visual review. Its `paper` API name identifies the collection;
it does not establish identity with the manuscript's experimental cohort or
seven-predicate certification. Exact coverage and requirements are recorded in
[requirements.json](../data/paper/requirements.json).

The preserved [reference archive](../data/benchmark.json) retains its original
996 triplet identities, action labels, and image bytes. The curated core has
revised membership. Results reported for the original cohort must retain that
cohort and cannot be assigned to the revised core or expanded collection.

## State-control tiers

| Original tier | Permitted causal changes | Visibility contract |
| --- | --- | --- |
| `pilot-v3-strict` | Target Boolean predicate only | Target bounding box ≥1% of image area |
| `extended-target-articulation-v1` | Target predicate and the openable target's own position/rotation | Target bounding box ≥1% |
| `extended-small-visible-v1` | Core state-control rules | Dirty vessel ≥0.5%; cooked food ≥0.15%; target-union RMS ≥1.0 |

Core inclusion checks intervention success, controlled non-target state,
inventory and reciprocal containment, separated causal action sets, nuisance
branch preservation, target visibility, and image decoding. Selection does not
use predictions from the evaluated methods.

Additional predicate coverage uses explicit extension contracts:

| Extension tier | State evidence and permitted target changes |
| --- | --- |
| `extension-liquid-content-v1` | Raw liquid Boolean plus the target's liquid contents |
| `extension-target-topology-v1` | Raw slicing state, target descendants, and their direct reciprocal containment |
| `extension-intervention-trace-v1` | Successful consumable intervention and visible target-local change; trace-derived rather than raw-Boolean evidence |

These contracts are distinct from the original three tiers. Trace-derived
consumable records preserve raw metadata unchanged and require explicit loader
opt-in. Detailed thresholds and geometry controls appear in the
[dataset card](dataset_card.md).

## Visual diversity

The reference profile suite combines in-simulator illumination with brightness,
color temperature, contrast, gamma, saturation, sharpening, Gaussian blur,
sensor noise, vignetting, JPEG compression, resolution changes, quantization,
illumination gradients, and channel tint. Profile eligibility varies by unit.
Simulator viewpoint and material variants are indexed separately.

For image-space profile `r`, a shared light transform is applied to Anchor and
Causal, while a stronger nuisance transform starts from the original Anchor:

```text
Anchor_r   = L_r(original_anchor)
Causal_r   = L_r(original_causal)
Nuisance_r = S_r(original_anchor)
```

The expansion adds motion blur, chromatic effects, sensor banding, shot noise,
digital scaling, translation, occlusion, and pixel dropout. Its bulk suite
contains 20 families at six severity levels across 92 source units. Bulk
outputs carry `unreviewed_expansion`; inherited source labels and validation
remain separately identified from delivered-image evidence.

## Curation and reproducibility

Every selected core triplet has an agent visual-review annotation covering
state contrast, nuisance cue preservation, context/action consistency, and
occlusion or artifacts. SHA-256 bindings connect each annotation to its exact
images, raw metadata, symbolic states, and labels. Reviewer provenance is
explicitly agent-based. The [curation protocol](../data/review/) describes the
inspection procedure.

Raw simulator metadata, deduplicated symbolic snapshots, reconstruction
recipes, nuisance parameters, unit membership, and file checksums make the
release inspectable and extensible.

## Aggregation

The causal-state unit is the intervention cluster. Multiple profiles of the
same unit are correlated replicas. The manuscript's main estimand pools
triplets; uncertainty estimates and paired comparisons must cluster complete
units. A unit-balanced average is a separate estimand.

For any release subset, report the selected collection, predicates, evidence
tiers, review status, and denominator. Metric definitions are provided in
[schema.md](schema.md).
