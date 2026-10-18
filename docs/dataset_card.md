# Dataset card

## Intended use

Diagnose local next-action grounding under matched visual interventions in
AI2-THOR household scenes. These are checkpoint decisions, not complete
interactive episodes or real-world robot demonstrations.

The archived population is preserved for comparison with the manuscript.
The classified catalog additionally contains predicate and nuisance extensions.
Use `release_partition`, `certification_tier`, and `review_status` to specify
the population in any reported result.

## Certification and provenance

The archived tier names retain their historical meaning:

| Tier | State changes | Visibility |
| --- | --- | --- |
| `pilot-v3-strict` | Target Boolean only | Target bbox ≥1% |
| `extended-target-articulation-v1` | Target Boolean and its own openable pose | Target bbox ≥1%; two archived Microwave units are corrected to this tier in the classified catalog |
| `extended-small-visible-v1` | Historical core controls | Dirty vessel bbox ≥0.5%, cooked bbox ≥0.15%; target-union RMS gate |

New predicate extensions use separate, explicit contracts:

- `extension-liquid-content-v1`: `isFilledWithLiquid` and the target's
  `fillLiquid` may change together. Other symbolic objects, inventory, poses,
  and the agent remain equal across fresh reconstructions.
- `extension-target-topology-v1`: original target `isSliced`, creation of its
  own simulator-named descendants, original target parent membership and direct
  receptacle reciprocal membership may change. Unrelated creation or membership
  changes are rejected. Visibility is the union of the target and its slices.
- `extension-intervention-trace-v1`: consumable state comes from a successful
  controlled intervention and a visible target-local image change. Raw metadata
  remains untouched and still reports `isUsedUp=false`. This layer is weaker
  than metadata certification and is excluded by the default loader.

New slice/liquid units require target/descendant bbox ≥1%; the separately
qualified trace-derived consumable layer permits bbox ≥0.15% to retain the
visible exhausted core. All new units require target-local causal RMS ≥5 and
nuisance full-image RMS ≥1, identical nuisance symbolic state, and separated
action branches. Each role is reconstructed with a fresh scene reset; physics
is paused before setup and interventions. This differs from starting a new
Unity process per observation and is explicitly recorded in the recipe.

The explicitly named `simulator_irrelevant_object` nuisance is an exception to
whole-scene equality: only a visible, empty Drawer unrelated to the vessel
task may change its open state and own articulated pose. The target, all
other objects, inventory, and agent remain controlled. Its exact permitted
paths are recorded in `nuisance_allowed_symbolic_differences`; the task-relevant
state and acceptable action branch remain unchanged. It is a correlated
replica of an existing unit, not a new causal-state unit.

Recipes can restore movable/pickupable poses with `SetObjectPoses` before
inspection. When a record declares `numeric_geometry_tolerance`, comparison
allows only position noise ≤0.00001 m and rotation noise ≤0.001 degrees;
predicates, inventory, and containment still require exact equality. Raw
metadata/snapshots and all raw difference paths are retained. Records without
this field use exact equality. This numerical allowance is an extension
construction rule, not a retroactive change to the archived strict contract.

Automated render replicas require causal target-union RMS ≥1 and nuisance RMS
≥1. These numerical gates check retained visual signal, **not human semantic
visibility**. The resulting records remain `automated_checks_only` until a real
review is recorded. No generated model response is used for selection.

The archived strict tier does not specify an RMS≥1 gate. Primary pairs below
that additional diagnostic threshold are retained and explicitly tagged in
`primary_visual_signal.quality_flags`. This signals weak average pixel contrast
and a need for visual review, not an automatic revision of the paper's labels.
New render replicas still must pass their RMS gate. The exact flagged-unit
count is recorded in `data/counts.json`.

## Context and labels

Acceptable actions are sets. `done none` is appropriate when the local task is
already satisfied. It is not interchangeable with an unnecessary object action.
Placement uses the destination receptacle as the object argument.

New slice contexts hold a Knife. Liquid tasks ask to empty the vessel; this
avoids introducing a new `fill` action outside the manuscript vocabulary.
Consumable tasks ask to collect a usable supply and leave an exhausted one.
All roles share the same instruction and action history. The fixed object-type
list is the union across roles, preventing slice creation from changing textual
context between Anchor and Causal.

Labels, target predicates, oracle facts, metadata, recipes, profile IDs, role
names, filenames, and certification fields are privileged material. Model
inputs should contain only instruction, history, legal action interface,
the fixed object-type list, and a single current image. `observation_input`
returns this restricted input; an adapter must send image bytes rather than
exposing its informative filesystem path.

## Grouping and limitations

`causal_state_unit` identifies the intervention cluster. Replica counts do not
increase the independent sample size. New units can share physical scenes or
objects with archived units; they are additional checkpoint contexts, not a
claim of statistical independence between all scenes. Optional construction
partitions group complete physical scenes, including all sibling units and
replicas. They are not the manuscript's experimental splits and do not establish
that any model was trained without contamination.

The extension is a small, procedurally generated household diagnostic. Its
conditional consumable instructions, paused physics, and fixed inspection
poses can be easier than unconstrained deployment. Labels encode the documented
local high-level action interface; no new execution-success or end-to-end task
completion claims are made. Renderer versions and hardware may affect pixels.

## Licensing

Construction code retains the upstream MIT license. Consult
[AI2-THOR](https://github.com/allenai/ai2thor) for simulator/asset terms. This
release does not relicense third-party visual assets or include a new claim
that all rendered observations have MIT asset rights.
