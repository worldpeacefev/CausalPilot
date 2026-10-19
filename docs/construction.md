# Build and extend CausalPilot-Bench

Construct new causal-state units, generate reproducible nuisance variants, and
assemble classified collections with explicit evidence and curation provenance.

## Environment

Run commands from the repository root with Python 3.10+ and `uv`:

```bash
uv sync
```

The reader and image-space expansion tools use Pillow and NumPy. Simulator
construction additionally requires Linux, Xvfb, a working graphics driver,
and AI2-THOR:

```bash
uv sync --extra simulator
```

## Reconstruct source units

Write construction outputs to an external working directory, keeping temporary
artifacts and execution logs separate from the release:

```bash
xvfb-run -a uv run --extra simulator python -m construction.expand_simulator \
  --output /tmp/causalpilot-extension \
  --scenes FloorPlan1 FloorPlan2 FloorPlan3 FloorPlan4 FloorPlan5 FloorPlan301 FloorPlan401 FloorPlan402 FloorPlan403 \
  --limit 40
```

The generator pins AI2-THOR 5.0.0 and Unity build
`f0825767cd50d69f666c7f282e54abfe58f1e917`. Each accepted unit saves unmodified
metadata, images, inspection poses, intervention commands, its evidence contract,
and a reconstruction recipe. Each role starts from a fresh scene reset with
paused physics. Generation resumes accepted units from `unit.json`; use a new
working directory when changing construction rules.

Recipes can preserve movable/pickupable poses. Declared geometry tolerances
apply only to position and rotation; Boolean and relational controls remain
exact. Thresholds and permitted changes are specified in the
[dataset card](dataset_card.md).

A strict television unit can be reconstructed with:

```bash
LP_NUM_THREADS=2 xvfb-run -a uv run --extra simulator python -m construction.native_toggle \
  --output /tmp/native-tv --scenes FloorPlan216
```

This constructor holds a RemoteControl, selects a reachable visible screen pose,
and permits only the target `isToggled` change. Its recipe records the poses and
commands needed for reconstruction.

## Assemble source data

```bash
uv run python -m construction.build_release \
  --legacy-benchmark data/benchmark.json \
  --legacy-states data/causal_states.json \
  --extension /tmp/causalpilot-extension/dataset.json \
  --paper /path/to/all-finish.pdf
```

The builder preserves archived image bytes and labels, imports accepted source
units, generates screened nuisance profiles, writes classified shards and unit
indexes, and groups optional partitions by physical scene. `--paper` binds the
identity manifest to the manuscript SHA-256. Multiple independently constructed
folders can follow `--extension`.

`data/source_catalog.json` is the authoritative source catalog before bulk
expansion. Update it deliberately when adding source units to an existing split
release; the collection assembler reads it rather than an overwritten top-level
catalog. Original construction modules remain in `AsgardBench/causal/`:
`generator`, `extended`, `viewpoint`, `appearance`, and `render_replicas`.

## Generate the bulk stress suite

```bash
uv run python -m construction.expand_bulk --workers 4 --severities 6 --size 256
```

Twenty families at six severities create 120 replicas per eligible primary
source unit. The released bulk suite covers 92 source units, yielding 11,040
triplets. Each output records its source identity, source hashes, parameters,
seed, dimensions, and JPEG settings. The three strict television units in
FloorPlan216/205/214 contribute additional screened profiles separately.

Bulk outputs inherit source action labels and carry `unreviewed_expansion` /
`extension-unreviewed-render-v1`. They have no semantic visibility gate; higher
severities can remove task evidence. Source validation remains under
`source_validation`, distinct from any delivered-image certificate.

## Curate and partition

Prepare an external eligibility JSON object containing `eligible_triplet_ids`
and `agent_primary_visual_review_units`. Core eligibility follows the original
three tiers: successful intervention, raw metadata, exact controlled state,
nonempty disjoint causal action sets, nuisance branch preservation, visibility
thresholds, and retained image signal. Selection excludes evaluated-model outputs.

Record per-triplet visual judgments using the
[curation protocol](../data/review/). Then assemble the two disjoint collections:

```bash
uv run python -m construction.partition --eligibility /outside/review-selection.json \
  --visual-review data/review/proxy_visual_review.jsonl
```

The assembler verifies reviewed image, metadata, state, and label hashes,
requires passing visual verdicts for reviewed core membership, and removes
obsolete classified shards. It uses the distributed annotation file by default
when present. New triplets require their own review; changed assets or labels
invalidate existing annotations. Source construction alone records automated
checks and does not supply human-review evidence.

The resulting catalogs, counts, and `paper/requirements.json` record actual
membership, predicate coverage, and review provenance. Curation annotations are
part of the dataset. Temporary contact sheets, review tools, validation reports,
and model evaluation outputs remain outside the repository.

## Add simulator nuisances

The reference suite contains simulator illumination and 14 image-space profiles,
with viewpoint and material variants indexed separately. Additional screened
profiles include motion blur, chromatic aberration, sensor banding, shot noise,
and digital scaling. The shot-noise profile uses a normal approximation to
Poisson photon noise and shares a standardized spatial field across Anchor and
Causal. Digital scaling retains image content, adds gray framing, and updates
bounding boxes. Parameters are stored per replica.

The task-irrelevant-object constructor opens a visible, empty Drawer beside a
vessel inspection. It checks all non-Drawer state, target visibility, and the
preserved action branch:

```bash
xvfb-run -a uv run --extra simulator python -m construction.expand_simulator \
  --output /tmp/causalpilot-bedroom --scenes FloorPlan301 --predicates isFilledWithLiquid --limit 2

xvfb-run -a uv run --extra simulator python -m construction.irrelevant_object \
  --dataset /tmp/causalpilot-bedroom/dataset.json \
  --output /tmp/causalpilot-irrelevant
```

Pass the output to the source builder with
`--additional-triplets /tmp/causalpilot-irrelevant/dataset.json`. This adds a
replica to an existing unit. Image-space profiles apply the same light transform
to Anchor/Causal and a stronger transform to the original Anchor. They reference
source symbolic states and raw metadata, with post-processing recorded separately.

## Finalize integrity

After assembling and reviewing the release, regenerate its file manifest:

```bash
uv run python -m construction.checksums
```

The manifest hashes distributed files with SHA-256. Construction tools do not
call a model API. The submission package contains data, construction code, and
provenance; benchmark test programs and model results remain external.
