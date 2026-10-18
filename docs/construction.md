# Reconstruct and extend

Run commands from the repository root. Python 3.10+ and `uv` are required.
The lightweight reader/renderer uses Pillow and NumPy. Simulator generation
additionally requires Linux, Xvfb, a working graphics driver, and AI2-THOR.

```bash
uv sync --extra simulator
```

## Build additional predicate units

Choose an output directory **outside this repository** so candidate rejection
logs and working artifacts are not included in the data release:

```bash
xvfb-run -a uv run --extra simulator python -m construction.expand_simulator \
  --output /tmp/causalpilot-extension \
  --scenes FloorPlan1 FloorPlan2 FloorPlan3 FloorPlan4 FloorPlan5 FloorPlan301 FloorPlan401 FloorPlan402 FloorPlan403 \
  --limit 40
```

The generator uses AI2-THOR 5.0.0 with pinned Unity build
`f0825767cd50d69f666c7f282e54abfe58f1e917`. Each accepted unit saves unmodified
metadata, images, exact inspection pose, interventions, the evidence contract,
and a recipe. It resumes accepted units from `unit.json`. To change generation
rules, use a new output directory instead of mixing cached units from older
rules. Failed candidates are retained only in the external working directory.

Newer recipes also save the movable/pickupable pose list. Small explicitly
declared float tolerances apply only to geometry; all Boolean and relational
state remains exact. See the dataset card for the numerical contract.

## Assemble the classified release

```bash
uv run python -m construction.build_release \
  --legacy-benchmark data/benchmark.json \
  --legacy-states data/causal_states.json \
  --extension /tmp/causalpilot-extension/dataset.json \
  --paper /path/to/all-finish.pdf
```

The builder preserves archived image bytes and labels, imports accepted new
units, generates nuisance replicas, writes classified shards and indexes, and
groups optional partitions by physical scene. `--paper` binds the preserved
identity manifest to the manuscript via SHA-256; the PDF is not uploaded.

Multiple independently constructed folders can be passed after `--extension`.
After assembling the final files, update their integrity manifest:

```bash
uv run python -m construction.checksums
```

The original generation pipeline is retained in `AsgardBench/causal/`:
`generator`, `extended`, `viewpoint`, `appearance`, and `render_replicas`.
Those modules contain historical construction conventions and are not a claim
that all historical candidates passed the final manuscript contract. New
assembly uses the explicit provenance and review fields in this release.

## Nuisance construction

The existing suite includes simulator lighting and 14 deterministic image-space
profiles. Optional archived simulator viewpoint/material subsets remain intact.
The extension adds horizontal motion blur, chromatic aberration, sensor banding,
signal-dependent shot noise (a normal approximation to Poisson photon noise),
and digital view scaling with gray framing. The signal-dependent profile uses
the same standardized spatial noise field across Anchor/Causal, so a changed
target pixel cannot shift the random-number stream of unrelated background pixels.
Digital scaling retains all image content and updates target bounding boxes;
it is an image-space framing change, not a simulated camera move.
Parameters are stored per replica.

For a task-irrelevant object-state nuisance, the targeted constructor opens a
visible, empty Drawer beside a vessel inspection. It verifies reconstruction,
all non-Drawer state, target visibility, and preserved action labels:

```bash
xvfb-run -a uv run --extra simulator python -m construction.expand_simulator \
  --output /tmp/causalpilot-bedroom --scenes FloorPlan301 --predicates isFilledWithLiquid --limit 2

xvfb-run -a uv run --extra simulator python -m construction.irrelevant_object \
  --dataset /tmp/causalpilot-bedroom/dataset.json \
  --output /tmp/causalpilot-irrelevant
```

Pass its output to the release builder with
`--additional-triplets /tmp/causalpilot-irrelevant/dataset.json`. It references
an existing unit and does not increase the unit count.

The same light transform is applied to Anchor/Causal. The stronger nuisance
transform is applied to the original Anchor. Image-space replicas reuse the
certified source symbolic snapshot and reference its original raw metadata;
metadata does not claim that post-processing occurred inside the simulator.

Construction does not call a model API. New data stays `automated_checks_only`;
numerical visual-signal gates do not substitute for the manual review described
in the paper. Test scripts, local validation reports, model outputs, credentials,
and intermediate rejected candidates are excluded from the repository.
