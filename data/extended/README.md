# Expanded collection

**95 source causal-state units · 11,865 triplets · 35,595 observations · 7 predicates**

Explore more decision boundaries and a wider range of visual conditions through
a single catalog. The collection spans all five semantic aspects and adds a
parameterized stress suite for systematic robustness studies.

[Open the catalog](catalog.json) · [Bulk suite](bulk_catalog.json) ·
[Build your own extensions](../../docs/construction.md)

## Visual stress at scale

The bulk suite combines **20 nuisance families × 6 severity levels × 92 source
units**, contributing **11,040 triplets / 33,120 images**. Delivered bulk images
are 256×256 JPEGs encoded at quality 82. Each record stores its family, severity,
parameters, seed, source identity, and source-image hashes.

| Variation | Families |
| --- | --- |
| Appearance | Exposure, contrast, saturation, gamma, color temperature |
| Image formation | Gaussian blur, motion blur, JPEG compression, quantization, resolution |
| Lighting and sensors | Vignetting, illumination gradients, Gaussian noise, shot noise, sensor banding |
| Spatial and channel changes | Chromatic shift, translation, view scaling, occlusion, pixel dropout |

Anchor and Causal share the light transform and random field; the stronger
Nuisance transform starts from the source Anchor. Replicas retain their
originating unit ID for grouped analysis.

## Evidence and review layers

Simulator source records carry explicit evidence contracts. Liquid units track
the target's contents; slicing units track the target and its descendants;
consumable units use intervention-trace evidence. See the
[dataset card](../../docs/dataset_card.md) for the exact contracts.

Bulk outputs are labeled `unreviewed_expansion` with tier
`extension-unreviewed-render-v1`. They inherit source action labels and have no
semantic visibility certificate; strong perturbations may obscure task evidence.
`source_validation` describes the source image and state, including source-image
bounding boxes. It does not certify the delivered bulk image. Other records
retain their individual automated or agent-review status, including exclusions.

Trace-derived `isUsedUp` records and their bulk descendants require explicit
opt-in. Their raw metadata remains unchanged; the extension label is derived
from the documented intervention rather than a raw Boolean observation.

```python
from construction.load import iter_triplets

expanded = iter_triplets("data", collection="extended", metadata_only=False)
```

The collection has disjoint triplet IDs from the core and shares source units
with it. The combined release contains 95 unique units.

Browse [access](scenarios/access/), [cleaning](scenarios/cleaning/),
[device control](scenarios/device_control/), [food preparation](scenarios/food_preparation/),
and [liquid and consumables](scenarios/liquid_and_consumables/).
All observation paths resolve relative to the top-level `data/` directory.
