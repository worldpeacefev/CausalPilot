# CausalPilot-Bench

Matched **Anchor / Causal / Nuisance** observations for diagnosing whether the
next embodied action follows decision-critical visual state.

本仓库按 `all-finish.pdf` 的定义整理，同时保留原始发布集，并提供明确分层的扩展数据。
全部数据以实际文件为准；[data/counts.json](data/counts.json) 给出精确数量，
[data/catalog.json](data/catalog.json) 是完整发布集入口。

**论文与历史数据存在覆盖差异。** 论文写了七类谓词，但历史 71 个 causal-state
units / 996 个 triplets 实际只有四类。本次完整保留这部分数据，并单独加入缺失谓词的
扩展；扩展不冒充原论文实验数据。`isUsedUp` 的模拟器元数据目前是常量 `false`，
所以该类采用独立标注的干预轨迹层，默认读取器不包含这一层。
详见[论文对齐说明](docs/paper_alignment.md)。

## Browse by scenario, predicate, and physical scene

| Semantic scenario | Predicate | Directory |
| --- | --- | --- |
| Cleaning / 清洁 | `isDirty` | [cleaning](data/scenarios/cleaning) |
| Access / 容器访问 | `isOpen` | [access](data/scenarios/access) |
| Device control / 设备控制 | `isToggled` | [device_control](data/scenarios/device_control) |
| Food preparation / 食物加工 | `isCooked`, `isSliced` | [food_preparation](data/scenarios/food_preparation) |
| Liquid / consumables / 液体与耗材 | `isFilledWithLiquid`, `isUsedUp` | [liquid_and_consumables](data/scenarios/liquid_and_consumables) |

```text
data/
  catalog.json                  # all classified shards and counts
  counts.json                   # units, triplets, profiles, tiers
  unit_index.json               # one entry per causal-state unit
  splits.json                   # optional partitions grouped by physical scene
  paper_identity.json           # archived IDs, labels, original image hashes
  benchmark.json                # compatibility manifest: archived 996 triplets
  causal_states.json            # compatibility manifest: archived 71 units
  scenarios/<aspect>/<predicate>/<FloorPlan>/
    triplets.jsonl              # classified records, including extensions
    <causal_state_unit>/<nuisance_profile>/{anchor,causal,nuisance}.png
simulation/
  metadata/                     # unmodified simulator observations
  states/                       # content-addressed symbolic snapshots
  recipes/                      # extension reconstruction instructions
  plans/                        # archived source trajectories
construction/                   # construction, nuisance transforms, release assembly
```

## Load data

```bash
uv sync
```

```python
from construction.load import iter_triplets, observation_input

for triplet in iter_triplets("data", predicate="isSliced"):
    sample = observation_input(triplet, "anchor", "data")
    # sample contains fixed context and one image; labels stay in triplet.
```

`iter_triplets(..., metadata_only=False)` explicitly includes trace-derived
consumable extensions. Filter `partition="paper_archive"` for the preserved
996-triplet population. Read [the schema](docs/schema.md) before implementing
an adapter; paths in every shard are relative to `data/`.

## Construction and interpretation

- [Paper alignment and unresolved discrepancies](docs/paper_alignment.md)
- [Dataset card, certification tiers, and review status](docs/dataset_card.md)
- [Reconstruction and extension commands](docs/construction.md)
- [Record schema and metric definitions](docs/schema.md)

A render replica is **not** an additional causal-state unit. Keep all replicas
of a unit together when estimating uncertainty. New extensions have automated
checks and retain `review_status="automated_checks_only"`; human review is not
claimed. Model inference, test programs, and evaluation outputs are excluded
from this release.

The simulator construction helpers originate in
[Microsoft AsgardBench](https://github.com/microsoft/AsgardBench); retain the
included MIT [LICENSE](LICENSE). AI2-THOR assets and generated visual content
remain subject to their upstream terms; the MIT code license does not grant
additional rights to simulator assets.
