# CausalPilot-Bench

Anchor / Causal / Nuisance 三元观测 benchmark，按场景、谓词类型和物理房间分类。

数据分为两个互不重复的 triplet 集合：

| 集合 | Causal units | Triplets | Observations | 谓词类型 |
| --- | ---: | ---: | ---: | ---: |
| [论文规模候选集 paper](data/paper/) | 71 | 996 | 2,988 | 4 |
| [扩展集 extended](data/extended/) | 95 | 11,865 | 35,595 | 7 |

**已按用户要求完成全部论文候选的模拟人工复核，实际审核者为 Codex 代理。**
逐条核查原始元数据、状态、动作、可见性及图像信号，并逐图查看 Anchor/Causal/Nuisance。
本轮查看 1,036 条候选，23 条因熟化表面不可见、水面反光遮蔽或强噪声破坏线索而移入扩展；
用严格重建并经过逐图复核的候选补足，最终仍为 71 units / 996 triplets。
[逐条质量标注与复核流程](data/review/)绑定原图、元数据、状态快照和标签的哈希。

**仍不能标为完全符合论文。** 液体、切片、耗材三类尚无法通过论文原三类认证规则，
没有放宽规则补类别，也没有将代理复核写成真实人类审核。
具体口径见 [paper/requirements.json](data/paper/requirements.json)。
此前排除的 FloorPlan224 侧面电视继续只留在扩展/归档；新建电视 units 位于
FloorPlan216、205、214。候选成员有变动，不能直接沿用论文原实验指标。

扩展集包含七类谓词、额外独立 units，以及 **20 类 × 6 档 × 92 个来源 units =
11,040 条**新增图像扰动 triplets。批量部分为 256×256 JPEG，标为
`unreviewed_expansion`，继承来源动作标签，允许强扰动抹去视觉证据，不声称质量合格。

两个集合共享部分 units：更多扰动不增加独立 causal unit 数。总计 **95 个独立
units、12,861 个 triplets、38,583 个 observations**。
历史 71 units / 996 triplets 的 ID、标签和原图完整保存在归档清单，且都仍在完整发布中。

```text
data/
  paper/                     # 论文规模候选清单、要求缺口、分类入口
  extended/                  # 扩展清单、批量图像、分类入口
  review/                    # 样本质量标注、代理复核流程与哈希绑定
  catalog.json               # 两个集合的完整入口
  counts.json                # 精确数量，含两个集合的分别统计
  source_catalog.json        # 批量扩展前的可追溯构建入口
  scenarios/                 # 共享原始图像与来源分类清单
  unit_index.json            # 独立 unit 索引与集合归属数量
  benchmark.json             # 历史 996 triplets 归档；不是认证合格声明
  causal_states.json         # 历史 71 units 归档
simulation/                  # 原始元数据、状态快照、构建轨迹和配方
construction/                # 数据读取与扩展构建代码
```

所有清单中的数据路径均相对于顶层 `data/`；两个集合用清单引用共享资产。

```bash
uv sync
```

```python
from construction.load import iter_triplets, observation_input

for row in iter_triplets("data", collection="paper"):
    sample = observation_input(row, "anchor", "data")

# isUsedUp 含明确标注的轨迹派生状态，须显式选择才会返回：
for row in iter_triplets("data", collection="extended", metadata_only=False):
    pass
```

| 语义场景 | 谓词 |
| --- | --- |
| 清洁 | `isDirty` |
| 容器访问 | `isOpen` |
| 设备控制 | `isToggled` |
| 食物加工 | `isCooked`, `isSliced` |
| 液体与耗材 | `isFilledWithLiquid`, `isUsedUp` |

[论文对齐与限制](docs/paper_alignment.md) · [数据说明](docs/dataset_card.md) ·
[扩展构建](docs/construction.md) · [格式说明](docs/schema.md)

仓库只发布数据、构建代码和数据来源说明；测试程序、模型推理与相应测试结果均不发布。
构建辅助代码源于 [Microsoft AsgardBench](https://github.com/microsoft/AsgardBench)，
保留 MIT [LICENSE](LICENSE)。AI2-THOR 资产与生成视觉内容仍适用上游条款。
