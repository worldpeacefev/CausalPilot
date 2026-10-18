# 对齐 `all-finish.pdf`

依据 2026-10-19 提供的最终 PDF：正文 §3.1–3.4、附录 A.1/A.2、表 4/5
及附录 B。原文件的 SHA-256 记录于 `data/paper_identity.json`。仓库不附带论文 PDF。

## 保留且统一的口径

- 独立计数单位为 causal-state unit；多 nuisance 的 triplets 属于同一 unit。
- 每个 triplet 都有 Anchor、Causal、Nuisance 三个 observation；固定 instruction、
  history、action interface 和可用物体类型上下文。
- Anchor 与 Causal 的非空 acceptable-action sets 不相交；Nuisance 与 Anchor
  的 acceptable-action sets 相同。
- 图像变换采用 `Anchor=L_r(original_anchor)`、
  `Causal=L_r(original_causal)`、`Nuisance=S_r(original_anchor)`。
  两个 causal 观测共享同一个轻变换。
- 五类语义场景为 cleaning、access、device control、food preparation、
  liquid/consumable availability。物理房间和 FloorPlan 是另一分类维度。
- 冻结保存历史 71 units、996 triplets、2,988 observations 的全部 ID、标签与
  图像内容。为方便分类调整了文件位置，并更新路径；原图 hash 可逐一追溯。
- 论文的主估计量是 pooled-triplet weighting；置信区间应对 unit 聚类。
  unit-balanced weighting 是另一个估计量，不能混用。

## 论文与历史发布文件的实际差异

历史仓库提交 `4946240` 的内容，经逐记录统计如下：

| Predicate | Historical units | Historical triplets |
| --- | ---: | ---: |
| `isCooked` | 6 | 62 |
| `isDirty` | 11 | 174 |
| `isOpen` | 31 | 447 |
| `isToggled` | 23 | 313 |
| **Total** | **71** | **996** |

1. **七类谓词没有全部出现在历史实验集。** `isSliced`、`isFilledWithLiquid`、
   `isUsedUp` 出现在论文定义和 boundary vocabulary 中，但没有出现在这 996 个
   triplets 的目标谓词里。不能仅修改 counts 或重新命名四类样本来声称已经覆盖七类。
2. **历史 tier 标签有两处错误。** 历史发布标记为 66 个
   `pilot-v3-strict` 和 5 个 `extended-small-visible-v1` units；其中
   `FloorPlan10`、`FloorPlan11` 的两个微波炉 `isOpen` unit 实际伴随目标自身
   position/rotation 变化，符合表 4 的 `extended-target-articulation-v1`。
   分类集按实际状态差异修正为 **64 strict / 2 target-articulation / 5 small-visible**，
   保留 `original_certification_tier`。历史兼容清单保留原标签便于追溯。
3. **15 是标准 profile suite，不是所有实际 profile 名称的总数。** 历史文件含
   15 个 suite profiles，以及 `simulator_viewpoint` 的 27 个 triplets 和
   `simulator_material` 的 1 个 triplet，共 17 个实际名称。Profile eligibility
   按 unit 不同，不能直接将总数解释成 `71×15`。
4. **论文中的 318 episodes、125 rejected units 缺少可完全复核的对应清单。**
   已保留仓库原有 source/candidate plans；不以计划文件数冒充论文的 source episode
   或 rejection 数量。本次没有修改论文，也没有重造这些历史数值。
5. **无法从现有材料独立确认“每个历史 triplet 均已人工审核”。** 历史数据标记为
   `paper_claim_not_independently_reverified`；新数据标记实际自动校验状态。仓库不
   生成虚假的 reviewer、人工审核日期或模型测试结果。

## 扩展如何补齐类型，同时保持证据边界

| 新增类别 | 实际依据 | 与论文 certification 的关系 |
| --- | --- | --- |
| `isFilledWithLiquid` | 原始 metadata 的 Boolean 和 `fillLiquid` | 单独的 `extension-liquid-content-v1`；允许目标液体内容随 Boolean 联动 |
| `isSliced` | 原物体的 `isSliced`、子切片 IDs、可见切片 union | 单独的 `extension-target-topology-v1`；允许目标切片及其直接互反容器关系变化 |
| `isUsedUp` | 同一目标的成功 `UseUpObject` 轨迹、可见目标和局部图像差异 | 单独的 `extension-intervention-trace-v1`；不是 raw metadata Boolean 认证 |

`isUsedUp` 的限制来自模拟器实现：[AI2-THOR SimObjPhysics.cs](https://github.com/allenai/ai2thor/blob/main/unity/Assets/Scripts/SimObjPhysics.cs)
将 `CanBeUsedUp` 和 `IsUsedUp` 访问器实现为固定 `false`；
[UsedUp.cs](https://github.com/allenai/ai2thor/blob/main/unity/Assets/Scripts/UsedUp.cs)
内部的使用耗尽行为与公开 metadata 访问器不一致。当前使用的 5.0.0 / build
`f0825767cd50d69f666c7f282e54abfe58f1e917` 中已实际复现该问题。
`SoapBottle` / `TissueBox` 还可能抛出组件异常，失败候选不会进入发布数据。

因此，本次发布在**类别目录上覆盖论文的七类，并完整包含原始 996 条**；但严格
依赖原始 metadata 的使用者应排除 trace-derived `isUsedUp`。这项已知限制不能通过
数据整理消除，论文中“71 个 units 本身覆盖七类”的历史陈述仍需作者核对。
扩展集的增加不能反向支持原论文实验数字。

所有扩展数量及 profiles 的实际覆盖均以 `data/counts.json` 为准。新增干预轨迹、
图像变换、代码和标签可以用于后续研究；本次没有声称新数据已通过完整人工审核。

本次补充了 digital view scaling，以及经完整重建核验的 task-irrelevant empty
Drawer 变化。后者只允许该空抽屉的 `isOpen` 和自身关节位姿变化，杯子状态、
其他物体、inventory、agent 和动作分支仍受控；它是既有 unit 的额外 nuisance，
不增加 causal-state unit 计数。

历史电视 unit `istoggled__turn_on_tv__Confirmatory_FloorPlan224_C1` 的 primary
目标区域像素 RMS 约为 0.312，低于本次 render 扩展的 1.0 门槛。论文的 strict
tier 本身没有规定这项 RMS 门槛，因此原条目保留并标记为弱视觉信号，等待人工复核；
不会为了让校验全绿而删除历史数据或放宽新增 replica 的数值要求。
