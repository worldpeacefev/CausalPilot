# 论文规模候选集

71 causal-state units / 996 triplets / 2,988 observations；当前四类目标谓词。

入口：[catalog.json](catalog.json)；认证缺口：[requirements.json](requirements.json)。

全部 996 条经过原始元数据、精确状态差分、动作分支、可见性、bbox 与图像信号核查，
并按用户要求逐图完成模拟人工复核。实际审核者为 Codex 代理，不是真实人类审核。
[逐条质量标注和流程](../review/)可追溯到每个图像、元数据、状态与标签的哈希。

本轮复核 1,036 条候选，23 条未通过：1 条面包熟化表面被遮住、15 条碗样本的
脏净线索被水面反光遮住、7 条强噪声样本丢失细微表面线索。未通过者移入扩展，
用严格重建且逐图复核通过的样本补足规模。

使用论文表 4 原三种 tiers，未引入液体联动、切片拓扑或耗材轨迹例外。
`isFilledWithLiquid`、`isSliced`、`isUsedUp` 仍不能按原规则认证，
因此 `fully_conforms_to_paper=false`。代理复核完成不等于七类认证缺口消失。

新候选包含 971 条历史 triplets 与 25 条新增 triplets。成员变动后，原论文指标
不能直接沿用；原始 996 条另有完整归档。

按场景浏览：[access](scenarios/access/) · [cleaning](scenarios/cleaning/) ·
[device_control](scenarios/device_control/) · [food_preparation](scenarios/food_preparation/)。
所有数据路径相对于顶层 `data/`，而非本目录。
