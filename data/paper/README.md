# 论文规模候选集

71 causal-state units / 996 triplets / 2,988 observations；当前只有四类目标谓词。

入口：[catalog.json](catalog.json)；认证缺口：[requirements.json](requirements.json)。

入选条目经过原始元数据、精确状态差分、动作分支、可见性、bbox 和图像信号逐条核查；
每个 unit 的 primary 三元图经过 Codex 代理视觉审核。使用论文表 4 原来的三种 tiers，
未添加液体联动、切片拓扑或耗材轨迹例外。代理视觉审核不记作人工审核。

仍缺少原规则下的 `isFilledWithLiquid`、`isSliced`、`isUsedUp`，且没有可核实的
每条生成 triplet 人工复核记录。因此 `fully_conforms_to_paper=false`，
本目录不能作为“已满足全部论文要求”的认证发布集。

历史侧面电视 unit 被排除，以另一个场景中新重建的严格 unit 替换；成员变动意味着
原论文结果不能直接用于本候选集。原始 996 条另有完整归档。

按场景浏览：[access](scenarios/access/) · [cleaning](scenarios/cleaning/) ·
[device_control](scenarios/device_control/) · [food_preparation](scenarios/food_preparation/)。
所有数据路径相对于顶层 `data/`，而非本目录。
