# 扩展集

95 个来源 causal units / 11,865 triplets / 35,595 observations，覆盖七类谓词。
与 paper 的 triplet IDs 不重叠，但可以共享 unit；同一 unit 的图像变体不是独立样本。

入口：[catalog.json](catalog.json)；批量部分：[bulk_catalog.json](bulk_catalog.json)。

本轮代理复核移出的 23 条也保留在这里，带 `proxy_visual_review.verdict="reject"`
及具体原因。其他未审核扩展不因共享来源而自动获得审核通过状态。

批量扩展包含 20 类 nuisance、每类 6 档强度，共 11,040 triplets（33,120 张真实图像文件）。
来源为 92 个已有 simulator primary units；FloorPlan216、205、214 的新建电视
units 另有既有构建流程的额外 profiles。批量图像为 256×256 JPEG，质量参数 82。

批量部分遵照用户“扩展数量可以很多、质量没有要求”的指示，不做语义质量筛选，
标为 `unreviewed_expansion` / `extension-unreviewed-render-v1`。动作标签继承来源，
强扰动可能遮蔽或破坏决策证据。来源 validation 单独保存在 `source_validation`，
不作为输出图的可见性或质量认证；source bbox 的坐标属于来源原图。

扰动包括亮度、对比度、饱和度、gamma、色温、Gaussian/motion blur、JPEG 压缩、
固定 RGB 量化、降采样、暗角、照明梯度、Gaussian/shot noise、条带、色散、
平移、缩放、遮挡与像素丢失。随机场与轻变换在 Anchor/Causal 间共享，
Nuisance 从来源 Anchor 生成；每条保留 family、severity、参数、seed 和来源图 hash。

液体、切片及耗材的来源证据限制见[论文对齐说明](../../docs/paper_alignment.md)。
`isUsedUp` 的 raw metadata 始终为 false，不能当成原始 Boolean 认证；
默认读取器排除这类来源及其批量后代，显式 `metadata_only=False` 才包含。

按场景浏览：[access](scenarios/access/) · [cleaning](scenarios/cleaning/) ·
[device_control](scenarios/device_control/) · [food_preparation](scenarios/food_preparation/) ·
[liquid_and_consumables](scenarios/liquid_and_consumables/)。
所有数据路径相对于顶层 `data/`。
