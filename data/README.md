# Classified benchmark

92 units, 1761 triplets, 5283 observations across 45 physical scenes.

| Predicate | Archived units | All units | All triplets |
| --- | ---: | ---: | ---: |
| [isDirty](scenarios/cleaning/isDirty/) | 11 | 11 | 228 |
| [isOpen](scenarios/access/isOpen/) | 31 | 31 | 602 |
| [isToggled](scenarios/device_control/isToggled/) | 23 | 23 | 422 |
| [isCooked](scenarios/food_preparation/isCooked/) | 6 | 6 | 90 |
| [isSliced](scenarios/food_preparation/isSliced/) | 0 | 7 | 139 |
| [isFilledWithLiquid](scenarios/liquid_and_consumables/isFilledWithLiquid/) | 0 | 12 | 240 |
| [isUsedUp](scenarios/liquid_and_consumables/isUsedUp/) | 0 | 2 | 40 |

The archived 996 triplets remain intact. Extension tiers and review status are explicit; trace-derived consumables are excluded by the default loader. See [the dataset card](../docs/dataset_card.md).
