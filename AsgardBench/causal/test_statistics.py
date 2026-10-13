from AsgardBench.causal.statistics import exact_sign_pvalue, holm_adjust, paired_unit_contrast


def _rows(method: str, outcomes: dict[str, tuple[bool, bool, bool]]) -> list[dict]:
    rows = []
    for unit, values in outcomes.items():
        predictions = ("open fridge", "put fridge", "open fridge")
        for variant, correct, prediction in zip(("anchor", "causal", "nuisance"), values, predictions):
            rows.append({
                "model": "m", "method": method, "triplet_id": unit,
                "causal_state_unit": unit, "variant": variant,
                "correct": correct, "prediction": prediction,
            })
    return rows


def test_paired_contrast_counts_units_not_rows() -> None:
    direct = _rows("direct", {"u1": (True, False, True), "u2": (True, False, True)})
    bgcp = _rows("bgcp", {"u1": (True, True, True), "u2": (True, False, True)})
    result = paired_unit_contrast(direct, bgcp, "causal_response", iterations=100)
    assert result["n_units"] == 2
    assert result["difference"] == 0.5
    assert (result["wins"], result["losses"], result["ties"]) == (1, 0, 1)


def test_exact_sign_and_holm() -> None:
    assert exact_sign_pvalue(7, 0) == 0.015625
    assert holm_adjust([0.01, 0.04, None]) == [0.02, 0.04, None]
