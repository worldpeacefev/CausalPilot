from AsgardBench.causal.evaluator import PROMPT_VARIANTS, build_prompt


def test_prompt_variants_preserve_condition_information() -> None:
    triplet = {
        "instruction": "Put away the Apple.",
        "history": ["find Apple", "pickup Apple", "find Fridge"],
        "action_interface": ["open OBJECT", "put DESTINATION_RECEPTACLE", "done none"],
    }
    variant = {"oracle_state": ["Fridge: isOpen=false", "inventory: holding Apple"]}
    prompts = [build_prompt(triplet, variant, "oracle_state", ["Apple", "Fridge"], name) for name in PROMPT_VARIANTS]
    assert len(set(prompts)) == 3
    for prompt in prompts:
        for required in (
            "Put away the Apple", "find Apple", "Fridge", "isOpen=false",
            "put", "open", "done none", '"action"', '"object"',
        ):
            assert required in prompt
