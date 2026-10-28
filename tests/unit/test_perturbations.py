import pytest

torch = pytest.importorskip("torch")

from neural_thickets.perturbations.weights import (  # noqa: E402
    PerturbationState,
    apply_perturbation,
    restore_anchor,
)
from neural_thickets.types import PerturbationSemantics, PerturbationSpec  # noqa: E402


def _parameters() -> list[tuple[str, torch.nn.Parameter]]:
    return [
        ("layer.0", torch.nn.Parameter(torch.ones(64, dtype=torch.bfloat16))),
        ("layer.1", torch.nn.Parameter(torch.ones(64, dtype=torch.bfloat16))),
        ("head", torch.nn.Parameter(torch.ones(31, dtype=torch.bfloat16))),
    ]


def test_paper_noise_is_tied_and_exact_restore_is_exact() -> None:
    parameters = _parameters()
    originals = {name: value.detach().clone() for name, value in parameters}
    state = PerturbationState.capture(parameters)
    spec = PerturbationSpec(index=0, seed=1234, sigma=0.001)
    apply_perturbation(parameters, spec, PerturbationSemantics.PAPER_TIED_EXACT, state=state)
    left = parameters[0][1].detach() - originals["layer.0"]
    right = parameters[1][1].detach() - originals["layer.1"]
    assert torch.equal(left, right)
    restore_anchor(parameters, PerturbationSemantics.PAPER_TIED_EXACT, state=state)
    assert all(torch.equal(value, originals[name]) for name, value in parameters)


def test_paper_tied_arithmetic_matches_materialized_reference_expression() -> None:
    seed = 9182
    sigma = 0.001
    anchor = torch.linspace(-1, 1, 4096, dtype=torch.bfloat16)
    parameter = torch.nn.Parameter(anchor.clone())
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(
        parameter.shape,
        dtype=parameter.dtype,
        generator=generator,
    )
    expected = anchor + sigma * noise
    state = PerturbationState.capture([("weight", parameter)])
    apply_perturbation(
        [("weight", parameter)],
        PerturbationSpec(index=0, seed=seed, sigma=sigma),
        PerturbationSemantics.PAPER_TIED_EXACT,
        state=state,
    )
    assert torch.equal(parameter, expected)


def test_iid_noise_advances_one_generator_across_tensors() -> None:
    parameters = _parameters()
    originals = {name: value.detach().clone() for name, value in parameters}
    state = PerturbationState.capture(parameters)
    spec = PerturbationSpec(index=0, seed=1234, sigma=0.001)
    apply_perturbation(parameters, spec, PerturbationSemantics.IID_EXACT, state=state)
    left = parameters[0][1].detach() - originals["layer.0"]
    right = parameters[1][1].detach() - originals["layer.1"]
    assert not torch.equal(left, right)
    restore_anchor(parameters, PerturbationSemantics.IID_EXACT, state=state)
    assert all(torch.equal(value, originals[name]) for name, value in parameters)


def test_iid_arithmetic_matches_materialized_audit_expression() -> None:
    seed = 9182
    sigma = 0.001
    anchors = [
        torch.linspace(-1, 1, 257, dtype=torch.bfloat16),
        torch.linspace(1, -1, 257, dtype=torch.bfloat16),
    ]
    parameters = [
        (f"weight.{index}", torch.nn.Parameter(anchor.clone()))
        for index, anchor in enumerate(anchors)
    ]
    generator = torch.Generator().manual_seed(seed)
    expected = [
        anchor
        + sigma
        * torch.randn(anchor.shape, dtype=anchor.dtype, generator=generator)
        for anchor in anchors
    ]
    state = PerturbationState.capture(parameters)
    apply_perturbation(
        parameters,
        PerturbationSpec(index=0, seed=seed, sigma=sigma),
        PerturbationSemantics.IID_EXACT,
        state=state,
    )
    assert all(
        torch.equal(parameter, target)
        for (_, parameter), target in zip(parameters, expected)
    )


def test_reference_restore_regenerates_and_subtracts_noise() -> None:
    generator = torch.Generator().manual_seed(9)
    parameter = torch.nn.Parameter(
        torch.randn(4096, generator=generator, dtype=torch.bfloat16)
    )
    original = parameter.detach().clone()
    parameters = [("weight", parameter)]
    spec = PerturbationSpec(index=0, seed=4321, sigma=0.002)
    noise = torch.randn(
        parameter.shape,
        generator=torch.Generator().manual_seed(spec.seed),
        dtype=parameter.dtype,
    )
    expected = original + spec.sigma * noise
    expected = expected + (-spec.sigma * noise)
    apply_perturbation(
        parameters,
        spec,
        PerturbationSemantics.PAPER_TIED_DRIFTING,
    )
    restore_anchor(
        parameters,
        PerturbationSemantics.PAPER_TIED_DRIFTING,
        spec=spec,
    )
    assert torch.equal(parameter, expected)
    assert not torch.equal(parameter, original)


@pytest.mark.parametrize("storage", ["same_device", "cpu"])
def test_anchor_storage_modes_restore_without_intermediate_conversion(storage: str) -> None:
    device = "cuda" if storage == "cpu" and torch.cuda.is_available() else "cpu"
    parameter = torch.nn.Parameter(torch.arange(128, device=device, dtype=torch.bfloat16))
    original = parameter.detach().clone()
    state = PerturbationState.capture([("weight", parameter)], storage=storage)
    if storage == "cpu":
        assert state.anchor["weight"].device.type == "cpu"
    spec = PerturbationSpec(index=0, seed=12, sigma=0.001)
    apply_perturbation(
        [("weight", parameter)],
        spec,
        PerturbationSemantics.PAPER_TIED_EXACT,
        state=state,
    )
    restore_anchor(
        [("weight", parameter)],
        PerturbationSemantics.PAPER_TIED_EXACT,
        state=state,
    )
    assert torch.equal(parameter, original)

