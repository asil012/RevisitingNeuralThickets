from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch

from neural_thickets.types import PerturbationSemantics, PerturbationSpec

NamedParameters = Iterable[tuple[str, torch.nn.Parameter]]


@dataclass
class PerturbationState:
    """Anchor snapshot required by exact-restoration perturbations."""

    anchor: dict[str, torch.Tensor]
    active: PerturbationSpec | None = None

    @classmethod
    def capture(
        cls,
        parameters: NamedParameters,
        *,
        storage: str = "same_device",
    ) -> PerturbationState:
        assert storage in {"same_device", "cpu"}
        anchor = {
            name: (
                parameter.detach().clone()
                if storage == "same_device"
                else parameter.detach().cpu().clone()
            )
            for name, parameter in parameters
        }
        assert anchor
        return cls(anchor=anchor)


def apply_perturbation(
    parameters: NamedParameters,
    spec: PerturbationSpec,
    semantics: PerturbationSemantics,
    *,
    state: PerturbationState | None = None,
) -> None:
    """Apply a RandOpt perturbation in-place with explicit historical semantics."""

    items = list(parameters)
    assert items
    assert state is None or state.active is None
    if semantics is not PerturbationSemantics.PAPER_TIED_DRIFTING:
        assert state is not None

    generators: dict[str, torch.Generator] = {}
    with torch.no_grad():
        for name, parameter in items:
            generator = _generator(parameter, spec.seed, semantics, generators)
            noise = torch.randn(
                parameter.shape,
                dtype=parameter.dtype,
                device=parameter.device,
                generator=generator,
            )
            parameter.add_(spec.sigma * noise)
    if state is not None:
        state.active = spec


def restore_anchor(
    parameters: NamedParameters,
    semantics: PerturbationSemantics,
    *,
    state: PerturbationState | None = None,
    spec: PerturbationSpec | None = None,
) -> None:
    """Restore the anchor using the selected experiment's actual arithmetic."""

    items = list(parameters)
    assert items
    if semantics is PerturbationSemantics.PAPER_TIED_DRIFTING:
        assert spec is not None
        generators: dict[str, torch.Generator] = {}
        with torch.no_grad():
            for _, parameter in items:
                generator = _generator(parameter, spec.seed, semantics, generators)
                noise = torch.randn(
                    parameter.shape,
                    dtype=parameter.dtype,
                    device=parameter.device,
                    generator=generator,
                )
                parameter.add_(-spec.sigma * noise)
        return

    assert state is not None
    assert state.active is not None
    _copy_anchor(items, state)
    state.active = None


def _copy_anchor(items: list[tuple[str, torch.nn.Parameter]], state: PerturbationState) -> None:
    assert set(state.anchor) == {name for name, _ in items}
    with torch.no_grad():
        for name, parameter in items:
            parameter.copy_(state.anchor[name], non_blocking=False)


def _generator(
    parameter: torch.nn.Parameter,
    seed: int,
    semantics: PerturbationSemantics,
    generators: dict[str, torch.Generator],
) -> torch.Generator:
    if semantics in {
        PerturbationSemantics.PAPER_TIED_DRIFTING,
        PerturbationSemantics.PAPER_TIED_EXACT,
    }:
        return torch.Generator(device=parameter.device).manual_seed(seed)
    assert semantics is PerturbationSemantics.IID_EXACT
    key = str(parameter.device)
    if key not in generators:
        generators[key] = torch.Generator(device=parameter.device).manual_seed(seed)
    return generators[key]

