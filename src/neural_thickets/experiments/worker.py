from __future__ import annotations

import hashlib
import json
import random

import numpy as np
import torch

from neural_thickets.perturbations.weights import (
    PerturbationState,
    apply_perturbation,
    restore_anchor,
)
from neural_thickets.types import PerturbationSemantics, PerturbationSpec


class PerturbationWorker:
    """vLLM worker extension implementing every perturbation arm in the audit."""

    def initialize_perturbations(
        self,
        semantics: str,
        anchor_storage: str,
    ) -> dict[str, object]:
        assert not hasattr(self, "_thicket_semantics")
        self._thicket_semantics = PerturbationSemantics(semantics)
        parameters = list(self.model_runner.model.named_parameters())
        assert parameters and all(parameter.dtype == torch.bfloat16 for _, parameter in parameters)
        drifting = self._thicket_semantics is PerturbationSemantics.PAPER_TIED_DRIFTING
        assert (anchor_storage == "none") is drifting
        self._thicket_state = None if drifting else PerturbationState.capture(
            parameters,
            storage=anchor_storage,
        )
        self._thicket_active = None
        self._thicket_anchor_fingerprint = self.weight_fingerprint("sampled")
        return {
            "semantics": self._thicket_semantics.value,
            "anchor_fingerprint": self._thicket_anchor_fingerprint,
            "parameter_order_sha256": _parameter_order_sha256(parameters),
            "anchor_storage": anchor_storage,
        }

    def initialize_anchor_audit(self) -> dict[str, object]:
        parameters = list(self.model_runner.model.named_parameters())
        return {
            "anchor_fingerprint": self.weight_fingerprint("sampled"),
            "parameter_order_sha256": _parameter_order_sha256(parameters),
        }

    def perturb(self, index: int, seed: int, sigma: float) -> bool:
        assert self._thicket_active is None
        spec = PerturbationSpec(index=int(index), seed=int(seed), sigma=float(sigma))
        if self._thicket_semantics is not PerturbationSemantics.IID_EXACT:
            _set_global_seed(spec.seed)
        apply_perturbation(
            self.model_runner.model.named_parameters(),
            spec,
            self._thicket_semantics,
            state=self._thicket_state,
        )
        torch.cuda.synchronize()
        self._thicket_active = spec
        return True

    def restore(self, verify: bool = False) -> dict[str, object] | bool:
        assert self._thicket_active is not None
        if self._thicket_semantics is not PerturbationSemantics.IID_EXACT:
            _set_global_seed(self._thicket_active.seed)
        restore_anchor(
            self.model_runner.model.named_parameters(),
            self._thicket_semantics,
            state=self._thicket_state,
            spec=self._thicket_active,
        )
        torch.cuda.synchronize()
        self._thicket_active = None
        if not verify:
            return True
        fingerprint = self.weight_fingerprint("sampled")
        if self._thicket_semantics is not PerturbationSemantics.PAPER_TIED_DRIFTING:
            assert fingerprint == self._thicket_anchor_fingerprint
        return fingerprint

    def weight_fingerprint(self, mode: str = "sampled") -> dict[str, object]:
        assert mode in {"sampled", "full"}
        tensors = []
        for name, parameter in self.model_runner.model.named_parameters():
            data = parameter.detach().reshape(-1)
            if mode == "sampled" and data.numel() > 256:
                indices = torch.linspace(
                    0,
                    data.numel() - 1,
                    steps=256,
                    device=data.device,
                    dtype=torch.float64,
                ).long()
                data = data[indices]
            data = data.contiguous().cpu()
            digest = hashlib.sha256(data.view(torch.uint8).numpy().tobytes()).hexdigest()
            tensors.append((name, list(parameter.shape), str(parameter.dtype), digest))
        total = hashlib.sha256(json.dumps(tensors, sort_keys=True).encode()).hexdigest()
        return {"mode": mode, "sha256": total, "tensor_count": len(tensors)}


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _parameter_order_sha256(
    parameters: list[tuple[str, torch.nn.Parameter]],
) -> str:
    order = hashlib.sha256()
    for name, parameter in parameters:
        order.update(name.encode())
        order.update(str(tuple(parameter.shape)).encode())
        order.update(str(parameter.dtype).encode())
    return order.hexdigest()

