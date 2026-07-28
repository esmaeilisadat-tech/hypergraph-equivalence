"""Permutation-invariant Phase 3 models and normalized-mass Sinkhorn matching."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def _masks(
    bx: torch.Tensor, by: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    row_mask_x = (bx.abs().sum(dim=-1) > 0).to(dtype=bx.dtype)
    row_mask_y = (by.abs().sum(dim=-1) > 0).to(dtype=by.dtype)
    column_mask = (
        (bx.abs().sum(dim=1) + by.abs().sum(dim=1)) > 0
    ).to(dtype=bx.dtype)
    return row_mask_x, row_mask_y, column_mask


def _masked_mean(
    values: torch.Tensor, mask: torch.Tensor, dimension: int
) -> torch.Tensor:
    numerator = (values * mask.unsqueeze(-1)).sum(dim=dimension)
    denominator = mask.sum(dim=dimension, keepdim=True).clamp_min(1.0)
    return numerator / denominator


def hungarian_coupling(
    cost: torch.Tensor, mask_x: torch.Tensor, mask_y: torch.Tensor
) -> torch.Tensor:
    """Return a compact hard assignment with total mass one per sample."""

    coupling = torch.zeros_like(cost)
    for batch_index in range(cost.shape[0]):
        valid_x = torch.nonzero(
            mask_x[batch_index] > 0, as_tuple=False
        ).flatten()
        valid_y = torch.nonzero(
            mask_y[batch_index] > 0, as_tuple=False
        ).flatten()
        if len(valid_x) == 0 or len(valid_y) == 0:
            raise ValueError("Hungarian coupling received an empty valid submatrix")
        compact = (
            cost[batch_index][valid_x][:, valid_y].detach().cpu().numpy()
        )
        row_index, column_index = linear_sum_assignment(compact)
        mass = 1.0 / len(row_index)
        for row, column in zip(row_index, column_index):
            coupling[batch_index, valid_x[row], valid_y[column]] = mass
    return coupling


class DifferentiableIncidenceMatchingNetwork(nn.Module):
    """Legacy-compatible fixed-L1 matcher.

    This class is an algorithmic differentiable solver, not evidence of learned
    discovery. New configurations expose it as
    ``DeterministicDifferentiableMatching``.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = dict(config)
        self.model_name = str(config["model_name"])
        self.hidden_dim = int(config.get("hidden_dim", 16))
        self.beta = float(config.get("beta", 5.0))
        self.train_iters = int(config.get("train_iters", 40))
        self.eval_iters = int(config.get("eval_iters", 100))
        self.learned_encoder = bool(config.get("learned_encoder", False))
        self.learned_residual_cost = bool(
            config.get("learned_residual_cost", False)
        )
        self.sinkhorn_enabled = bool(config.get("sinkhorn_enabled", True))
        self.straight_through_hungarian = bool(
            config.get("straight_through_hungarian", False)
        )
        self.register_buffer(
            "temperature", torch.tensor(float(config.get("tau_start", 0.02)))
        )

        if self.learned_encoder:
            self.node_mlp = nn.Sequential(
                nn.Linear(2, self.hidden_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Tanh(),
            )
            self.row_mlp = nn.Sequential(
                nn.Linear(self.hidden_dim + 1, self.hidden_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        if self.learned_residual_cost:
            self.cost_mlp = nn.Sequential(
                nn.Linear(self.hidden_dim * 3, self.hidden_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_dim, 1),
            )
            self.raw_alpha = nn.Parameter(torch.tensor(-2.0))
            self.alpha_max = 0.25

        if self.sinkhorn_enabled or self.straight_through_hungarian:
            classifier_input = 2
        elif self.learned_encoder:
            classifier_input = self.hidden_dim * 2
        else:
            classifier_input = 2
        self.classifier = nn.Linear(classifier_input, 1)
        with torch.no_grad():
            if classifier_input == 2 and (
                self.sinkhorn_enabled or self.straight_through_hungarian
            ):
                self.classifier.weight.copy_(
                    torch.tensor([[-2.0, 2.0]], dtype=self.classifier.weight.dtype)
                )
                self.classifier.bias.zero_()

    @property
    def temp(self) -> float:
        return float(self.temperature.item())

    @temp.setter
    def temp(self, value: float) -> None:
        self.temperature.fill_(float(value))

    def encode_rows(
        self, bx: torch.Tensor, by: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.learned_encoder:
            raise RuntimeError("Row encoder is disabled for this model")
        row_mask_x, row_mask_y, column_mask = _masks(bx, by)
        degree_x = bx.sum(dim=1)
        degree_y = by.sum(dim=1)
        scale = (row_mask_x.sum(dim=1) + row_mask_y.sum(dim=1)).unsqueeze(-1)
        scale = scale.clamp_min(1.0)
        node_features = torch.stack(
            ((degree_x + degree_y) / scale, (degree_x - degree_y).abs() / scale),
            dim=-1,
        )
        node_embeddings = self.node_mlp(node_features)
        node_embeddings = node_embeddings * column_mask.unsqueeze(-1)
        row_size_x = bx.sum(dim=-1, keepdim=True).clamp_min(1.0)
        row_size_y = by.sum(dim=-1, keepdim=True).clamp_min(1.0)
        raw_x = torch.bmm(bx, node_embeddings) / row_size_x
        raw_y = torch.bmm(by, node_embeddings) / row_size_y
        row_x = self.row_mlp(
            torch.cat((raw_x, row_size_x / column_mask.sum(dim=1).view(-1, 1, 1).clamp_min(1.0)), dim=-1)
        )
        row_y = self.row_mlp(
            torch.cat((raw_y, row_size_y / column_mask.sum(dim=1).view(-1, 1, 1).clamp_min(1.0)), dim=-1)
        )
        return (
            row_x * row_mask_x.unsqueeze(-1),
            row_y * row_mask_y.unsqueeze(-1),
        )

    @staticmethod
    def _sinkhorn_order(
        cost: torch.Tensor,
        log_a: torch.Tensor,
        log_b: torch.Tensor,
        mask_x: torch.Tensor,
        mask_y: torch.Tensor,
        epsilon: float,
        iterations: int,
        reverse: bool,
    ) -> torch.Tensor:
        """Epsilon-scaled dual Sinkhorn updates with warm-started potentials."""

        negative = torch.tensor(-1e30, dtype=cost.dtype, device=cost.device)
        valid = mask_x.unsqueeze(-1) * mask_y.unsqueeze(1)
        f = torch.zeros_like(log_a)
        g = torch.zeros_like(log_b)
        if iterations >= 100 and epsilon < 0.1:
            schedule = [max(0.1, epsilon * 5), max(0.05, epsilon * 2.5), epsilon]
        else:
            schedule = [epsilon]
        per_stage = [iterations // len(schedule)] * len(schedule)
        for index in range(iterations % len(schedule)):
            per_stage[index] += 1

        for stage_epsilon, stage_iterations in zip(schedule, per_stage):
            for _ in range(stage_iterations):
                if reverse:
                    column_scores = (
                        -cost + f.unsqueeze(-1)
                    ) / stage_epsilon
                    column_scores = column_scores.masked_fill(valid == 0, negative)
                    g = torch.where(
                        mask_y > 0,
                        stage_epsilon
                        * (
                            log_b
                            - torch.logsumexp(column_scores, dim=1)
                        ),
                        torch.zeros_like(g),
                    )
                    row_scores = (
                        -cost + g.unsqueeze(1)
                    ) / stage_epsilon
                    row_scores = row_scores.masked_fill(valid == 0, negative)
                    f = torch.where(
                        mask_x > 0,
                        stage_epsilon
                        * (log_a - torch.logsumexp(row_scores, dim=2)),
                        torch.zeros_like(f),
                    )
                else:
                    row_scores = (
                        -cost + g.unsqueeze(1)
                    ) / stage_epsilon
                    row_scores = row_scores.masked_fill(valid == 0, negative)
                    f = torch.where(
                        mask_x > 0,
                        stage_epsilon
                        * (log_a - torch.logsumexp(row_scores, dim=2)),
                        torch.zeros_like(f),
                    )
                    column_scores = (
                        -cost + f.unsqueeze(-1)
                    ) / stage_epsilon
                    column_scores = column_scores.masked_fill(valid == 0, negative)
                    g = torch.where(
                        mask_y > 0,
                        stage_epsilon
                        * (
                            log_b
                            - torch.logsumexp(column_scores, dim=1)
                        ),
                        torch.zeros_like(g),
                    )
        log_coupling = (
            -cost + f.unsqueeze(-1) + g.unsqueeze(1)
        ) / epsilon
        return torch.exp(log_coupling.masked_fill(valid == 0, negative))

    def sinkhorn(
        self,
        cost: torch.Tensor,
        mask_x: torch.Tensor,
        mask_y: torch.Tensor,
        eps: float | None = None,
        num_iters: int | None = None,
    ) -> torch.Tensor:
        """Balanced log-domain coupling with mass one and transpose symmetry."""

        if torch.any(mask_x.sum(dim=1) <= 0) or torch.any(mask_y.sum(dim=1) <= 0):
            raise ValueError("Sinkhorn received an empty valid submatrix")
        epsilon = float(self.temp if eps is None else eps)
        iterations = int(
            self.train_iters if self.training else self.eval_iters
            if num_iters is None
            else num_iters
        )
        if num_iters is not None:
            iterations = int(num_iters)
        mass_x = mask_x.sum(dim=1, keepdim=True)
        mass_y = mask_y.sum(dim=1, keepdim=True)
        negative = torch.tensor(-1e30, dtype=cost.dtype, device=cost.device)
        log_a = torch.where(mask_x > 0, -torch.log(mass_x), negative)
        log_b = torch.where(mask_y > 0, -torch.log(mass_y), negative)
        valid = mask_x.unsqueeze(-1) * mask_y.unsqueeze(1)
        forward = self._sinkhorn_order(
            cost,
            log_a,
            log_b,
            mask_x,
            mask_y,
            epsilon,
            iterations,
            False,
        )
        reverse = self._sinkhorn_order(
            cost,
            log_a,
            log_b,
            mask_x,
            mask_y,
            epsilon,
            iterations,
            True,
        )
        coupling = 0.5 * (forward + reverse)
        coupling = coupling.masked_fill(valid == 0, 0.0)
        if (
            not self.training
            and iterations >= 500
            and not torch.is_grad_enabled()
        ):
            coupling = self._newton_balance(
                coupling, mask_x, mask_y, tolerance=1e-13
            )
        total_mass = coupling.sum(dim=(1, 2), keepdim=True).clamp_min(1e-30)
        return coupling / total_mass

    @staticmethod
    def _newton_balance(
        coupling: torch.Tensor,
        mask_x: torch.Tensor,
        mask_y: torch.Tensor,
        tolerance: float,
    ) -> torch.Tensor:
        """Project compact couplings onto the non-negative transport polytope.

        Low-temperature Sinkhorn can approach a nearly disconnected hard
        assignment before the requested iteration ceiling. Dykstra projections
        retain the closest coupling while enforcing both uniform marginals.
        Averaging the direct and transposed projection paths makes the numerical
        finish explicitly transpose equivariant.
        """

        balanced = torch.zeros_like(coupling)

        def project(compact: torch.Tensor) -> torch.Tensor:
            rows, columns = compact.shape
            target_row = compact.new_full((rows,), 1.0 / rows)
            target_column = compact.new_full((columns,), 1.0 / columns)
            current = compact
            affine_correction = torch.zeros_like(compact)
            cone_correction = torch.zeros_like(compact)
            for _ in range(2_000):
                affine_input = current + affine_correction
                row_residual = target_row - affine_input.sum(dim=1)
                column_residual = target_column - affine_input.sum(dim=0)
                total_residual = 1.0 - affine_input.sum()
                affine = (
                    affine_input
                    + row_residual[:, None] / columns
                    + column_residual[None, :] / rows
                    - total_residual / (rows * columns)
                )
                affine_correction = affine_input - affine
                cone_input = affine + cone_correction
                next_current = torch.clamp(cone_input, min=0.0)
                cone_correction = cone_input - next_current
                current = next_current
                residual = max(
                    float((current.sum(1) - target_row).abs().max().item()),
                    float((current.sum(0) - target_column).abs().max().item()),
                )
                if residual <= tolerance:
                    break
            return current

        for batch_index in range(coupling.shape[0]):
            valid_x = torch.nonzero(
                mask_x[batch_index] > 0, as_tuple=False
            ).flatten()
            valid_y = torch.nonzero(
                mask_y[batch_index] > 0, as_tuple=False
            ).flatten()
            compact = coupling[batch_index][valid_x][:, valid_y].clamp_min(
                0.0
            )
            rows, columns = compact.shape
            if rows != columns:
                raise ValueError(
                    "Phase 3 balanced matching requires equal row cardinality"
                )
            direct = project(compact)
            transposed = project(compact.transpose(0, 1)).transpose(0, 1)
            compact_balanced = 0.5 * (direct + transposed)
            balanced[batch_index][valid_x[:, None], valid_y[None, :]] = (
                compact_balanced
            )
        return balanced

    def get_embeddings(
        self, bx: torch.Tensor, by: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        row_mask_x, row_mask_y, column_mask = _masks(bx, by)
        valid_k = column_mask.sum(dim=1).view(-1, 1, 1).clamp_min(1.0)
        cost_base = (
            ((bx.unsqueeze(2) - by.unsqueeze(1)).abs() * column_mask[:, None, None, :])
            .sum(dim=-1)
            / valid_k
        )
        cost = cost_base
        row_x: torch.Tensor | None = None
        row_y: torch.Tensor | None = None
        alpha: torch.Tensor | float = 0.0
        if self.learned_encoder:
            row_x, row_y = self.encode_rows(bx, by)
        if self.learned_residual_cost:
            assert row_x is not None and row_y is not None
            expanded_x = row_x.unsqueeze(2).expand(
                -1, -1, row_y.shape[1], -1
            )
            expanded_y = row_y.unsqueeze(1).expand(
                -1, row_x.shape[1], -1, -1
            )
            residual_features = torch.cat(
                (
                    (expanded_x - expanded_y).abs(),
                    expanded_x + expanded_y,
                    expanded_x * expanded_y,
                ),
                dim=-1,
            )
            alpha = self.alpha_max * torch.sigmoid(self.raw_alpha)
            cost = cost_base + alpha * F.softplus(
                self.cost_mlp(residual_features).squeeze(-1)
            )

        auxiliary: dict[str, torch.Tensor | float] = {
            "C_base": cost_base,
            "C": cost,
            "mask_x": row_mask_x,
            "mask_y": row_mask_y,
            "column_mask": column_mask,
            "alpha": alpha,
        }
        if self.sinkhorn_enabled:
            iterations = self.train_iters if self.training else self.eval_iters
            coupling = self.sinkhorn(
                cost, row_mask_x, row_mask_y, self.temp, iterations
            )
        elif self.straight_through_hungarian:
            soft = self.sinkhorn(
                cost,
                row_mask_x,
                row_mask_y,
                self.temp,
                self.train_iters if self.training else self.eval_iters,
            )
            hard = hungarian_coupling(cost, row_mask_x, row_mask_y)
            coupling = hard.detach() - soft.detach() + soft
        else:
            coupling = torch.zeros_like(cost)
        auxiliary["P"] = coupling

        if self.sinkhorn_enabled or self.straight_through_hungarian:
            valid_m = row_mask_x.sum(dim=1, keepdim=True).clamp_min(1.0)
            valid_k_flat = column_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            matching_cost = (coupling * cost_base).sum(dim=(1, 2)).unsqueeze(-1)
            scaled_cost = matching_cost * valid_m * valid_k_flat
            if self.straight_through_hungarian:
                # Linear-assignment tie breaking may choose different optimal
                # couplings after a row permutation. The optimal total L1 cost
                # is invariant, so the straight-through ablation's equality
                # feature must be a function of that cost rather than of an
                # arbitrary optimal assignment.
                equality_score = torch.exp(-self.beta * matching_cost)
            else:
                equality_score = (
                    coupling * torch.exp(-self.beta * cost_base)
                ).sum(dim=(1, 2)).unsqueeze(-1)
            representation = torch.cat((scaled_cost, equality_score), dim=-1)
            auxiliary["matching_cost"] = matching_cost
            auxiliary["scaled_matching_cost"] = scaled_cost
            auxiliary["equality_score"] = equality_score
        elif row_x is not None and row_y is not None:
            pooled_x = _masked_mean(row_x, row_mask_x, 1)
            pooled_y = _masked_mean(row_y, row_mask_y, 1)
            representation = torch.cat(
                (pooled_x + pooled_y, (pooled_x - pooled_y).abs()), dim=-1
            )
        else:
            total_x = bx.sum(dim=(1, 2))
            total_y = by.sum(dim=(1, 2))
            representation = torch.stack(
                (total_x + total_y, (total_x - total_y).abs()), dim=-1
            )
        return representation, auxiliary

    def forward_direction(
        self, bx: torch.Tensor, by: torch.Tensor, return_aux: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        representation, auxiliary = self.get_embeddings(bx, by)
        logit = self.classifier(representation).squeeze(-1)
        if return_aux:
            return logit, auxiliary
        return logit

    def forward(
        self, bx: torch.Tensor, by: torch.Tensor, return_aux: bool = False
    ) -> torch.Tensor | tuple[
        torch.Tensor,
        dict[str, torch.Tensor | float],
        dict[str, torch.Tensor | float],
    ]:
        if return_aux:
            forward_logit, forward_aux = self.forward_direction(
                bx, by, return_aux=True
            )
            reverse_logit, reverse_aux = self.forward_direction(
                by, bx, return_aux=True
            )
            return 0.5 * (forward_logit + reverse_logit), forward_aux, reverse_aux
        return 0.5 * (
            self.forward_direction(bx, by) + self.forward_direction(by, bx)
        )


class LearnedSinkhornMatcher(nn.Module):
    """Randomly initialized matcher with a fully learned row-pair cost.

    No fixed L1/Hamming cost or exact zero/nonzero separator enters this model.
    A shared columnwise network learns a symmetric representation for every
    candidate row pair, a second learned head produces the transport cost, and
    the classifier consumes only learned matched features and generic transport
    summaries.
    """

    _sinkhorn_order = staticmethod(
        DifferentiableIncidenceMatchingNetwork._sinkhorn_order
    )
    _newton_balance = staticmethod(
        DifferentiableIncidenceMatchingNetwork._newton_balance
    )
    sinkhorn = DifferentiableIncidenceMatchingNetwork.sinkhorn

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = dict(config)
        self.model_name = str(config["model_name"])
        self.hidden_dim = int(config.get("hidden_dim", 24))
        self.train_iters = int(config.get("train_iters", 30))
        self.eval_iters = int(config.get("eval_iters", 100))
        self.use_column_context = bool(config.get("use_column_context", True))
        self.transport_mode = str(config.get("transport_mode", "sinkhorn"))
        self.cost_scale = float(config.get("cost_scale", 1.0))
        if self.transport_mode not in {"sinkhorn", "uniform"}:
            raise ValueError(f"Unknown learned transport mode: {self.transport_mode}")
        self.register_buffer(
            "temperature", torch.tensor(float(config.get("tau_start", 0.2)))
        )

        self.base_column_pair_mlp = nn.Sequential(
            nn.Linear(2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        if self.use_column_context:
            self.context_column_pair_mlp = nn.Sequential(
                nn.Linear(2, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        self.pair_score_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim + 6, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )

    @property
    def temp(self) -> float:
        return float(self.temperature.item())

    @temp.setter
    def temp(self, value: float) -> None:
        self.temperature.fill_(float(value))

    def _learned_pair_representation(
        self, bx: torch.Tensor, by: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        row_mask_x, row_mask_y, column_mask = _masks(bx, by)
        x = bx[:, :, None, :, None]
        y = by[:, None, :, :, None]
        # For binary incidence entries, sum and product are a symmetric complete
        # description of the unordered pair. Their transformation into a cost is
        # entirely learned and randomly initialized.
        base_features = torch.cat(((x + y) * 0.5, x * y), dim=-1)
        column_embeddings = self.base_column_pair_mlp(base_features)
        if self.use_column_context:
            valid_rows = (
                row_mask_x.sum(1, keepdim=True)
                + row_mask_y.sum(1, keepdim=True)
            ).clamp_min(1.0)
            degree_x = bx.sum(1)
            degree_y = by.sum(1)
            degree_sum = ((degree_x + degree_y) / valid_rows).view(
                bx.shape[0], 1, 1, bx.shape[2], 1
            )
            degree_product = (
                (degree_x * degree_y) / (valid_rows * valid_rows)
            ).view(bx.shape[0], 1, 1, bx.shape[2], 1)
            context_features = torch.cat(
                (
                    degree_sum.expand(
                        -1, bx.shape[1], by.shape[1], -1, -1
                    ),
                    degree_product.expand(
                        -1, bx.shape[1], by.shape[1], -1, -1
                    ),
                ),
                dim=-1,
            )
            column_embeddings = (
                column_embeddings
                + self.context_column_pair_mlp(context_features)
            )
        column_weights = column_mask[:, None, None, :, None]
        valid_columns = column_mask.sum(1).view(-1, 1, 1, 1).clamp_min(1.0)
        weighted_columns = column_embeddings * column_weights
        if weighted_columns.dtype == torch.float64:
            # A reduction over a mathematical set must not inherit the input
            # permutation's floating-point addition order. Canonically sorting
            # each output coordinate preserves the exact sum while making the
            # required float64 column-permutation audit numerically stable.
            weighted_columns = weighted_columns.sort(dim=3).values
        pair_embeddings = weighted_columns.sum(dim=3) / valid_columns
        pair_score = self.pair_score_head(pair_embeddings).squeeze(-1)
        cost = self.cost_scale * F.softplus(
            -pair_score
        )
        return pair_embeddings, cost, row_mask_x, row_mask_y

    def score_row_pairs(
        self, first: torch.Tensor, second: torch.Tensor
    ) -> torch.Tensor:
        """Score equal-vs-perturbed row pairs for train-only pretraining."""

        if first.shape != second.shape or first.ndim != 2:
            raise ValueError("Expected equally shaped [batch, columns] rows")
        x = first.unsqueeze(-1)
        y = second.unsqueeze(-1)
        features = torch.cat(((x + y) * 0.5, x * y), dim=-1)
        representation = self.base_column_pair_mlp(features).mean(dim=1)
        return self.pair_score_head(representation).squeeze(-1)

    def get_embeddings(
        self, bx: torch.Tensor, by: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        pair_embeddings, cost, row_mask_x, row_mask_y = (
            self._learned_pair_representation(bx, by)
        )
        if self.transport_mode == "sinkhorn":
            iterations = self.train_iters if self.training else self.eval_iters
            coupling = self.sinkhorn(
                cost, row_mask_x, row_mask_y, self.temp, iterations
            )
        else:
            valid = row_mask_x.unsqueeze(2) * row_mask_y.unsqueeze(1)
            coupling = valid / valid.sum((1, 2), keepdim=True).clamp_min(1.0)

        matched_embedding = (
            coupling.unsqueeze(-1) * pair_embeddings
        ).sum(dim=(1, 2))
        matching_cost_raw = (coupling * cost).sum(dim=(1, 2)).unsqueeze(-1)
        matching_cost = matching_cost_raw / self.cost_scale
        entropy = -(
            coupling.clamp_min(1e-30).log() * coupling
        ).sum(dim=(1, 2)).unsqueeze(-1)
        valid_pair_count = (
            row_mask_x.sum(1) * row_mask_y.sum(1)
        ).clamp_min(2.0)
        entropy = entropy / valid_pair_count.log().unsqueeze(-1)
        normalized_cost = cost / self.cost_scale
        centered_cost = normalized_cost - matching_cost.view(-1, 1, 1)
        cost_spread = (
            coupling * centered_cost.square()
        ).sum(dim=(1, 2)).clamp_min(1e-12).sqrt().unsqueeze(-1)
        row_mass = coupling.sum(2).clamp_min(1e-30)
        row_cost = (coupling * normalized_cost).sum(2) / row_mass
        masked_row_cost = row_cost.masked_fill(row_mask_x == 0, -1e30)
        row_cost_max = masked_row_cost.amax(1, keepdim=True)
        masked_row_cost_min = row_cost.masked_fill(row_mask_x == 0, 1e30)
        row_cost_min = masked_row_cost_min.amin(1, keepdim=True)
        row_cost_mean = (
            row_cost * row_mask_x
        ).sum(1, keepdim=True) / row_mask_x.sum(1, keepdim=True).clamp_min(1.0)
        representation = torch.cat(
            (
                matched_embedding,
                matching_cost,
                entropy,
                cost_spread,
                row_cost_max,
                row_cost_min,
                row_cost_mean,
            ),
            dim=-1,
        )
        return representation, {
            "C": cost,
            "P": coupling,
            "mask_x": row_mask_x,
            "mask_y": row_mask_y,
            "matching_cost": matching_cost,
            "raw_matching_cost": matching_cost_raw,
            "equality_score": torch.exp(-matching_cost),
            "row_cost_max": row_cost_max,
            "row_cost_min": row_cost_min,
            "row_cost_mean": row_cost_mean,
            "pair_embeddings": pair_embeddings,
        }

    def forward_direction(
        self, bx: torch.Tensor, by: torch.Tensor, return_aux: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        representation, auxiliary = self.get_embeddings(bx, by)
        logit = self.classifier(representation).squeeze(-1)
        if return_aux:
            return logit, auxiliary
        return logit

    def forward(
        self, bx: torch.Tensor, by: torch.Tensor, return_aux: bool = False
    ) -> torch.Tensor | tuple[
        torch.Tensor,
        dict[str, torch.Tensor | float],
        dict[str, torch.Tensor | float],
    ]:
        if return_aux:
            forward_logit, forward_aux = self.forward_direction(
                bx, by, return_aux=True
            )
            reverse_logit, reverse_aux = self.forward_direction(
                by, bx, return_aux=True
            )
            return 0.5 * (forward_logit + reverse_logit), forward_aux, reverse_aux
        return 0.5 * (
            self.forward_direction(bx, by) + self.forward_direction(by, bx)
        )


class DeterministicDifferentiableMatching(
    DifferentiableIncidenceMatchingNetwork
):
    """Explicit name for the fixed-L1 algorithmic Sinkhorn baseline."""


class ExactHungarianOracle(nn.Module):
    """Exact fixed-L1 assignment oracle, never optimized as a learned model."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = dict(config)
        self.model_name = str(config["model_name"])
        self.logit_magnitude = float(config.get("oracle_logit_magnitude", 20.0))

    def forward(
        self, bx: torch.Tensor, by: torch.Tensor, return_aux: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
        row_mask_x, row_mask_y, _ = _masks(bx, by)
        cost = (bx.unsqueeze(2) - by.unsqueeze(1)).abs().sum(dim=-1)
        coupling = hungarian_coupling(cost, row_mask_x, row_mask_y)
        valid_rows = row_mask_x.sum(1).clamp_min(1.0)
        total_cost = (coupling * cost).sum((1, 2)) * valid_rows
        logit = torch.where(
            total_cost <= 0,
            total_cost.new_full(total_cost.shape, self.logit_magnitude),
            total_cost.new_full(total_cost.shape, -self.logit_magnitude),
        )
        if return_aux:
            auxiliary = {
                "C": cost,
                "P": coupling,
                "mask_x": row_mask_x,
                "mask_y": row_mask_y,
                "matching_cost": total_cost.unsqueeze(-1),
                "equality_score": (total_cost <= 0).to(cost.dtype).unsqueeze(-1),
            }
            return logit, auxiliary, {
                **auxiliary,
                "C": cost.transpose(1, 2),
                "P": coupling.transpose(1, 2),
            }
        return logit


class SummaryMLP(nn.Module):
    """Invariant aggregate-feature baseline."""

    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _summary(matrix: torch.Tensor) -> torch.Tensor:
        row_mask = (matrix.abs().sum(dim=-1) > 0).to(matrix.dtype)
        column_mask = (matrix.abs().sum(dim=1) > 0).to(matrix.dtype)
        row_sum = matrix.sum(dim=-1)
        column_sum = matrix.sum(dim=1)
        row_mean = (row_sum * row_mask).sum(dim=1) / row_mask.sum(dim=1).clamp_min(1)
        column_mean = (column_sum * column_mask).sum(dim=1) / column_mask.sum(dim=1).clamp_min(1)
        return torch.stack(
            (
                row_mask.sum(dim=1),
                column_mask.sum(dim=1),
                matrix.sum(dim=(1, 2)),
                row_mean,
                column_mean,
            ),
            dim=-1,
        )

    def forward(
        self, bx: torch.Tensor, by: torch.Tensor, return_aux: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
        sx = self._summary(bx)
        sy = self._summary(by)
        features = torch.cat(
            (sx[:, :3] + sy[:, :3], (sx - sy).abs()), dim=-1
        )
        logit = self.network(features).squeeze(-1)
        if return_aux:
            return logit, {}, {}
        return logit


class NestedDeepSets(nn.Module):
    """Nested set-pooling baseline with explicit padding masks."""

    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.row_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def _encode(self, matrix: torch.Tensor) -> torch.Tensor:
        mask = (matrix.abs().sum(dim=-1) > 0).to(matrix.dtype)
        row_size = matrix.sum(dim=-1, keepdim=True)
        rows = self.row_mlp(row_size)
        return _masked_mean(rows, mask, 1)

    def get_embeddings(self, bx: torch.Tensor, by: torch.Tensor) -> torch.Tensor:
        hx = self._encode(bx)
        hy = self._encode(by)
        return torch.cat((hx + hy, (hx - hy).abs()), dim=-1)

    def forward(
        self, bx: torch.Tensor, by: torch.Tensor, return_aux: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
        logit = self.classifier(self.get_embeddings(bx, by)).squeeze(-1)
        if return_aux:
            return logit, {}, {}
        return logit


class PairConditionedBipartiteEncoder(nn.Module):
    """Column-permutation equivariant pair-conditioned baseline."""

    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.column_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def get_embeddings(self, bx: torch.Tensor, by: torch.Tensor) -> torch.Tensor:
        row_mask_x, row_mask_y, column_mask = _masks(bx, by)
        degrees_x = bx.sum(dim=1)
        degrees_y = by.sum(dim=1)
        features = torch.stack(
            (degrees_x + degrees_y, (degrees_x - degrees_y).abs()), dim=-1
        )
        columns = self.column_mlp(features) * column_mask.unsqueeze(-1)
        pooled = _masked_mean(columns, column_mask, 1)
        row_count = torch.stack(
            (row_mask_x.sum(dim=1), row_mask_y.sum(dim=1)), dim=-1
        )
        scale = row_count.sum(dim=1, keepdim=True).clamp_min(1.0)
        root_signal = (row_count[:, :1] - row_count[:, 1:]).abs() / scale
        return torch.cat((pooled, pooled * (1.0 + root_signal)), dim=-1)

    def forward(
        self, bx: torch.Tensor, by: torch.Tensor, return_aux: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
        logit = self.classifier(self.get_embeddings(bx, by)).squeeze(-1)
        if return_aux:
            return logit, {}, {}
        return logit


def initialize_model(config: dict[str, Any]) -> nn.Module:
    architecture = config["architecture"]
    hidden_dim = int(config.get("hidden_dim", 16))
    if architecture == "summary_mlp":
        return SummaryMLP(hidden_dim)
    if architecture == "nested_deepsets":
        return NestedDeepSets(hidden_dim)
    if architecture == "bipartite_encoder":
        return PairConditionedBipartiteEncoder(hidden_dim)
    if architecture == "learned_matching":
        return LearnedSinkhornMatcher(config)
    if architecture == "deterministic_matching":
        return DeterministicDifferentiableMatching(config)
    if architecture == "exact_hungarian":
        return ExactHungarianOracle(config)
    if architecture == "matching":
        return DifferentiableIncidenceMatchingNetwork(config)
    raise KeyError(f"Unknown model architecture: {architecture}")
