# Mathematical formulation

This guide describes the active implementation in [`src/differentiable_matching.py`](../src/differentiable_matching.py). Shapes below refer to compact valid matrices, with padding removed.

## Local target

Let $B_x,B_y\in\{0,1\}^{m\times k}$ contain the incidence stars of two roots, with a shared indexing of the $k$ neighboring vertices. Rows represent incident hyperedges. The target is

$$y=\mathbf{1}\!\left[\operatorname{multiset}(B_x)=\operatorname{multiset}(B_y)\right].$$

Repeated rows retain their multiplicity. This is the binary target within the generator's EE-equivalent candidate family, not an arbitrary global isomorphism definition.

## Learned symmetric cost

For candidate rows $i,j$ and common column $\ell$, the primary forms

$$s_{ij\ell}=\left(\frac{B_{x,i\ell}+B_{y,j\ell}}{2},\;B_{x,i\ell}B_{y,j\ell}\right).$$

A shared neural encoder transforms these features. An optional context encoder adds symmetric normalized column-degree features. With $d_{x\ell}=\sum_i B_{x,i\ell}$ and $d_{y\ell}=\sum_j B_{y,j\ell}$, the context features are

$$q_\ell=\left(\frac{d_{x\ell}+d_{y\ell}}{2m},\;\frac{d_{x\ell}d_{y\ell}}{(2m)^2}\right).$$

The learned row-pair representation and nonnegative cost are

$$h_{ij}=\frac1k\sum_{\ell=1}^k\left[\phi_\theta(s_{ij\ell})+\psi_\theta(q_\ell)\right],\qquad C_{ij}=\alpha\,\operatorname{softplus}(-g_\theta(h_{ij})).$$

Here $\alpha$ is the configured `cost_scale`. Removing column context drops the $\psi_\theta$ term. These learned quantities are not fixed L1 distances, a metric guarantee, or exact equivalence labels.

## Balanced normalized transport

The standard entropy-regularized transport objective motivating Sinkhorn is

$$\min_{P\ge0}\;\langle P,C\rangle+\tau\sum_{i,j}P_{ij}(\log P_{ij}-1),\qquad P\mathbf1=a,\quad P^\top\mathbf1=b,$$

with uniform marginals $a_i=b_j=1/m$. Thus

$$\sum_{i,j}P_{ij}=1.$$

An ideal hard match has $m$ nonzero entries of mass $1/m$, rather than $m$ entries of mass one. For example, with four equally weighted rows, a permutation coupling contains four entries of `0.25`.

The implementation uses log-domain finite Sinkhorn updates, averages the two update orders and normalizes total mass. In no-gradient evaluation with at least 500 iterations, it additionally uses a symmetric Dykstra projection to enforce marginal constraints. This numerical finish is part of the saved audit procedure; finite output should not be described as a proof of exact entropy-objective optimality.

The classifier receives the matched representation $\sum_{ij}P_{ij}h_{ij}$ and six transport/cost summaries. The public logit averages the two directional logits:

$$z(B_x,B_y)=\tfrac12\left[z_{\rightarrow}(B_x,B_y)+z_{\rightarrow}(B_y,B_x)\right],\qquad p(y=1)=\sigma(z).$$

## Required symmetries

For row permutation matrices $R_x,R_y$ and a **shared** column permutation matrix $Q$,

$$p(B_x,B_y)=p(R_xB_xQ,R_yB_yQ),\qquad p(B_x,B_y)=p(B_y,B_x).$$

Costs and couplings transform equivariantly: root swapping should transpose them. Independent column permutations on the two sides change the alignment of common neighbors and are not the same invariance requirement.

## Reading the ablation

`LearnedNoSinkhorn` uses uniform mass over all valid row pairs. It retains a normalized coupling but removes learned constrained assignment. Its strong IID classification and weak OOD-size classification, alongside strong OOD-size ROC-AUC, show why both thresholded metrics and ranking metrics are needed. These results support the contribution within the tested generator; they do not prove transport is necessary for every possible architecture.
