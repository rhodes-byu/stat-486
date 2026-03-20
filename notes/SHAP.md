# SHAP (SHapley Additive exPlanations) - Structured Notes

## 1. Overview
SHAP is a method for explaining individual predictions of machine learning models by assigning each feature a contribution value.

**Core idea:**
> A prediction is decomposed into a baseline plus contributions from each feature.

$$
f(x) = \mathbb{E}[f(X)] + \sum_{i=1}^p \phi_i(x)
$$

- $\phi_i(x)$: contribution of feature $i$ for input $x$
- $\mathbb{E}[f(X)]$: baseline (average model output under the chosen output scale)

---

## 2. Local vs Global Importance

### Local (primary purpose)
- SHAP explains **individual predictions**
- Each $\phi_i(x)$ is specific to a single observation

### Global (derived)
- Aggregate local values across the dataset

Common metric:
$$
\mathbb{E}[|\phi_i(X)|]
$$

---

## 3. Formal Definition

Let:
- $N = \{1, \dots, p\}$: feature set
- $S \subseteq N$: subset of features

For a chosen value function $v_x(S)$ that measures the model output when only features in $S$ are treated as known, the Shapley value is:
$$
\phi_i(x) = \sum_{S \subseteq N \setminus \{i\}}
\frac{|S|!(p-|S|-1)!}{p!}
\big[v_x(S \cup \{i\}) - v_x(S)\big]
$$

One common choice is the conditional value function:
$$
v_x(S) = \mathbb{E}[f(X) \mid X_S = x_S]
$$

---

## 4. Interpretation

$$
\phi_i(x) = \text{average marginal contribution of feature } i
$$

- More precisely, compare the value function before and after adding feature $i$ to a subset $S$
- Average that marginal contribution across all possible subsets (contexts)
- The meaning of "without feature $i$" depends on the chosen value function

---

## 5. Role of Feature Order

Features have **no natural order**, but SHAP introduces order as a mathematical tool.

### Key issue
Feature contribution depends on context due to interactions.

### Solution
- Consider all possible orderings of features
- Measure the contribution when a feature enters
- Average over all orderings

This is equivalent to averaging over all subsets, with a weight determined by how many orderings produce each subset as the set before feature $i$.

---

## 6. Permutation View

$$
\phi_i(x) = \mathbb{E}_{\pi} \left[
 v_x(\mathrm{Pre}_\pi(i) \cup \{i\})
 - v_x(\mathrm{Pre}_\pi(i))
\right]
$$

- $\pi$: random permutation of features
- $\mathrm{Pre}_\pi(i)$: features before $i$ in permutation $\pi$

---

## 7. Where the Weighting Term Comes From

Weight assigned to subset $S$:
$$
\frac{|S|!(p-|S|-1)!}{p!}
$$

### Interpretation
$$
\Pr(\mathrm{Pre}_\pi(i) = S)
$$

So the coefficient is the probability that $S$ is exactly the set of features appearing before $i$ in a random permutation.

### Derivation
- Order the $|S|$ features in $S$ before $i$: $|S|!$
- Place feature $i$ next
- Order the remaining $p-|S|-1$ features after $i$: $(p-|S|-1)!$
- Total permutations: $p!$

---

## 8. Example (3 Features)

For feature $A$, let the other features be $B$ and $C$.

All permutations and the set before $A$:
- `ABC` -> $S = \varnothing$
- `ACB` -> $S = \varnothing$
- `BAC` -> $S = \{B\}$
- `CAB` -> $S = \{C\}$
- `BCA` -> $S = \{B, C\}$
- `CBA` -> $S = \{B, C\}$

Resulting weights:
- $\varnothing$: $2/6 = 1/3$
- $\{B\}$ and $\{C\}$: $1/6$ each
- $\{B, C\}$: $2/6 = 1/3$

This matches the SHAP weighting formula exactly.

---

## 9. Why Weighting Matters

It ensures:
- Equal treatment of all permutations
- Fair representation of different interaction contexts
- Equal total weight for each subset size after summing over all subsets of that size

---

## 10. Linearity Property

For models $f$ and $g$ defined under the same value-function semantics:

$$
\phi_i^{f+g}(x) = \phi_i^f(x) + \phi_i^g(x)
$$

More generally:
$$
\phi_i^{af + bg}(x) = a\phi_i^f(x) + b\phi_i^g(x)
$$

### Why it holds
- SHAP is defined via sums and expectations
- Both are linear operations

---

## 11. Implications of Linearity

### (a) Linear models
For a linear model $f(x) = \beta_0 + \sum_i \beta_i x_i$, interventional SHAP gives:
$$
\phi_i(x) = \beta_i (x_i - \mathbb{E}[X_i])
$$

If features are dependent and you use conditional SHAP instead, attribution can be redistributed across correlated features.

### (b) Additive models
If
$$
f(x) = \sum_j f_j(x)
$$
then
$$
\phi_i(x) = \sum_j \phi_i^{f_j}(x)
$$

### (c) Tree ensembles
- Model = sum of trees
- SHAP = sum of per-tree contributions

---

## 12. Value Function Choices

### Interventional SHAP
$$
v_x(S) = \mathbb{E}_{X_{\bar S}}[f(x_S, X_{\bar S})]
$$
- Breaks dependence between observed and missing features
- Often easier to compute

### Conditional SHAP
$$
v_x(S) = \mathbb{E}[f(X) \mid X_S = x_S]
$$
- Respects observed feature correlations
- Harder to estimate and can redistribute credit across correlated features

---

## 13. Key Properties (Axioms)

For a fixed value function, Shapley values are the **unique solution** satisfying:

1. **Efficiency**
$$
\sum_i \phi_i(x) = v_x(N) - v_x(\varnothing)
$$

With the usual SHAP baseline, this becomes:
$$
\sum_i \phi_i(x) = f(x) - \mathbb{E}[f(X)]
$$

2. **Symmetry**
If two features contribute identically in all coalitions, they receive equal contributions.

3. **Dummy**
If adding a feature never changes the value function, its contribution is zero.

4. **Linearity**
Attributions add across summed models.

---

## 14. Computational Considerations

Exact computation requires $2^p$ subsets per explained point.

Approximations:
- Kernel SHAP (sampling + regression)
- Tree SHAP (efficient exact method for many tree models)
- Deep SHAP (approximate method for neural networks)

---

## 15. Final Summary

- SHAP is a **local explanation method** based on Shapley values
- It assigns contributions by averaging over all feature subsets
- The weighting term comes from counting permutations
- It ensures fair attribution under interactions
- Global importance is derived by aggregating local values

**One-line takeaway:**
> SHAP values are the expected marginal contributions of features across all possible contexts, computed in a way that guarantees fairness and consistency for a chosen value function.
