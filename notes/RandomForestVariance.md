# Random Forests: Variance, Correlation, and the Correlation Floor

## 1. Objective

Understand why random feature selection in Random Forests reduces
variance and improves ensemble performance.

The central quantity of interest is:

$$
\operatorname{Var}(\bar{T})
$$

where:

$$
\bar{T}(x) = \frac{1}{M} \sum_{i=1}^{M} T_i(x)
$$

is the ensemble prediction at a fixed input $x$.

------------------------------------------------------------------------

## 2. Exact Variance Decomposition

Start with:

$$
\operatorname{Var}(\bar{T})
= \operatorname{Var}\!\left(\frac{1}{M}\sum_{i=1}^{M} T_i\right)
= \frac{1}{M^2}\operatorname{Var}\!\left(\sum_{i=1}^{M} T_i\right)
$$

Expand the variance of a sum:

$$
\operatorname{Var}\!\left(\sum_{i=1}^{M} T_i\right)
= \sum_{i=1}^{M}\operatorname{Var}(T_i)
+ \sum_{i\ne j}\operatorname{Cov}(T_i, T_j)
$$

Thus:

$$
\operatorname{Var}(\bar{T})
= \frac{1}{M^2}\left(\sum_{i=1}^{M}\operatorname{Var}(T_i)
+ \sum_{i\ne j}\operatorname{Cov}(T_i, T_j)\right)
$$

------------------------------------------------------------------------

## 3. With Exchangeability (Simplified Model)

If trees are exchangeable:

$$
\operatorname{Var}(T_i) = \sigma^2
$$

$$
\operatorname{Cov}(T_i,T_j) = \rho\sigma^2
$$

Then:

$$
\operatorname{Var}(\bar{T}) = \rho\sigma^2 + \frac{1-\rho}{M}\sigma^2
$$

As $M \to \infty$:

$$
\operatorname{Var}(\bar{T}) \to \rho\sigma^2
$$

The correlation floor is now $\rho\sigma^2$.

------------------------------------------------------------------------

## 4. What Drives the Correlation Floor?

The floor equals the **average pairwise covariance** between trees.

Correlation arises from:

1.  Shared underlying signal (f(x))
2.  Dominant predictors
3.  Redundant (correlated) features
4.  High signal-to-noise ratio
5.  Structural similarity of the tree-growing algorithm

Even with randomness, trees approximate the same function, so covariance
cannot be zero.

------------------------------------------------------------------------

## 5. Why Random Feature Selection Helps

At each split, sample $m$ out of $p$ features.

Probability a dominant feature is available:

$$
P(\text{strong feature available}) = \frac{m}{p}
$$

If $m$ is small:

-   Strong predictors are often excluded
-   Trees follow alternative split paths
-   Tree structures diversify
-   Correlation $\rho$ between trees decreases

Thus:

$$
\operatorname{Var}(\bar{T}) \approx \rho\sigma^2
$$

Reducing $m$ reduces the correlation floor.

------------------------------------------------------------------------

## 6. Bias--Variance Tradeoff

However, smaller $m$:

-   Weakens individual trees
-   May increase per-tree variance
-   May increase bias

Approximate ensemble MSE:

$$
MSE(m) = Bias(m)^2 + \rho(m)\sigma^2(m) + \frac{1-\rho(m)}{M}\sigma^2(m)
$$

For large $M$:

$$
MSE(m) \approx Bias(m)^2 + \rho(m)\sigma^2(m)
$$

Choose $m$ to balance strength and correlation.

------------------------------------------------------------------------

## 7. Why $m \approx \sqrt{p}$?

Heuristic compromise:

-   Ensures $m/p$ decreases with dimension
-   Reduces probability dominant predictors dominate all trees
-   Still allows enough features per split to maintain strength

It balances reduction in covariance against increase in bias.

------------------------------------------------------------------------

## 8. Final Logical Flow

1.  Under exchangeability, ensemble variance depends on $\sigma^2$ and
	$\rho$.
2.  As forest size grows, variance converges to $\rho\sigma^2$.
3.  Correlation arises from shared signal and dominant predictors.
4.  Random feature selection reduces shared structure across trees.
5.  Reduced covariance lowers the correlation floor.
6.  Optimal $m$ balances tree strength and decorrelation.

------------------------------------------------------------------------

**Core Insight:**

Random forests work because they reduce the average covariance between
trees faster than they weaken individual trees.
