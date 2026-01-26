# Gaussian Naive Bayes — *By-Hand* Worked Example

This file walks through a complete **Naive Bayes classification example** where features are assumed to follow **normal (Gaussian) distributions**. All calculations are shown explicitly, with **all conditional means and variances chosen to be different** for clarity.

---

## Problem setup

We consider a **binary classification** problem with classes:
- $C_0$
- $C_1$

We observe two continuous features:
$$
x = (x_1, x_2)
$$

### Model assumptions

1. **Class priors**: $P(C_k)$
2. **Conditional independence** (Naive Bayes assumption):
$$
p(x_1, x_2 \mid C_k) = p(x_1 \mid C_k)\,p(x_2 \mid C_k)
$$
3. **Gaussian likelihoods** for each feature:
$$
p(x_j \mid C_k) = \frac{1}{\sqrt{2\pi\sigma_{k,j}^2}}
\exp\left(-\frac{(x_j - \mu_{k,j})^2}{2\sigma_{k,j}^2}\right)
$$

---

## Given quantities

### Class priors

$$
P(C_0) = 0.6, \quad P(C_1) = 0.4
$$

### Feature distributions

| Class | Feature | Mean $\mu$ | Variance $\sigma^2$ |
|------|--------|---------------|------------------------|
| $C_0$ | $x_1$ | 0.0 | 1.0 |
| $C_0$ | $x_2$ | 0.0 | 4.0 |
| $C_1$ | $x_1$ | 2.0 | 1.0 |
| $C_1$ | $x_2$ | 3.0 | 1.0 |

### New observation

$$
x = (1.0,\;2.0)
$$

---

## Step 1 — Compute likelihoods

### Likelihood under $C_0$

#### Feature $x_1 = 1.0$

$$
p(x_1 \mid C_0)
= \frac{1}{\sqrt{2\pi(1)}}
\exp\left(-\frac{(2 - 0)^2}{2}\right)
\approx 0.24197
$$

#### Feature $x_2 = 2.0$

$$
p(x_2 \mid C_0)
= \frac{1}{\sqrt{2\pi(4)}}
\exp\left(-\frac{(2 - 0)^2}{2\cdot 4}\right)
\approx 0.17603
$$

#### Joint likelihood (Naive Bayes)

$$
p(x \mid C_0) = 0.24197 \times 0.17603 \approx 0.04259
$$

---

### Likelihood under $C_1$

#### Feature $x_1 = 1.0$

$$
p(x_1 \mid C_1)
= \frac{1}{\sqrt{2\pi(1)}}
\exp\left(-\frac{(1 - 2)^2}{2}\right)
\approx 0.24197
$$

#### Feature $x_2 = 2.0$

$$
p(x_2 \mid C_1)
= \frac{1}{\sqrt{2\pi(1)}}
\exp\left(-\frac{(2 - 3)^2}{2}\right)
\approx 0.05399
$$

#### Joint likelihood

$$
p(x \mid C_1) = 0.24197 \times 0.05399 \approx 0.01306
$$

---

## Step 2 — Multiply by priors (unnormalized posteriors)

$$
P(C_0 \mid x) \propto 0.04259 \times 0.6 = 0.02556
$$

$$
P(C_1 \mid x) \propto 0.01306 \times 0.4 = 0.00523
$$

---

## Step 3 — Normalize

Sum of unnormalized posteriors:
$$
0.02556 + 0.00523 = 0.03079
$$

Final posterior probabilities:

$$
P(C_0 \mid x) = \frac{0.02556}{0.03079} \approx 0.8302
$$

$$
P(C_1 \mid x) = \frac{0.00523}{0.03079} \approx 0.1698
$$

---

## Final classification

$$
\boxed{\text{Predict class } C_0}
$$

---

## Notes

- All **means and variances differ** across classes and features to make each Gaussian term visually and numerically distinct.
- Independence is assumed **conditionally on the class**.
- In practice, implementations use **log-probabilities** to avoid numerical underflow.
- This corresponds directly to `sklearn.naive_bayes.GaussianNB`.

---

*End of file.*

