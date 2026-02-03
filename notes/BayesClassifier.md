# Guided Notes: A Concrete Case Where We *Know* the Bayes Classifier

## Learning goals
By the end of this activity, you should be able to:
- State the Bayes classification rule in terms of posterior probability.
- Use Bayes’ rule to rewrite the decision in terms of likelihood × prior.
- Derive the Bayes classifier for a simple **two-class Gaussian** model.
- Identify the **decision threshold** and interpret it.
- Explain how knowing the Bayes classifier helps us benchmark ML models.

---

## Setup: a world where the “truth” is known

We consider a binary classification problem with:
- **Classes:** $Y \in \{0, 1\}$
- **Feature:** a single real-valued feature $X \in \mathbb{R}$

Assume the data are generated as:

### Priors (class frequencies)
$$
P(Y=0) = \tfrac{1}{2}, \quad P(Y=1) = \tfrac{1}{2}
$$

### Class-conditional distributions (likelihoods)
$$
X \mid Y=0 \sim \mathcal{N}(\mu_0, \sigma^2), \quad
X \mid Y=1 \sim \mathcal{N}(\mu_1, \sigma^2)
$$
with $\mu_0 \neq \mu_1$ and the **same variance** $\sigma^2$.

> **Interpretation:** Each class generates points around its own mean. The “noise level” is $\sigma$.

---

## 1) The Bayes classifier definition

The Bayes classifier predicts the class with largest posterior probability:

$$
\hat{y}(x) = \arg\max_{y \in \{0,1\}} P(Y=y \mid X=x)
$$

### Guided check
- What does $P(Y=y \mid X=x)$ mean in words?
  - **Your answer:** ________________________________________________

---

## 2) Rewrite using Bayes’ rule

Bayes’ rule:
$$
P(Y=y \mid X=x) = \frac{p(X=x \mid Y=y)\,P(Y=y)}{p(X=x)}
$$

Since $p(X=x)$ is the same for both classes, the argmax does not depend on it:

$$
\hat{y}(x) = \arg\max_{y \in \{0,1\}} p(x \mid y)\,P(y)
$$

### Guided check
- Why can we ignore $p(x)$ in the argmax?
  - **Your answer:** ________________________________________________

---

## 3) Plug in our priors

Here the priors are equal:
$$
P(0) = P(1) = \tfrac{1}{2}
$$

So the decision rule simplifies to:

$$
\hat{y}(x) = \arg\max_{y \in \{0,1\}} p(x \mid y)
$$

> **Meaning:** choose the class under which $x$ is more likely.

---

## 4) Write the Gaussian likelihoods

For a normal distribution $\mathcal{N}(\mu, \sigma^2)$,
$$
p(x) = \frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

So:
$$
p(x \mid 0) =
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left(-\frac{(x-\mu_0)^2}{2\sigma^2}\right)
$$
$$
p(x \mid 1) =
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left(-\frac{(x-\mu_1)^2}{2\sigma^2}\right)
$$

### Guided check
- Which pieces are the same for both classes?
  - **Your answer:** ________________________________________________

---

## 5) Compare likelihoods by comparing log-likelihoods

We decide class 1 if:
$$
p(x \mid 1) > p(x \mid 0)
$$

Take logs (log is monotone increasing, so it preserves “greater than”):
$$
\log p(x \mid 1) > \log p(x \mid 0)
$$

Because the constant $-\tfrac{1}{2}\log(2\pi\sigma^2)$ appears on both sides, it cancels.

What remains is:
$$
-\frac{(x-\mu_1)^2}{2\sigma^2} > -\frac{(x-\mu_0)^2}{2\sigma^2}
$$

Multiply both sides by $-2\sigma^2$ (note the inequality flips when multiplying by a negative number):
$$
(x-\mu_1)^2 < (x-\mu_0)^2
$$

> **Interpretation:** pick the class whose mean is *closer* to $x$.

---

## 6) Solve for the decision boundary (the threshold)

We want the set of $x$ where both classes are equally likely:
$$
(x-\mu_1)^2 = (x-\mu_0)^2
$$

Expand both sides:
$$
x^2 - 2\mu_1 x + \mu_1^2 = x^2 - 2\mu_0 x + \mu_0^2
$$

Cancel $x^2$ on both sides:
$$
- 2\mu_1 x + \mu_1^2 = - 2\mu_0 x + \mu_0^2
$$

Bring the $x$-terms to one side:
$$
2(\mu_0 - \mu_1)x = \mu_0^2 - \mu_1^2
$$

Factor the right-hand side:
$$
\mu_0^2 - \mu_1^2 = (\mu_0 - \mu_1)(\mu_0 + \mu_1)
$$

Cancel $(\mu_0 - \mu_1)$ (valid because $\mu_0 \neq \mu_1$):
$$
2x = \mu_0 + \mu_1
$$

So the boundary is:
$$
x^{*} = \frac{\mu_0 + \mu_1}{2}
$$

---

## 7) Final Bayes classifier (known exactly)

Assuming $\mu_1 > \mu_0$, the Bayes classifier is:

$$
\boxed{
\hat{y}(x) =
\begin{cases}
0 & x < \tfrac{\mu_0 + \mu_1}{2} \\
1 & x \ge \tfrac{\mu_0 + \mu_1}{2}
\end{cases}
}
$$

### Guided check
- In words, how do we classify a point $x$?
  - **Your answer:** ________________________________________________

---

## 8) A numeric example you can compute quickly

Let:
- $\mu_0 = 0$
- $\mu_1 = 2$
- $\sigma = 1$

Then:
$$
x^{*} = \frac{0 + 2}{2} = 1
$$

So:
- if $x < 1$, predict class 0
- if $x \ge 1$, predict class 1

### Practice
Classify each point:
- $x = -0.2$: _______
- $x = 0.9$: _______
- $x = 1.0$: _______
- $x = 3.4$: _______

---

## 9) How do we *use* the fact that Bayes is known?

### Use case A: Benchmarking ML models (simulation study)
If we generate training/test data from this known model, then:
- the Bayes classifier is the **best possible** classifier
- its error is the **irreducible error** for this problem

So if a learned model has test error much larger than Bayes error, the gap is due to:
- limited data (estimation error),
- model mismatch,
- optimization issues,
- etc.

### Use case B: Understanding decision boundaries
This derivation shows:
- equal-variance Gaussians $\Rightarrow$ a **linear** decision rule (a threshold in 1D)
- the boundary sits exactly midway between means (when priors are equal)

### Use case C: Connecting to LDA
This is the 1D version of **Linear Discriminant Analysis (LDA)**.
In higher dimensions, the boundary is a hyperplane when covariances are equal.

---

## 10) Quick “what if” questions (concept checks)

1. If $P(Y=1)$ were larger than $P(Y=0)$, would the threshold stay at $\tfrac{\mu_0+\mu_1}{2}$?
   - **Prediction:** ________________________________________________

2. If the variances were different ($\sigma_0^2 \neq \sigma_1^2$), would the boundary still be a single midpoint threshold?
   - **Prediction:** ________________________________________________

3. What happens to the overlap (and difficulty) when $\sigma$ increases?
   - **Prediction:** ________________________________________________

---

## Summary
- Bayes classifier: choose the class with highest posterior $P(y\mid x)$.
- With equal priors, this is equivalent to choosing the larger likelihood $p(x\mid y)$.
- For two Gaussians with equal variance, the Bayes decision is a simple threshold:
  $$
  x^{*} = \frac{\mu_0+\mu_1}{2}
  $$
- Knowing Bayes lets us quantify the *best achievable performance* and benchmark learning algorithms.
