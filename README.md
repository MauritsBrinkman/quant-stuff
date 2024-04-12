# Identifying small mean-reverting portfolios

## Introduction
Suppose you have a huge data set with assets and would like to construct a portfolio that is as mean reverting as 
possible while being sparse. How to filter out such a portfolio out of the data set? That is exactly what this code is 
meant to do.

Being sparse means that we don't want to have many assets. There are a few reasons for this:
* High transaction costs
* Overfitting
* Instability
* Little variance

Generally the more assets our portfolio has, the more mean reverting it is but the less volatile it is. We would like to 
find the golden middle of mean reverting and volatile enough while still being stable.

The set-up of this code was inspired by [this article of Alex D'Aspremont](https://arxiv.org/pdf/0708.3048.pdf). Nevertheless, the explanation below should be
self-containing enough to understand the necessary theory to follow along with the provided code. Now,
let's have some free lunch!

## The portfolio
We assume that the assets in our portfolio satisfy the vector autoregressive process $$S_t = S_{t-1}A + Z_t,$$ where $S_t$
is the vector containing all asset values at time $t$, $A$ is an $n \times n$-matrix and $Z_t$ a vector of Gaussian noise
with zero mean and covariance $\Sigma$. Without loss of generality, we can assume that the assets $S_t$ have zero mean.

Our portfolio $P_t$ at time $t$ will simply be defined by $P_t = S_t x$, where $x$ is a vector of (normalized) portfolio 
weights. Our goal is to find a mean-reverting portfolio, which can be formulated as an Ornstein-Uhlenbeck process: $$\text{d}P_t
= \lambda (\tilde{P} - P_t) \text{d}t + \sigma \text{d}Z_t.$$ $\text{d}P_t$ represents the change in portfolio value, $\lambda$
is the strength of mean reversion, $\sigma$ represents the variance of the noise and $Z_t$ the Gaussian standard noise.

## Variance ratio and predictability

For simplicity, let us first assume that $n=1$ and hence $S_t$ is a single asset. Then one can define the predictability
of the asset as $$\nu = \frac{E[(S_{t-1}A)^2]}{E[S_t^2]}.$$ Let's try to understand this intuitively: the term in the 
numerator is the variance of our predictions, and the term in the denominator is the variance of our asset. Hence, when 
$\nu$ is small, our assets have higher variance than the predictions, which means that we cannot predict our assets well
using the model above. On the other hand, when $\nu$ is high, our predictions have higher variance than our assets, which
means we can predict quite well using the above autoregressive model. Therefore, we see that $\nu$ can be used as a measure
of mean reversion.

In general $S_t$ will contain more than just a single asset, hence for $n>1$ and inclusion of the portfolio weights, we extend our above simplification to the 
multivariate case: $$E[S_t x] = x^T \Gamma x,$$ where $\Gamma$ is the covariance matrix of all our assets. Translating
this to the next time step, we obtain $$E[S_t A x] = x^T A^T \Gamma A x,$$ from which we find $$\nu(x) = 
\frac{x^T A^T \Gamma A x}{x^T \Gamma x}.$$

## Fitting the weights
$A$ can be estimated via a simple Least Squares Regression: $$\hat{A} = (S_{t-1}^T S_{t-1})^{-1} S_{t-1}^T S_t.$$ This 
we can then use to find the weights $x$ that minimize the predictability of our portfolio. This we do in multiple steps.

First, as the covariance matrix $\Gamma$ is symmetric and positive semi-definite, it is possible the split $\Gamma$ as 
$\Gamma = \Gamma^{1/2} \Gamma^{1/2}$, and find the square root of the covariance matrix (known as the Cholesky decomposition). Secondly,
let us define a new vector $z = \Gamma^{1/2} x$, for which we have $x = \Gamma^{-1/2} z$. Plugging this back into the 
definition of predictability $\nu$ that we have above, we find $$\nu(z) = 
\frac{z^T \Gamma^{-1/2} A^T \Gamma A \Gamma^{-1/2} z}{z^T \Gamma^{-1/2} \Gamma \Gamma^{-1/2} z} = \frac{z^T \Gamma^{-1/2} A^T \Gamma A \Gamma^{-1/2} z}{z^T z}.$$ 
Having a nice and simple denominator, we can still simplify notation of the numerator by defining $B = 
\Gamma^{-1/2} A^T \Gamma A \Gamma^{-1/2}$. Plugging this back into predictability $\nu$ we finally get $$\nu(z) = 
\frac{z^T B z}{z^T z}.$$ This is a lot simpler, and so will be the minimization! For that, it suffices to find the 
eigenvector corresponding to the smallest eigenvalue of $B$. Our final portfolio weights will then be $$x = \Gamma^{-1/2} z.$$
Alternatively, one can create a portfolio that is as predictable as possible, having as much momentum as possible, by using the eigenvector corresponding 
to the biggest eigenvalue of $B$.

## Johansen procedure
This procedure can be used to test if multiple L(1) time series are co-integrated or not. L(1) refers to the lag of the
autoregressive processes: if it is L(1), then it means that if you would subtract the last value from your time series
and subtract it from the current value, then your time series becomes stationary. Similarly, L(2) would mean you need to 
subtract the last two values to make the series stationary, etcetera.

## Greedy search
So far we have been looking at constructing a mean-reverting portfolio but not a sparse one. We want to find the most mean
reverting portfolio in our entire asset universe that only contains at most $k$ assets. Let us formulate this problem 
mathematically: $$\mathop{\arg\min}_{x}\frac{x^T A^T \Gamma A x}{x^T \Gamma x},\qquad \text{Card}(x) \leq k,\, \lVert x \rVert = 1.$$
This makes sure that we find weights that minimize our portfolios predictability while having the number of non-zero elements
in our vector containing the weights to be smaller or equal to $k$. As usual, one would maximize (instead of minimize) to obtain
a momentum portfolio.

The optimization problem is NP-hard. Therefore, we aim for a suboptimal solution that is still good and fast enough. One 
of the algorithm that we can employ for this objective is called Greedy Search. In short, this algorithm does the following:

1. Use a brute force technique to get the most mean-revering pair of assets
2. Add one asset that will yield the most mean reverting triplet
3. Continue adding assets in this way until you reach $k$ number of assets

### Brute force to obtain best pair
To start out Greedy Search, we use brute force to find the most mean-reverting pair. For practicality -- under the realm of
diversification --  we also add the constraint that the weight assigned to a single asset should not be over 80%, as minimizing
predictability tends to give weights where often you have 99% or more of the total capital in one single asset, which is 
obviously not desirable. 

Also, even with a sparse portfolio, it does not make sense to have assets in our portfolio that have a tiny weight assigned 
to them. Therefore, we propose an additional constraint that the smallest weight in $x$ should be bigger than $1/(a \cdot k)$,
where $a$ is a parameter that determines the lower bound for the weight minimum. For example, when the goal is to construct
a portfolio of at maximum $k = 5$ assets, then $a=4$ ensures that each asset has a weight of at least 5%.

