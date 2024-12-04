---
layout: distill
title: Variational inference is Bayesian inference is optimization
description: a tutorial on variational inference and interactive demo
tags: variational-inference demos
giscus_comments: false
date: 2024-12-5
featured: true

authors:
  - name: Luigi Acerbi
    url: "https://lacerbi.github.io/"
    affiliations:
      name: University of Helsinki, Helsinki, Finland

bibliography: 2024-12-5-vi-is-just-inference.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
  - name: What is variational inference?
    - name: Why do we need to approximate the target?
    - name: Making the intractable tractable
  - name: Variational inference on a general target density
    - name: The Evidence Lower BOund (ELBO)
  - name: Variational inference to approximate a target posterior
  - name: Variational inference is just optimization
    - name: Variational inference is just inference
  - name: Playing with variational inference
---

The goal of this post is to show that variational inference is a natural way of thinking about Bayesian inference and not some shady approximate method.<d-footnote>Unlike what the MCMC mafia wants you to think.</d-footnote>
At the end, you will also be able to directly play around with variational inference via an interactive visualization. In fact, you can also just skip the article and go straight to play with the [interactive thingie at the bottom](), and then come back if you feel like it.

## What is variational inference?

Let's start with the textbook definitions. We have a *target* distribution

$$
p^\star(\theta) = \frac{\widetilde{p}(\theta)}{\mathcal{Z}},
$$

which we know up to its normalization constant $$ \mathcal{Z} $$. At the core, variational inference is a way to approximate $$ p^\star(\theta) $$ having only the ability to evaluate the *unnormalized* target $$ \widetilde{p}(\theta) $$. The target can be continuous or discrete (or mixed), there are no restrictions!

If we go on reading a textbook, it will tell us that variational inference "approximates" the target with a (simpler) distribution $$ q_\psi(\theta) $$ parameterized by $$ \psi $$.

For example, if $$ q $$ is a multivariate normal, $$ \psi $$ could be the mean and covariance matrix of the distribution, $$ \psi = (\mu, \Sigma) $$. Please note that while normal distributions are a common choice in variational inference, they are not the only one -- you could choose $$ q $$ to be *any* distribution of your choice!

### Why do we need to approximate the target?

That is a great question. Why can't we just use the target as is? *Because we can't.*

Yes, we may be able to evaluate $$ \widetilde{p}(\theta) $$ for any chosen value of $$ \theta $$, but that alone does not tell us much.<d-footnote>Even knowing the normalization constant might not help that much.</d-footnote> What is the shape of this distribution? What are its moments? Its covariance structure? Does it have multiple modes? What is the expectation of an arbitrary function $$ f(\theta) $$ under the target? We may not know any of that!

*One way* to compute these values might be to get samples from the target... but how do we get those? How do we draw samples from the target if we only know an unnormalized $$ \widetilde{p}(\theta) $$?<d-footnote>Yes, one answer is MCMC (Markov Chain Monte Carlo), as surely you know thanks to the MCMC mafia. Point is, there are other answers.</d-footnote>

In short, we have our largely-unusable target and we would like to replace it with something that is easy to use and compute with for all the quantities we care about. There is an imponderable word for that: we want a distribution which is *tractable*.

###  Making the intractable tractable

This is the magic of what variational inference does: it takes an intractable target distribution and it gives back a *tractable* approximation $$ q $$, belonging to a class of our choice. We are using here tractable in a loose sense, meaning that at the very least we expect these properties:

- $$ q $$ is normalized
- We can draw samples from $$ q $$
- We can evaluate the density of $$ q $$ at any point

There are more precise and nuanced definitions of tractability based on the specific type of probabilistic queries we can compute in polynomial time (e.g., marginals, conditionals, expectations, etc.), and you are encouraged to read Choi et al. (2020)<d-cite key="choi2020probabilistic"></d-cite>.

## Variational inference on a general target density

So, how does $$ q $$ approximate the target? Intuitively, we want $$ q $$ to be as similar as possible to the *normalized* target $$ p^\star $$.

So we can take a measure of discrepancy between two distributions, and say that we want that discrepancy to be as small as possible. Traditionally, variational inference chooses the reverse [Kullback-Leibler (KL) divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) as its discrepancy function:<d-footnote>For concreteness, we write the following equations as integrals, but more generally they should be written as expectations (where the notation works for discrete, continuous, and mixed distributions).</d-footnote>

$$
\text{KL}(q_\psi(\theta) \,\mid\mid\, p^\star(\theta)) = \int q_\psi(\theta) \log \frac{q_\psi(\theta)}{p^\star(\theta)} \, d\theta
$$ 

This measures how the approximation $$ q_\psi(\theta) $$ diverges (differs) from the normalized target distribution $$ p^\star(\theta) $$. It is *reverse* because we put the approximation $$ q $$ first (the KL is not symmetric). The *direct* KL divergence would have the "real" target distribution $$ p^\star $$ first.

So for a given family of approximating distributions $$ q_\psi(\theta) $$, variational inference chooses the best value of the parameters $$ \psi $$ that make $$ q_\psi $$ "as close as possible" to $$ p^\star $$ by minimizing the KL divergence between $$ q_\psi $$ and $$ p^\star $$.

Done? Not quite yet.

### The Evidence Lower BOund (ELBO)

There is a caveat to the logic above: remember that we only have the unnormalized $$ \widetilde{p} $$, we do not have $$ p^\star $$! However, it turns out that this is no problem at all. First, we present the main results, and we will provide a full derivation after, for the interested readers.

Minimizing the KL divergence between $$ q_\psi $$ and $$ p^\star $$ can be achieved by maximizing the so-called ELBO, or Evidence Lower BOund, defined as:

$$
\text{ELBO}(q_\psi) = 
\underbrace{\int q_\psi(\theta) \log \widetilde{p}(\theta) \, d\theta}_{\text{Cross-entropy}} \;
\underbrace{- \int q_\psi(\theta) \log q_\psi(\theta) \, d\theta}_{\text{Entropy}}.
$$

First, note that the ELBO only depends on $$ q_\psi $$ and $$ \widetilde{p} $$.
The ELBO is indeed a lower bound to the log normalization constant, that is $$ \log \mathcal{Z} \ge \text{ELBO}(\psi)$$. It is composed of two terms, a cross-entropy term between $$ q $$ and $$ \widetilde{p} $$ and the **entropy** of $$ q $$. The two terms represent opposing forces:

- The (negative) [cross-entropy](https://en.wikipedia.org/wiki/Cross-entropy) term ensures that $$ q $$ avoids regions where $$ p $$ is low, shrinking towards high-density regions (mode-seeking behavior).
- The [entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)) term ensures that $$ q $$ is as spread-out as possible.

In conclusion, in variational inference we want to tweak the parameters $$ \psi $$ of $$ q $$ such that that the approximation $$ q_\psi $$ is as close as possible to $$ p^\star $$, according to the ELBO and, equivalently, to the KL divergence.

{% details Expand to see the full derivation of the ELBO %}

This is the full derivation of the ELBO, courtesy of `o1-mini` and `gpt-4o`, with just a sprinkle of human editing.

---

### **Step 1: Define the KL divergence**
The reverse Kullback-Leibler (KL) divergence between $$ q_\psi(\theta) $$ and the normalized target $$ p^\star(\theta) $$ is:

$$
\text{KL}(q_\psi(\theta) \,\mid\mid\, p^\star(\theta)) = \int q_\psi(\theta) \log \frac{q_\psi(\theta)}{p^\star(\theta)} \, d\theta
$$

---

### **Step 2: Express $$ p^\star(\theta) $$ in terms of $$ \widetilde{p}(\theta) $$**
The normalized target $$ p^\star(\theta) $$ is related to the unnormalized target $$ \widetilde{p}(\theta) $$ through the normalization constant $$ \mathcal{Z} $$:

$$
p^\star(\theta) = \frac{\widetilde{p}(\theta)}{\mathcal{Z}}, \quad \text{where} \quad \mathcal{Z} = \int \widetilde{p}(\theta) \, d\theta.
$$

Substitute this expression for $$ p^\star(\theta) $$ into the KL divergence:

$$
\text{KL}(q_\psi(\theta) \,\mid\mid\, p^\star(\theta)) = \int q_\psi(\theta) \log \left( q_\psi(\theta) \cdot \frac{\mathcal{Z}}{\widetilde{p}(\theta)} \right) \, d\theta
$$

---

### **Step 3: Split the logarithm**
Using the property of logarithms $$ \log(ab) = \log(a) + \log(b) $$, split the term inside the integral:

$$
\text{KL}(q_\psi(\theta) \,\mid\mid\, p^\star(\theta)) = \int q_\psi(\theta) \big( \log q_\psi(\theta) + \log \mathcal{Z} - \log \widetilde{p}(\theta) \big) \, d\theta
$$

---

### **Step 4: Separate the terms**
Distribute $$ q_\psi(\theta) $$ over the sum:

$$
\text{KL}(q_\psi(\theta) \,\mid\mid\, p^\star(\theta)) = \int q_\psi(\theta) \log q_\psi(\theta) \, d\theta + \int q_\psi(\theta) \log \mathcal{Z} \, d\theta - \int q_\psi(\theta) \log \widetilde{p}(\theta) \, d\theta
$$

---

### **Step 5: Simplify the second term**
Since $$ \mathcal{Z} $$ is a constant, $$ \log \mathcal{Z} $$ is also constant and can be factored out of the integral:

$$
\int q_\psi(\theta) \log \mathcal{Z} \, d\theta = \log \mathcal{Z} \int q_\psi(\theta) \, d\theta
$$

Because $$ q_\psi(\theta) $$ is a valid, normalized probability distribution, $$ \int q_\psi(\theta) \, d\theta = 1 $$. Therefore:

$$
\int q_\psi(\theta) \log \mathcal{Z} \, d\theta = \log \mathcal{Z}
$$

Substitute this simplification back into the KL divergence:

$$
\text{KL}(q_\psi(\theta) \,\mid\mid\, p^\star(\theta)) = \int q_\psi(\theta) \log q_\psi(\theta) \, d\theta + \log \mathcal{Z} - \int q_\psi(\theta) \log \widetilde{p}(\theta) \, d\theta
$$

---

### **Step 6: Rearrange terms**
Rearrange the equation to isolate $$ \log \mathcal{Z} $$, grouping terms related to $$ q_\psi(\theta) $$:

$$
\log \mathcal{Z} = \text{KL}(q_\psi(\theta) \,\mid\mid\, p^\star(\theta)) + \left( \int q_\psi(\theta) \log \widetilde{p}(\theta) \, d\theta - \int q_\psi(\theta) \log q_\psi(\theta) \, d\theta \right)
$$

---

### **Step 7: Define the ELBO**
The ELBO is defined as:

$$
\text{ELBO}(q_\psi) = \int q_\psi(\theta) \log \widetilde{p}(\theta) \, d\theta - \int q_\psi(\theta) \log q_\psi(\theta) \, d\theta
$$

Substitute this into the equation for $$ \log \mathcal{Z} $$:

$$
\log \mathcal{Z} = \text{KL}(q_\psi(\theta) \,\mid\mid\, p^\star(\theta)) + \text{ELBO}(q_\psi)
$$

---

### **Step 8: Rearrange for the ELBO**
Rearranging to isolate $$ \text{ELBO}(q_\psi) $$:

$$
\text{ELBO}(q_\psi) = \log \mathcal{Z} - \text{KL}(q_\psi(\theta) \,\mid\mid\, p^\star(\theta))
$$

---

### **Step 9: Interpretation**
- $$ \log \mathcal{Z} $$ is a constant with respect to $$ q_\psi(\theta) $$.
- To minimize $$ \text{KL}(q_\psi(\theta) \,\mid\mid\, p^\star(\theta)) $$, we maximize $$ \text{ELBO}(q_\psi) $$.

Thus, **minimizing the KL divergence is equivalent to maximizing the ELBO**.

Moreover, since the $\text{KL}$ divergence is non-negative and zero if $p = q$: 
- $\text{ELBO}(q_\psi) \le \log \mathcal{Z} \Longrightarrow$ the ELBO is a lower bound to $\log Z$.
- If $q = p$, $\text{ELBO}(q_\psi) = \log \mathcal{Z}$.

{% enddetails %}


## Variational inference to approximate a target posterior

While variational inference can be performed for any generic target density $$ \widetilde{p}(\theta) $$, the common scenario is that our target density is a *posterior distribution*:

$$
{p^\star}(\theta) \equiv p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \theta) \pi(\theta)}{p(\mathcal{D})}
$$

where you should recognize on the right-hand side good old Bayes' theorem, with $$ p(\mathcal{D} \mid \theta) $$ the *likelihood* and $$ \pi(\theta) $$ the *prior*.<d-footnote>We denote the prior with $\pi$ to avoid confusion with the target.</d-footnote> The normalization constant at the denominator is $\mathcal{Z} \equiv p(\mathcal{D})$, also called the *model evidence* or *marginal likelihood*.

Of course, we invariably do not know the normalization constant, but we can instead compute the *unnormalized* posterior:

$$
\widetilde{p}(\theta) = p(\mathcal{D} \mid \theta) \pi(\theta).
$$

In this typical usage-case scenario for variational inference, the ELBO reads:

$$
\text{ELBO}(q_\psi) = \mathbb{E}_{q_\psi(\theta)}\left[ \log p(\mathcal{D} \mid \theta) \pi(\theta) \right] - \mathbb{E}_{q_\psi(\theta)}\left[\log q_\psi(\theta)\right]
$$

where we simply replaced $$ \widetilde{p} $$ with the unnormalized posterior, and we switched here to the expectation notation, instead of integrals, just to show you how that would look like.

## Variational inference is just optimization

Variational inference reduces Bayesian inference to an optimization problem. 
You have a candidate solution $q$, shake it and twist it and spread it around until you maximize the ELBO.
Variational inference per se is nothing more than this.

Most variational inference *algorithms* focus on:
- Specific families of $q$ (e.g., factorized, exponential families, etc.)
- Specific ways of estimating and optimizing the ELBO (block-wise coordinate updates, stochastic gradient ascent, etc.)

But these are all implementation details.<d-footnote>There are also variational inference methods that use other divergences than the reverse KL divergence, but we lose the meaning of the ELBO as a lower bound to the normalization constant.</d-footnote>

### Variational inference is just inference

For the reasons mentioned above, I believe that variational inference is possibly the most natural way of thinking about Bayesian inference: computing the posterior is not some esoteric procedure, but we are just literally trying to find the distribution that best matches the true target posterior.

Variational inference is often seen as "just an approximation method" -- as opposed to a true technique for performing Bayesian inference -- because historically we were forced to use very simple approximation families. However, it has been a while since we can use very flexible distributions, starting for example from the advent of normalizing flows.

## Playing with variational inference

In the widget below (full page [here](https://lacerbi.github.io/interactive-vi-demo/)) you can see variational inference for yourself. Move around the distributions (or just lazily press "Optimize") and see the ELBO value go up, getting closer and closer to the true $$ log \mathcal{Z} $$. It is very satisfying.


<iframe
    src="https://lacerbi.github.io/interactive-vi-demo/"
    width="100%"
    height="700px"
    style="border: none;"
    title="Interactive Variational Inference Demo">
</iframe>
