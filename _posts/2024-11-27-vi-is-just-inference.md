---
layout: distill
title: Variational inference is just Bayesian inference
description: an interactive tutorial on variational inference
tags: variational-inference
giscus_comments: false
date: 2024-11-27
featured: true

authors:
  - name: Luigi Acerbi
    url: "https://en.wikipedia.org/wiki/Albert_Einstein"
    affiliations:
      name: University of Helsinki, Helsinki, Finland

bibliography: 2024-11-27-vi-is-just-inference.bib

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
  - name: Variational inference on a general target density
  - name: Variational inference to approximate a target posterior
  - name: Things to know
---

The goal of this post is to show you that variational inference is a very natural way of thinking about Bayesian inference and not some shady approximate method.<d-footnote>Unlike what the MCMC mafia wants you to think.</d-footnote>
At the end, you will also be able to directly play around with variational inference via an interactive visualization. In fact, you can also just skip the article and go straight to play with the interactive thingie at the bottom, and then come back if you feel like it.

## What is variational inference

Let's start with the textbook definitions. We have a *target* distribution

$$
p^\star(\theta) = \frac{\widetilde{p}(\theta)}{\mathcal{Z}},
$$

which we know up to its normalization constant $$ \mathcal{Z} $$. At the core, variational inference is a way to approximate $$ p^\star(\theta) $$ having only the ability to evaluate the *unnormalized* target $$ \widetilde{p}(\theta) $$. The target can be continuous or discrete (or mixed), there are no restrictions!

If we go on reading a textbook, it will tell us that variational inference "approximates" the target with a (simpler) distribution $$ q_\psi(\theta) $$ parameterized by $$ \psi $$.

For example, if $$ q $$ is a multivariate normal, $$ \psi $$ could be the mean and covariance matrix of the distribution, $$ \psi = (\mu, \Sigma) $$. Please note that while normal distributions are a common choice in variational inference, they are not the only one -- you could choose $$ q $$ to be *any* distribution of your choice!

### Why do we need to approximate the target?

That is a great question. Why can't we just use the target as is? *Because we can't.*

Yes, we may be able to evaluate $$ \widetilde{p}(\theta) $$ for any chosen value of $$ \theta $$, but that alone does not tell us much.<d-footnote>Even knowing $$ \mathcal{Z} $$ might not help that much.</d-footnote> What is the shape of this distribution? What are its moments? Its covariance structure? Does it have multiple modes? What is the expectation of an arbitrary function $$ f(\theta) $$ under the target? We may not know any of that!

*One way* to compute these values might be to get samples from the target... but how do we get those? How do we draw samples from the target if we only know an unnormalized $$ \widetilde{p}(\theta) $$?<d-footnote>Yes, one answer is MCMC (Markov Chain Monte Carlo), as surely you know thanks to the MCMC mafia. Point is, there are *other* answers.</d-footnote>

In short, we have our largely-unusable target and we would like to replace it with something that is easy to use and compute with for all the quantities we care about. There is a magical imponderable word for that: we want a distribution which is *tractable*.

###  Making the target tractable

This is the key of what variational inference does: it takes an intractable target distribution and it gives back a *tractable* approximation $$ q $$, belonging to a class of our choice. What tractable exactly means is up for discussion, but at the very least we expect these properties:

- $$ q $$ is normalized
- We can draw samples from $$ q $$
- We can evaluate the density of $$ q $$ at any point

There is potentially a whole variety of desiderata for a tractable distribution, and you are encouraged to read Choi et al. (2020)<d-cite key="choi2020probabilistic"></d-cite>.

## Variational inference on a general target density

So, how does $$ q $$ approximate the target? Intuitively, we want $$ q $$ to be as similar as possible to the *normalized* target $$ p^\star $$.

So we can take a measure of discrepancy between two distributions, and say that we want that discrepancy to be as small as possible. Traditionally, variational inference chooses the reverse [Kullback-Leibler (KL) divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) as its discrepancy function:

$$
\text{KL}(q_\psi(\theta) \,\mid\mid\, p^\star(\theta)) = \int q_\psi(\theta) \log \frac{q_\psi(\theta)}{p^\star{target}(\theta)} \, d\theta
$$ 

This measures how the approximation $$ q_\psi(\theta) $$ diverges (differs) from the normalized target distribution $$ p^\star(\theta) $$. It is *reverse* because we put the approximation $$ q $$ first (the KL is not symmetric). The *direct* KL divergence would have the "real" target distribution $$ p^\star $$ first.

So for a given family of approximating distributions $$ q_\psi(\theta) $$, variational inference chooses the best value of the parameters $$ \psi $$ that make $$ q_\psi $$ "as close as possible" to $$ p^\star $$ by minimizing the KL divergence between $$ q_\psi $$ and $$ p^\star $$.

## The Evidence Lower BOund (ELBO)

There is a caveat to the logic above: remember that we only have the unnormalized $$ \widetilde{p} $$, we do not have $$ p^\star $$! However, it turns out that this is no problem at all.

We can demonstrate (open below if interested) that minimizing the KL divergence between $$ q_\psi $$ and $$ p^\star $$ can be achieved by maximizing the quantity defined as

$$
\text{ELBO}(\psi) = \mathbb{E}_{q_\psi(\theta)}\left[ \log p_\text{target}(\theta)\right] - \mathbb{E}_{q_\psi(\theta)}\left[\log q_\psi(\theta)\right]
$$

where the ELBO (Evidence Lower BOund) is indeed a lower bound to the log normalization constant, that is $$ \log \mathcal{Z} \ge \text{ELBO}(\psi)$$.

In other words, we can tweak the parameters $$ \psi $$ of $$ q $$ such that that the approximation is as close as possible to $$ p^\star $$, according to the ELBO and, equivalently, to the KL divergence.

## Variational inference to approximate a target posterior

While variational inference could be performed for any generic target density $$ p_\text{target}(\theta) $$, the common scenario is that our target density is an unnormalized posterior distribution: 

$$
p_\text{target}(\theta) = p(\mathcal{D} \mid \theta) p(\theta) \propto \frac{p(\mathcal{D} \mid \theta) p(\theta)}{p(\mathcal{D})}
$$

where $$ p(\mathcal{D} \mid \theta) p(\theta) = p(\mathcal{D}, \theta) $$ is the joint distribution. The (unknown) normalization constant is $p(\mathcal{D})$, also called the *model evidence* or *marginal likelihood*. In this typical usage-case scenario for variational inference, the ELBO reads

$$
\text{ELBO}(\psi) = \mathbb{E}_{q_\psi(\theta)}\left[ \log p(\mathcal{D} \mid \theta) p(\theta) \right] - \mathbb{E}_{q_\psi(\theta)}\left[\log q_\psi(\theta)\right]
$$

where we simply replaced $$ p_\text{target} $$ with the unnormalized posterior.

## Things to know

- The term $$ \mathbb{E}_{q_\psi(\theta)}\left[ \log p(\theta \mid \mathcal{D}) p(\theta) \right] $$ in the ELBO is the expected log joint.
- The term $$ -\mathbb{E}_{q_\psi(\theta)}\left[ \log q_\psi(\theta) \right] $$ is the *entropy* of $q_{\psi}(\theta)$, often written as $\mathcal{H}[q]$.
- Note that the ELBO is a function of $$ \psi $$. The optimization finds the $$ \psi^* $$ that maximizes the ELBO (in practice, the value $$ \psi^* $$ that minimizes the negative ELBO).
- The ELBO is a lower bound to the log normalization constant of the target density, that is $$ \log p(\mathcal{D}) $$ when the target is the unnormalized posterior.
- For notational convenience, the dependence of $$ q_\psi(\theta) $$ on $$ \psi $$ is often omitted. Also $$ \psi$ $ is an arbitrary notation, you will find other (Greek) letters to denote the variational parameters.

<iframe
    src="https://lacerbi.github.io/interactive-vi-demo/"
    width="100%"
    height="500px"
    style="border: none;"
    title="Interactive Variational Inference Demo">
</iframe>
