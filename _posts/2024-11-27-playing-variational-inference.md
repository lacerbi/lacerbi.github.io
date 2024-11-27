---
layout: distill
title: Playing with variational inference
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

bibliography: 2024-11-27-distill.bib

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
## Variational inference on a general target density

In general, variational inference approximates a target (unnormalized) distribution $$ p_\text{target}(\theta) $$ with a simpler distribution $$ q_\psi(\theta) $$ parameterized by $$ \psi $$.

For example, if $$ q $$ is a multivariate normal, $$ \psi $$ could be the mean and covariance matrix of the distribution, $$ \psi = (\mu, \Sigma) $$. Please note that while normal distributions are a common choice in variational inference, they are not the only one -- you could choose $$ q $$ to be *any* distribution of your choice!

For a given family of approximating distributions $$ q_\psi(\theta) $$, variational inference chooses the best value of the parameters $$ \psi $$ that make $$ q_\psi $$ "as close as possible" to $$ p $$ by maximizing the ELBO (evidence lower bound):  

$$
\text{ELBO}(\psi) = \mathbb{E}_{q_\psi(\theta)}\left[ \log p_\text{target}(\theta)\right] - \mathbb{E}_{q_\psi(\theta)}\left[\log q_\psi(\theta)\right]
$$

It can be shown that maximizing the ELBO is equivalent to minimizing $$ D_\text{KL}(q_\psi||p_\text{target}) $$, which is the Kullback-Leibler divergence between $$ q_\psi(\theta) $$ and $$ p_\text{target}(\theta) $$.

## Variational inference to approximate a target posterior

While variational inference could be performed for any generic target density $$ p_\text{target}(\theta) $$, the common scenario is that our target density is an unnormalized posterior distribution: 

$$
p_\text{target}(\theta) = p(\mathcal{D} | \theta) p(\theta) \propto \frac{p(\mathcal{D} | \theta) p(\theta)}{p(\mathcal{D})}
$$

where $$ p(\mathcal{D} | \theta) p(\theta) = p(\mathcal{D}, \theta) $$ is the joint distribution. The (unknown) normalization constant is $p(\mathcal{D})$, also called the *model evidence* or *marginal likelihood*. In this typical usage-case scenario for variational inference, the ELBO reads

$$
\text{ELBO}(\psi) = \mathbb{E}_{q_\psi(\theta)}\left[ \log p(\mathcal{D}|\theta) p(\theta) \right] - \mathbb{E}_{q_\psi(\theta)}\left[\log q_\psi(\theta)\right]
$$

where we simply replaced $$ p_\text{target} $$ with the unnormalized posterior.

## Things to know

- The term $$ \mathbb{E}_{q_\psi(\theta)}\left[ \log p(\theta|\mathcal{D}) p(\theta) \right] $$ in the ELBO is the expected log joint.
- The term $$ -\mathbb{E}_{q_\psi(\theta)}\left[ \log q_\psi(\theta) \right] $$ is the *entropy* of $q_{\psi}(\theta)$, often written as $\mathcal{H}[q]$.
- Note that the ELBO is a function of $$ \psi $$. The optimization finds the $$ \psi^* $$ that maximizes the ELBO (in practice, the value $$ \psi^* $$ that minimizes the negative ELBO).
- The ELBO is a lower bound to the log normalization constant of the target density, that is $$ \log p(\mathcal{D}) $$ when the target is the unnormalized posterior.
- For notational convenience, the dependence of $$ q_\psi(\theta) $$ on $$ \psi $$ is often omitted. Also $$ \psi$ $ is an arbitrary notation, you will find other (Greek) letters to denote the variational parameters.