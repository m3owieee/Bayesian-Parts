import scipy.stats as sts
import numpy as np
import matplotlib.pyplot as plt

# Assumes mean height between 1.65 and 1.8.
# Creates 50 values saved as an array to variable mu.
mu = np.linspace(1.65, 1.8, num=50)

# Prior Distributions
uniform_dist = np.ones_like(mu) / len(mu)  # Non-informative prior
beta_dist = sts.beta.pdf(mu, 2, 5, loc=1.65, scale=0.15)  # Subjective prior
beta_dist = beta_dist / beta_dist.sum()  # Normalize

plt.plot(mu, beta_dist, label="Beta Dist (subjective prior)")
plt.plot(mu, uniform_dist, label="Uniform Dist (non-informative prior)")
plt.title("Prior Probability Distributions of Hypothesized $\mu$")
plt.xlabel("Mean Height $\mu$ (meters)")
plt.ylabel("Probability Density")
plt.legend()
plt.show()


# Likelihood Function
def likelihood_func(datum, mu):
    likelihood_out = sts.norm.pdf(datum, mu, scale=0.1)
    return likelihood_out / likelihood_out.sum()


# Compute likelihood for observed height = 1.7m
likelihood_out = likelihood_func(1.7, mu)

plt.plot(mu, beta_dist, label="Beta Dist (subjective prior)")
plt.plot(mu, uniform_dist, label="Uniform Dist (non-informative prior)")
plt.plot(mu, likelihood_out, label="Likelihood Distribution")
plt.title("Likelihood Function of Hypothesized $\mu$ Given Observation 1.7m")
plt.xlabel("Mean Height $\mu$ (meters)")
plt.ylabel("Probability Density / Likelihood")
plt.legend()
plt.show()


# Compute the Unnormalized Posterior
unnormalized_posterior = likelihood_out * uniform_dist

plt.plot(mu, beta_dist, label="Beta Dist (subjective prior)")
plt.plot(mu, uniform_dist, label="Uniform Dist (non-informative prior)")
plt.plot(mu, likelihood_out, label="Likelihood Distribution")
plt.plot(mu, unnormalized_posterior, label="Unnormalized Posterior")
plt.title("Unnormalized Posterior Distribution of Hypothesized $\mu$")
plt.xlabel("Mean Height $\mu$ (meters)")
plt.ylabel("Probability Density / Likelihood")
plt.legend()
plt.show()


# Compute the Normalized Posterior
normalized_posterior = unnormalized_posterior / unnormalized_posterior.sum()

plt.plot(mu, beta_dist, label="Beta Dist (subjective prior)")
plt.plot(mu, uniform_dist, label="Uniform Dist (non-informative prior)")
plt.plot(mu, likelihood_out, label="Likelihood Distribution")
plt.plot(mu, unnormalized_posterior, label="Unnormalized Posterior")
plt.plot(mu, normalized_posterior, label="Normalized Posterior")
plt.title("Final Posterior Probability Distribution of $\mu$ Given Observation 1.7m")
plt.xlabel("Mean Height $\mu$ (meters)")
plt.ylabel("Probability Density")
plt.legend()
plt.show()

# The final plot now represents the actual posterior PDF.
