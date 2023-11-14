import jax.numpy as jnp
import jax
import distrax

def is_pos_def(x):
    return jnp.all(jnp.linalg.eigvals(x) > 0)

key = jax.random.PRNGKey(43)
keys = jax.random.split(key, 5)

Q = jax.random.uniform(keys[0], shape=(4,4), minval=-1, maxval=1)
D = jax.random.uniform(keys[1], shape=(4,), minval=-1, maxval=1)
D = jnp.diag(jnp.exp(D))
print(D)
print(Q)
covariance = Q.T @ D @ Q
print(covariance)
print(jnp.linalg.eigh(covariance))
mean = jax.random.uniform(keys[2], shape=(4,), minval=-1, maxval=1)


# key = jax.random.PRNGKey(43)
# keys = jax.random.split(key, 5)

# Q = jax.random.uniform(keys[0], shape=(4,4), minval=-1, maxval=1)
# D = jax.random.uniform(keys[1], shape=(4,), minval=-1, maxval=1)
# D = jnp.diag(D)
# covariance2 = Q @ D @ Q.T 
# mean2 = jax.random.uniform(keys[2], shape=(4,), minval=-1, maxval=1)

# mean = jnp.array((mean, mean2))
# covariance = jnp.array((covariance, covariance2))
# print(mean.shape)
# print(covariance.shape)

distribution = distrax.MultivariateNormalFullCovariance(mean, covariance)

print(distribution.sample(seed=2))

# key = jax.random.PRNGKey(42)
# keys = jax.random.split(key, 5)

# Q = jax.random.uniform(keys[0], shape=(2, 4,4), minval=-1, maxval=1)
# D = jax.random.uniform(keys[1], shape=(2, 4,), minval=-1, maxval=1)
# D = jax.vmap(jnp.diag)(D)

# print(Q.shape)
# print(D.shape)

# covariance = Q @ D @ Q.swapaxes(1,2)
# mean = jax.random.uniform(keys[2], shape=(2, 4,), minval=-1, maxval=1)

# print(covariance.shape)
# distribution = distrax.MultivariateNormalFullCovariance(mean, covariance)

# print(distribution.sample(seed=2))