import jax.numpy as jnp

def calculate_similarity(x, y):
    return jnp.dot(x, y)/(jnp.linalg.norm(x)*jnp.linalg.norm(y))