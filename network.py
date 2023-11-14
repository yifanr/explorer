import equinox as eqx
from jaxtyping import Array, Float, PyTree, Int  # https://github.com/google/jaxtyping'
from typing import List, Tuple
import numpy as np
import gymnasium as gym
import jax
import jax.numpy as jnp
import distrax

class VQVAE(eqx.Module):
    in_dim: int
    num_embeddings: int
    embedding_size: int
    embedding: eqx.nn.Embedding
    encoder: list
    decoder: list

    def __init__(self, in_dim, embedding_size, num_embeddings, key):
        self.in_dim = in_dim
        self.embedding_size = embedding_size
        self.num_embeddings = num_embeddings
        keys = jax.random.split(key, 7)
        dense_size = 2 * (in_dim + embedding_size)
        self.encoder = [
            eqx.nn.Linear(self.in_dim, dense_size, key=keys[0]),
            jax.nn.selu,
            eqx.nn.Linear(dense_size, dense_size, key=keys[1]),
            jax.nn.selu,
            eqx.nn.Linear(dense_size, self.embedding_size, key=keys[2])
        ]
        self.decoder = [
            eqx.nn.Linear(self.embedding_size, dense_size, key=keys[3]),
            jax.nn.selu,
            eqx.nn.Linear(dense_size, dense_size, key=keys[4]),
            jax.nn.selu,
            eqx.nn.Linear(dense_size, self.in_dim, key=keys[5])
        ]
        self.embedding = eqx.nn.Embedding(num_embeddings, embedding_size, key=keys[6])

    def encode(self, input):
        x = input
        for layer in self.encoder:
            x = layer(x)
        return x
    
    def decode(self, encoding):
        x = encoding
        for layer in self.decoder:
            x = layer(x)
        return x
        
    def quantize(self, input):
        distances = jnp.sum((self.embedding.weight - input) ** 2, axis=1)
        encoding_index = jnp.argmin(distances)
        encoding = jax.nn.one_hot(encoding_index, self.num_embeddings)

        return input + jax.lax.stop_gradient(self.embedding(encoding_index) - input), encoding_index
    
class AE(eqx.Module):
    in_dim: int
    embedding_size: int
    encoder: list
    decoder: list

    def __init__(self, in_dim, embedding_size, key):
        self.in_dim = in_dim
        self.embedding_size = embedding_size
        keys = jax.random.split(key, 7)
        dense_size = 2 * (in_dim + embedding_size)
        self.encoder = [
            eqx.nn.Linear(self.in_dim, dense_size, key=keys[0]),
            jax.nn.selu,
            eqx.nn.Linear(dense_size, dense_size, key=keys[1]),
            jax.nn.selu,
            eqx.nn.Linear(dense_size, self.embedding_size, key=keys[2])
        ]
        self.decoder = [
            eqx.nn.Linear(self.embedding_size, dense_size, key=keys[3]),
            jax.nn.selu,
            eqx.nn.Linear(dense_size, dense_size, key=keys[4]),
            jax.nn.selu,
            eqx.nn.Linear(dense_size, self.in_dim, key=keys[5])
        ]

    def encode(self, input):
        x = input
        for layer in self.encoder:
            x = layer(x)
        return x
    
    def decode(self, encoding):
        x = encoding
        for layer in self.decoder:
            x = layer(x)
        return x
        
    
class GMMVAE(eqx.Module):
    in_dim: int
    num_embeddings: int
    embedding_size: int
    means: eqx.nn.Embedding
    sizes: jax.Array
    covariances: jax.Array
    encoder: list
    decoder: list

    def __init__(self, in_dim, embedding_size, num_embeddings, key):
        self.in_dim = in_dim
        self.embedding_size = embedding_size
        self.num_embeddings = num_embeddings
        keys = jax.random.split(key, 8)
        dense_size = 2 * (in_dim + embedding_size)
        self.encoder = [
            eqx.nn.Linear(self.in_dim, dense_size, key=keys[0]),
            jax.nn.selu,
            eqx.nn.Linear(dense_size, dense_size, key=keys[1]),
            jax.nn.selu,
            eqx.nn.Linear(dense_size, self.embedding_size, key=keys[2])
        ]
        self.decoder = [
            eqx.nn.Linear(self.embedding_size, dense_size, key=keys[3]),
            jax.nn.selu,
            eqx.nn.Linear(dense_size, dense_size, key=keys[4]),
            jax.nn.selu,
            eqx.nn.Linear(dense_size, self.in_dim, key=keys[5])
        ]
        self.means = eqx.nn.Embedding(num_embeddings, embedding_size, key=keys[6])
        self.sizes = jnp.ones(num_embeddings) / num_embeddings
        cov = jnp.diag(jnp.ones(embedding_size))
        self.covariances = jnp.stack([cov]*num_embeddings)

    def encode(self, input):
        x = input
        for layer in self.encoder:
            x = layer(x)
        return x
    
    def decode(self, encoding):
        x = encoding
        for layer in self.decoder:
            x = layer(x)
        return x
        
    def quantize(self, input, seed):
        distributions = distrax.MultivariateNormalFullCovariance(jax.lax.stop_gradient(self.means.weight), jax.lax.stop_gradient(self.covariances))
        log_probs = jnp.log(jax.lax.stop_gradient(self.sizes)) + distributions.log_prob(input)
        cluster_dist = distrax.Categorical(logits=log_probs)
        key1, key2 = jax.random.split(seed, 2)
        cluster_index = cluster_dist.sample(seed=key1)
        cluster_index = jnp.argmax(log_probs)
        cluster = distrax.MultivariateNormalFullCovariance(self.means(cluster_index), self.covariances[cluster_index])
        sample = cluster.sample(seed=key2)

        return input + jax.lax.stop_gradient(sample - input), log_probs, cluster_index
    

class DGMMVAE(eqx.Module):
    in_dim: int
    num_embeddings: int
    embedding_size: int
    means: eqx.nn.Embedding
    sizes: jax.Array
    variances: jax.Array
    encoder: list
    decoder: list

    def __init__(self, in_dim, embedding_size, num_embeddings, key):
        self.in_dim = in_dim
        self.embedding_size = embedding_size
        self.num_embeddings = num_embeddings
        keys = jax.random.split(key, 8)
        dense_size = 2 * (in_dim + embedding_size)
        self.encoder = [
            eqx.nn.Linear(self.in_dim, dense_size, key=keys[0]),
            jax.nn.selu,
            eqx.nn.Linear(dense_size, dense_size, key=keys[1]),
            jax.nn.selu,
            eqx.nn.Linear(dense_size, self.embedding_size, key=keys[2])
        ]
        self.decoder = [
            eqx.nn.Linear(self.embedding_size, dense_size, key=keys[3]),
            jax.nn.selu,
            eqx.nn.Linear(dense_size, dense_size, key=keys[4]),
            jax.nn.selu,
            eqx.nn.Linear(dense_size, self.in_dim, key=keys[5])
        ]
        self.means = eqx.nn.Embedding(num_embeddings, embedding_size, key=keys[6])
        self.sizes = jnp.ones(num_embeddings) / num_embeddings
        self.variances = jnp.ones((num_embeddings, embedding_size))

    def encode(self, input):
        x = input
        for layer in self.encoder:
            x = layer(x)
        return x
    
    def decode(self, encoding):
        x = encoding
        for layer in self.decoder:
            x = layer(x)
        return x
        
    def quantize(self, input, seed):
        distributions = distrax.MultivariateNormalDiag(jax.lax.stop_gradient(self.means.weight), jax.lax.stop_gradient(self.variances))
        log_probs = jnp.log(jax.lax.stop_gradient(self.sizes)) + distributions.log_prob(input)
        cluster_dist = distrax.Categorical(logits=log_probs)
        key1, key2 = jax.random.split(seed, 2)
        cluster_index = cluster_dist.sample(seed=key1)
        cluster_index = jnp.argmax(log_probs)
        cluster = distrax.MultivariateNormalDiag(self.means(cluster_index), self.variances[cluster_index])
        sample = cluster.sample(seed=key2)

        return input + jax.lax.stop_gradient(sample - input), log_probs, cluster_index
    
# class GMMVAE(eqx.Module):
#     in_dim: int
#     num_embeddings: int
#     embedding_size: int
#     means: eqx.nn.Embedding
#     sizes: jax.Array
#     covariances: jax.Array
#     encoder: list
#     decoder: list

#     def __init__(self, in_dim, embedding_size, num_embeddings, key):
#         self.in_dim = in_dim
#         self.embedding_size = embedding_size
#         self.num_embeddings = num_embeddings
#         keys = jax.random.split(key, 8)
#         dense_size = 2 * (in_dim + embedding_size)
#         self.encoder = [
#             eqx.nn.Linear(self.in_dim, dense_size, key=keys[0]),
#             jax.nn.selu,
#             eqx.nn.Linear(dense_size, dense_size, key=keys[1]),
#             jax.nn.selu,
#             eqx.nn.Linear(dense_size, self.embedding_size, key=keys[2])
#         ]
#         self.decoder = [
#             eqx.nn.Linear(self.embedding_size, dense_size, key=keys[3]),
#             jax.nn.selu,
#             eqx.nn.Linear(dense_size, dense_size, key=keys[4]),
#             jax.nn.selu,
#             eqx.nn.Linear(dense_size, self.in_dim, key=keys[5])
#         ]
#         self.means = eqx.nn.Embedding(num_embeddings, embedding_size, key=keys[6])
#         self.sizes = jnp.ones(num_embeddings) / num_embeddings
#         cov = jnp.diag(jnp.ones(embedding_size))
#         self.covariances = jnp.stack([cov]*num_embeddings)

#     def encode(self, input):
#         x = input
#         for layer in self.encoder:
#             x = layer(x)
#         return x
    
#     def decode(self, encoding):
#         x = encoding
#         for layer in self.decoder:
#             x = layer(x)
#         return x
        
#     def quantize(self, input, seed):
#         distributions = distrax.MultivariateNormalFullCovariance(self.means.weight, self.covariances)
#         log_probs = jnp.log(self.sizes/jnp.sum(self.sizes)) + distributions.log_prob(input)
#         cluster_dist = distrax.Categorical(logits=log_probs)
#         key1, key2 = jax.random.split(seed, 2)
#         cluster_index = cluster_dist.sample(seed=key1)
#         cluster = distrax.MultivariateNormalFullCovariance(self.means(cluster_index), self.covariances[cluster_index])
#         sample = cluster.sample(seed=key2)

#         return input + jax.lax.stop_gradient(sample - input), log_probs, cluster_index
    
class CVQVAE(eqx.Module):
    in_channels: int
    num_embeddings: int
    embedding_size: int
    embedding: eqx.nn.Embedding
    encoder: list
    decoder: list

    def __init__(self, in_channels, embedding_size, num_embeddings, key):
        self.in_channels = in_channels
        self.embedding_size = embedding_size
        self.num_embeddings = num_embeddings
        keys = jax.random.split(key, 7)
        self.encoder = [
            eqx.nn.Conv2d(in_channels, 4, 3, key=keys[0]),
            jax.nn.selu,
            eqx.nn.Conv2d(4, 8, 3, key=keys[1]),
            jax.nn.selu,
            eqx.nn.Conv2d(8, 16, 3, key=keys[2])
        ]
        self.decoder = [
            eqx.nn.ConvTranspose2d(16, 8, 3, key=keys[3]),
            jax.nn.selu,
            eqx.nn.ConvTranspose2d(8, 4, 3, key=keys[4]),
            jax.nn.selu,
            eqx.nn.ConvTranspose2d(4, in_channels, 3, key=keys[5])
        ]
        self.embedding = eqx.nn.Embedding(num_embeddings, embedding_size, keys[6])

    def encode(self, input):
        x = input
        for layer in self.encoder:
            x = layer(x)
        return x
    
    def decode(self, encoding):
        x = encoding
        for layer in self.decoder:
            x = layer(x)
        return x
        
    def quantize(self, encoding):
        distances = jnp.sum((self.embedding.weight - input) ** 2, axis=1)
        encoding_index = jnp.argmin(distances)
        encoding = jax.nn.one_hot(encoding_index, self.num_embeddings)

        return self.embedding(encoding_index), encoding_index
    
    
