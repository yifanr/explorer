import equinox as eqx
from jaxtyping import Array, Float, PyTree, Int  # https://github.com/google/jaxtyping'
from typing import List, Tuple
import numpy as np
import gymnasium as gym
import jax
import jax.numpy as jnp
import distrax
from util import calculate_similarity

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
        # self.embedding = eqx.nn.Embedding(num_embeddings, embedding_size, key=keys[6])
        self.embedding = jax.random.normal(keys[6], (num_embeddings, embedding_size))
        self.embedding /= 1000

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
        distances = jnp.sum((self.embedding - input) ** 2, axis=1)
        # similarities = calculate_similarity(self.embedding.weight, input)
        encoding_index = jnp.argmin(distances)
        # encoding_index = jnp.argmax(similarities)
        encoding = jax.nn.one_hot(encoding_index, self.num_embeddings)

        return input + jax.lax.stop_gradient(self.embedding[encoding_index] - input), encoding_index
    
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
            eqx.nn.Conv2d(in_channels, 4, 5, stride=1, key=keys[0]),
            jax.nn.selu,
            eqx.nn.Conv2d(4, 8, 5, stride=1, key=keys[1]),
            jax.nn.selu,
            eqx.nn.Conv2d(8, embedding_size, 5, stride=1, key=keys[2])
        ]
        self.decoder = [
            eqx.nn.ConvTranspose2d(embedding_size, 8, 5, stride=1, key=keys[3]),
            jax.nn.selu,
            eqx.nn.ConvTranspose2d(8, 4, 5, stride=1, key=keys[4]),
            jax.nn.selu,
            eqx.nn.ConvTranspose2d(4, in_channels, 5, stride=1, key=keys[5])
        ]
        self.embedding = jax.random.normal(keys[6], (num_embeddings, embedding_size))
        self.embedding /= 1000

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
        # Distances for every pair of codebook vector and encoding
        input = jnp.moveaxis(input, 0, 2)
        # input: h x w x c
        # embeddings: n x c
        distances = jnp.sum((self.embedding[:,None,None,:] - input) ** 2, axis=-1)
        # similarities = calculate_similarity(self.embedding.weight, input)
        encoding_indices = jnp.argmin(distances, axis=0)
        # encoding_index = jnp.argmax(similarities)
        encoding = jax.nn.one_hot(encoding_indices, self.num_embeddings)

        quantized = input + jax.lax.stop_gradient(self.embedding[encoding_indices] - input)

        return jnp.moveaxis(quantized, 2, 0), encoding_indices
    
class GSAE(eqx.Module):
    in_dim: int
    out_dim: int
    encoder: list
    decoder: list

    def __init__(self, in_dim, out_dim, key, ratio=0.5):
        self.in_dim = in_dim
        self.out_dim = out_dim
        num_keys=3
        keys = jax.random.split(key, num_keys)
        self.encoder = [
            eqx.nn.Linear(self.in_dim, 64, key=keys[0]),
            jax.nn.selu,
            eqx.nn.Linear(64, 64, key=keys[1]),
            jax.nn.selu,
            eqx.nn.Linear(64, self.out_dim*2, key=keys[2])
        ]
        self.decoder = [
            eqx.nn.Linear(self.out_dim*2, 64, key=keys[3]),
            jax.nn.selu,
            eqx.nn.Linear(64, 64, key=keys[4]),
            jax.nn.selu,
            eqx.nn.Linear(64, self.in_dim, key=keys[5])
        ]

    def sample_gumbel_noise(self, x, key, eps=1e-20): 
        """Sample from Gumbel(0, 1)"""
        U = distrax.Uniform(jnp.zeros_like(x), jnp.ones_like(x))
        sample = U.sample(seed=key)
        return -jnp.log(-jnp.log(sample + eps) + eps)

    def encode(self, input, temperature, key):
        x = input
        for layer in self.encoder:
            x = layer(x)

        x = jnp.reshape(x, (-1, 2))
        y = x + self.sample_gumbel_noise(x, key)
        return jax.nn.softmax(y / temperature, axis=1)
    
    def decode(self, encoding):
        x = jnp.ravel(encoding)
        for layer in self.decoder:
            x = layer(x)
        return x
        
class CGSAE(eqx.Module):
    in_channels: int
    embed_size: int
    encoder: list
    decoder: list

    def __init__(self, in_channels, embed_size, key, ratio=0.5):
        self.in_channels = in_channels
        self.embed_size = embed_size
        num_keys=3
        keys = jax.random.split(key, num_keys)
        self.encoder = [
            eqx.nn.Conv2d(in_channels, 32, 5, stride=1, key=keys[0]),
            jax.nn.selu,
            eqx.nn.Conv2d(32, 32, 5, stride=1, key=keys[1]),
            jax.nn.selu,
            eqx.nn.Conv2d(32, embed_size*2, 5, stride=1, key=keys[2])
        ]
        self.decoder = [
            eqx.nn.ConvTranspose2d(embed_size*2, 32, 5, stride=1, key=keys[3]),
            jax.nn.selu,
            eqx.nn.ConvTranspose2d(32, 32, 5, stride=1, key=keys[4]),
            jax.nn.selu,
            eqx.nn.ConvTranspose2d(32, in_channels, 5, stride=1, key=keys[5])
        ]

    def sample_gumbel_noise(self, x, key, eps=1e-20): 
        """Sample from Gumbel(0, 1)"""
        U = distrax.Uniform(jnp.zeros_like(x), jnp.ones_like(x))
        sample = U.sample(seed=key)
        return -jnp.log(-jnp.log(sample + eps) + eps)

    def encode(self, input, temperature, key):
        x = input
        for layer in self.encoder:
            x = layer(x)
        x = jnp.reshape(x, (-1, 2, x.shape[-2], x.shape[-1]))
        y = x + self.sample_gumbel_noise(x, key)

        return jax.nn.softmax(y / temperature, axis=1)
    
    def decode(self, encoding):
        x = encoding.reshape((-1,encoding.shape[-2],encoding.shape[-1]))
        for layer in self.decoder:
            x = layer(x)
        return x

    
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
        eps = 5e-3
        covariances = jax.lax.stop_gradient(self.covariances) + (jnp.identity(self.embedding_size) * eps)
        distributions = distrax.MultivariateNormalFullCovariance(jax.lax.stop_gradient(self.means.weight), covariances)
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
    
class CGMMVAE(eqx.Module):
    in_channels: int
    num_embeddings: int
    embedding_size: int
    means: eqx.nn.Embedding
    sizes: jax.Array
    covariances: jax.Array
    encoder: list
    decoder: list

    def __init__(self, in_channels, embedding_size, num_embeddings, key):
        self.in_channels = in_channels
        self.embedding_size = embedding_size
        self.num_embeddings = num_embeddings
        keys = jax.random.split(key, 13)
        self.encoder = [
            eqx.nn.Conv2d(in_channels, 4, 5, key=keys[0]),
            jax.nn.selu,
            eqx.nn.Conv2d(4, 4, 5, key=keys[1]),
            jax.nn.selu,
            eqx.nn.Conv2d(4, 8, 5, key=keys[2]),
            jax.nn.selu,
            eqx.nn.Conv2d(8, 8, 5, key=keys[3]),
            jax.nn.selu,
            eqx.nn.Conv2d(8, 16, 5, key=keys[4]),
            jax.nn.selu,
            jnp.ravel,
            eqx.nn.Linear(8*8*16, embedding_size, key=keys[11])
        ]
        self.decoder = [
            eqx.nn.Linear(embedding_size, 8*8*16, key=keys[12]), 
            jax.nn.selu,
            eqx.nn.ConvTranspose2d(16, 8, 5, key=keys[5]),
            jax.nn.selu,
            eqx.nn.ConvTranspose2d(8, 8, 5, key=keys[6]),
            jax.nn.selu,
            eqx.nn.ConvTranspose2d(8, 4, 5, key=keys[7]),
            jax.nn.selu,
            eqx.nn.ConvTranspose2d(4, 4, 5, key=keys[8]),
            jax.nn.selu,
            eqx.nn.ConvTranspose2d(4, in_channels, 5, key=keys[9])
        ]
        self.means = eqx.nn.Embedding(num_embeddings, embedding_size, key=keys[10])
        self.sizes = jnp.ones(num_embeddings) / num_embeddings
        cov = jnp.diag(jnp.ones(embedding_size))
        self.covariances = jnp.stack([cov]*num_embeddings)

    def encode(self, input):
        x = input
        for layer in self.encoder:
            x = layer(x)
        return jnp.ravel(x)
    
    def decode(self, encoding):
        x = self.decoder[0](encoding)
        x = jnp.reshape(x, (16,8,8))
        for layer in self.decoder[1:]:
            x = layer(x)
        return x
        
    def quantize(self, input, seed):
        eps = 5e-3
        covariances = jax.lax.stop_gradient(self.covariances) + (jnp.identity(self.embedding_size) * eps)
        distributions = distrax.MultivariateNormalFullCovariance(jax.lax.stop_gradient(self.means.weight), covariances)
        log_probs = jnp.log(jax.lax.stop_gradient(self.sizes)) + distributions.log_prob(input)
        cluster_dist = distrax.Categorical(logits=log_probs)
        key1, key2 = jax.random.split(seed, 2)
        cluster_index = cluster_dist.sample(seed=key1)
        cluster_index = jnp.argmax(log_probs)
        cluster = distrax.MultivariateNormalFullCovariance(self.means(cluster_index), self.covariances[cluster_index])
        sample = cluster.sample(seed=key2)

        return input + jax.lax.stop_gradient(sample - input), log_probs, cluster_index
    
