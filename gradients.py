from network import VQVAE, AE, GMMVAE, DGMMVAE, GSAE
import equinox as eqx
import jax
import jax.numpy as jnp
import distrax
from util import calculate_similarity

@eqx.filter_value_and_grad
def update_VQ(model: VQVAE, input):
    # input: array of shape (n, d), where d is the size of expected inputs.

    encodings = eqx.filter_vmap(model.encode)(input)
    quantized, indices = eqx.filter_vmap(model.quantize)(encodings)
    codebook_vectors = model.embedding[indices]
    reconstructions = eqx.filter_vmap(model.decode)(quantized)

    e_latent_loss = jnp.mean((jax.lax.stop_gradient(codebook_vectors) - encodings) ** 2)
    q_latent_loss = jnp.mean((jax.lax.stop_gradient(encodings) - codebook_vectors) ** 2)
    reconstruction_loss = jnp.mean((reconstructions - input) ** 2)
    
    return reconstruction_loss + e_latent_loss + q_latent_loss * 0.9

@eqx.filter_value_and_grad
def update_CVQ(model: VQVAE, input):
    # input: array of shape (n, d), where d is the size of expected inputs.

    encodings = eqx.filter_vmap(model.encode)(input)
    quantized, indices = eqx.filter_vmap(model.quantize)(encodings)
    codebook_vectors = jnp.moveaxis(model.embedding[indices], -1, -3)
    reconstructions = eqx.filter_vmap(model.decode)(quantized)

    e_latent_loss = jnp.mean((jax.lax.stop_gradient(codebook_vectors) - encodings) ** 2)
    q_latent_loss = jnp.mean((jax.lax.stop_gradient(encodings) - codebook_vectors) ** 2)
    reconstruction_loss = jnp.mean((reconstructions - input) ** 2)
    
    return reconstruction_loss + e_latent_loss + q_latent_loss * 0.9

@eqx.filter_value_and_grad
def update_GS(model: GSAE, input, temperature, key):
    # input: array of shape (n, d), where d is the size of expected inputs.

    encodings = eqx.filter_vmap(model.encode, in_axes=(0,None,None))(input, temperature, key)
    reconstructions = eqx.filter_vmap(model.decode)(encodings)

    reconstruction_loss = jnp.mean((reconstructions - input) ** 2)
    
    return reconstruction_loss

@eqx.filter_value_and_grad
def update_CGS(model: GSAE, input, temperature, key):
    # input: array of shape (n, d), where d is the size of expected inputs.

    encodings = eqx.filter_vmap(model.encode, in_axes=(0,None,None))(input, temperature, key)
    reconstructions = eqx.filter_vmap(model.decode)(encodings)

    reconstruction_loss = jnp.mean((reconstructions - input) ** 2)
    
    return reconstruction_loss


@eqx.filter_value_and_grad
def update_AE(AE: AE, input):
    # input: array of shape (n, d), where d is the size of expected inputs.

    encodings = eqx.filter_vmap(AE.encode)(input)
    reconstructions = eqx.filter_vmap(AE.decode)(encodings)

    reconstruction_loss = jnp.mean((reconstructions - input) ** 2)
    
    return reconstruction_loss

# def stable_mse(x, y):
#     return 

@eqx.filter_value_and_grad(has_aux=True)
def update_GMM(model: GMMVAE, input, key):
    # input: array of shape (n, d), where d is the size of expected inputs.

    subkeys = jax.random.split(key, input.shape[0])

    encodings = eqx.filter_vmap(model.encode)(input)
    quantized, log_probs, indices = eqx.filter_vmap(model.quantize)(encodings, subkeys)
    centers = model.means.weight[indices]
    # print(jnp.count_nonzero(jnp.isnan(log_probs)))
    reconstructions = eqx.filter_vmap(model.decode)(encodings)

    # print(jnp.max(log_probs))

    # e_latent_loss = jnp.mean((jax.lax.stop_gradient(quantized) - encodings) ** 2)
    # q_latent_loss = jnp.mean((jax.lax.stop_gradient(encodings) - quantized) ** 2)
    reconstruction_loss = jnp.mean((reconstructions - input) ** 2)
    # print("_______________Testing______________________")
    r = jax.nn.softmax(log_probs, axis=1) + 1e-8
    # print(r.shape)
    m = jnp.sum(r, axis=0)
    # print(m.shape)
    pi = m / jnp.sum(m)
    # print(pi.shape)
    mu = (1 / m[:,None]) * jnp.sum(r[:,:,None] * encodings[:,None,:], axis=0)
    # print(mu.shape)
    distances = encodings[:,None,:] - model.means.weight[None,:,:]
    # print(distances.shape)
    sigma = (1 / m[:,None,None]) * jnp.sum(r[:,:,None,None] * (distances[:,:,None,:]*distances[:,:,:,None]), axis=0)
    # print(sigma.shape)
    # print("_____________________________________________")
    scale_factor = jnp.minimum(10, (1 / (jax.lax.stop_gradient(model.sizes) * model.num_embeddings)))
    # scale_factor = jnp.ones_like(model.sizes)
    size_loss = jnp.sum(jnp.square(model.sizes - pi) * scale_factor)
    mean_loss = jnp.sum(jnp.square(model.means.weight - mu)  * scale_factor[:,None])
    covariance_loss = jnp.sum(jnp.square(model.covariances - sigma) * scale_factor[:,None,None])
    commitment_loss = jnp.sum((encodings - jax.lax.stop_gradient(centers)) ** 2)
    
    return reconstruction_loss + size_loss + mean_loss + covariance_loss + (commitment_loss * 0.1), (reconstruction_loss, size_loss, mean_loss, covariance_loss)


@eqx.filter_value_and_grad(has_aux=True)
def update_DGMM(model: DGMMVAE, input, key):
    # input: array of shape (n, d), where d is the size of expected inputs.

    subkeys = jax.random.split(key, input.shape[0])

    encodings = eqx.filter_vmap(model.encode)(input)
    quantized, log_probs, _ = eqx.filter_vmap(model.quantize)(encodings, subkeys)
    reconstructions = eqx.filter_vmap(model.decode)(quantized)
    # reencodings = eqx.filter_vmap()

    reconstruction_loss = jnp.mean((reconstructions - input) ** 2)

    eps = 1e-4
    r = jax.nn.softmax(log_probs, axis=1) + eps
    #(64, 32) - responsibility of each cluster-point pair
    m = jnp.sum(r, axis=0)
    #(32) - responsibility for each cluster
    pi = m / jnp.sum(m)
    #(32) - fraction of total responsibility for each cluster
    mu = (1 / m[:,None]) * jnp.sum(r[:,:,None] * encodings[:,None,:], axis=0)
    #(32, 8) - weighted average of points for each cluster
    distances = encodings[:,None,:] - model.means.weight[None,:,:]
    #(64, 32, 8) - pairwise distances between cluster centers and points
    sigma = (1 / m[:,None]) * jnp.sum(r[:,:,None] * (distances ** 2), axis=0)
    #(32, 8) - weighted variance for clusters by variable
    # print(jnp.min(sigma))
    # print(jnp.max(sigma))

    # eps2 = 5e-3
    scale_factor = 1 / jax.lax.stop_gradient(model.sizes + eps) / len(model.sizes)
    # scale_factor = jnp.ones_like(scale_factor)
    size_loss = jnp.sum(jnp.square(model.sizes - pi) * scale_factor)
    mean_loss = jnp.sum(jnp.square(model.means.weight - mu)  * scale_factor[:,None])
    variance_loss = jnp.sum(jnp.square(model.variances - sigma) * scale_factor[:,None,None])
    
    return reconstruction_loss + size_loss + mean_loss + variance_loss - jnp.sum(jnp.exp(log_probs)) * 0, (reconstruction_loss, size_loss, mean_loss, variance_loss)