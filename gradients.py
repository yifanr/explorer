from network import VQVAE, AE, GMMVAE, DGMMVAE
import equinox as eqx
import jax
import jax.numpy as jnp
import distrax

@eqx.filter_value_and_grad
def update_VQ(VQ: VQVAE, input):
    # input: array of shape (n, d), where d is the size of expected inputs.

    encodings = eqx.filter_vmap(VQ.encode)(input)
    quantized, _ = eqx.filter_vmap(VQ.quantize)(encodings)
    reconstructions = eqx.filter_vmap(VQ.decode)(quantized)

    e_latent_loss = jnp.mean((jax.lax.stop_gradient(quantized) - encodings) ** 2)
    q_latent_loss = jnp.mean((jax.lax.stop_gradient(encodings) - quantized) ** 2)
    reconstruction_loss = jnp.mean((reconstructions - input) ** 2)
    
    return reconstruction_loss + e_latent_loss + q_latent_loss * 0.9


@eqx.filter_value_and_grad
def update_AE(AE: AE, input):
    # input: array of shape (n, d), where d is the size of expected inputs.

    encodings = eqx.filter_vmap(AE.encode)(input)
    reconstructions = eqx.filter_vmap(AE.decode)(encodings)

    reconstruction_loss = jnp.mean((reconstructions - input) ** 2)
    
    return reconstruction_loss

# def stable_mse(x, y):
#     return 

@eqx.filter_value_and_grad
def update_GMM(model: GMMVAE, input, key):
    # input: array of shape (n, d), where d is the size of expected inputs.

    subkeys = jax.random.split(key, input.shape[0])

    encodings = eqx.filter_vmap(model.encode)(input)
    quantized, log_probs, _ = eqx.filter_vmap(model.quantize)(encodings, subkeys)
    # print(jnp.count_nonzero(jnp.isnan(log_probs)))
    reconstructions = eqx.filter_vmap(model.decode)(quantized)

    # print(jnp.max(log_probs))

    # e_latent_loss = jnp.mean((jax.lax.stop_gradient(quantized) - encodings) ** 2)
    # q_latent_loss = jnp.mean((jax.lax.stop_gradient(encodings) - quantized) ** 2)
    reconstruction_loss = jnp.mean((reconstructions - input) ** 2)
    # print("_______________Testing______________________")
    eps = 1e-5
    r = jax.nn.softmax(log_probs, axis=1) + eps
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
    scale_factor = 1 / jax.lax.stop_gradient(model.sizes + eps)
    # scale_factor = jnp.ones_like(model.sizes)
    size_loss = jnp.sum(jnp.square(model.sizes - pi) * scale_factor)
    mean_loss = jnp.sum(jnp.square(model.means.weight - mu)  * scale_factor[:,None])
    covariance_loss = jnp.sum(jnp.square(model.covariances - sigma) * scale_factor[:,None,None])
    
    return reconstruction_loss + size_loss + mean_loss + covariance_loss - jnp.sum(jnp.exp(log_probs)) * 1


@eqx.filter_value_and_grad(has_aux=True)
def update_DGMM(model: DGMMVAE, input, key):
    # input: array of shape (n, d), where d is the size of expected inputs.

    subkeys = jax.random.split(key, input.shape[0])

    encodings = eqx.filter_vmap(model.encode)(input)
    quantized, log_probs, _ = eqx.filter_vmap(model.quantize)(encodings, subkeys)
    reconstructions = eqx.filter_vmap(model.decode)(quantized)

    reconstruction_loss = jnp.mean((reconstructions - input) ** 2)

    eps = 1e-9
    r = jax.nn.softmax(log_probs, axis=1) + eps
    m = jnp.sum(r, axis=0)
    pi = m / jnp.sum(m)
    mu = (1 / m[:,None]) * jnp.sum(r[:,:,None] * encodings[:,None,:], axis=0)
    distances = encodings[:,None,:] - model.means.weight[None,:,:]
    sigma = (1 / m[:,None,None]) * jnp.sum(r[:,:,None] * (distances ** 2), axis=0)
    # print(jnp.min(sigma))
    # print(jnp.max(sigma))

    eps2 = 5e-3
    scale_factor = 1 / jax.lax.stop_gradient(model.sizes + eps2)
    size_loss = jnp.sum(jnp.square(model.sizes - pi) * scale_factor)
    mean_loss = jnp.sum(jnp.square(model.means.weight - mu)  * scale_factor[:,None])
    variance_loss = jnp.sum(jnp.square(model.variances - sigma) * scale_factor[:,None,None])
    
    return reconstruction_loss + size_loss + mean_loss + variance_loss - jnp.sum(jnp.exp(log_probs)) * 1, (reconstruction_loss, size_loss, mean_loss, variance_loss)