import torch  # https://pytorch.org
import torchvision  # https://pytorch.org
from network import GMMVAE
from gradients import update_GMM
import optax
import equinox as eqx
from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping
import jax.numpy as jnp
import jax
import distrax
import jax.tree_util as jtu
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse

BATCH_SIZE = 64
LEARNING_RATE = 3e-4
STEPS = 30000
NUM_CLUSTERS = 12

normalise_data = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
    ]
)
train_dataset = torchvision.datasets.MNIST(
    "MNIST",
    train=True,
    download=True,
    transform=normalise_data,
)
test_dataset = torchvision.datasets.MNIST(
    "MNIST",
    train=False,
    download=True,
    transform=normalise_data,
)
trainloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
testloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=True
)
key = jax.random.PRNGKey(42)
key, subkey = jax.random.split(key, 2)
model = GMMVAE(28*28, 2, NUM_CLUSTERS, subkey)
# model = AE(28*28, 8, jax.random.PRNGKey(42))

filter = jtu.tree_map(lambda _: False, model)
filter = eqx.tree_at(
    lambda tree: (tree.means, tree.sizes, tree.covariances),
    filter,
    replace=(True, True, True),)
# optim = optax.multi_transform(
#     {False: optax.adam(LEARNING_RATE), True: optax.sgd(1e-4)},
#     filter
# )
optim = optax.adam(LEARNING_RATE)

@eqx.filter_jit
def make_step(
    model: GMMVAE,
    opt_state: PyTree,
    x: Float[Array, "batch 1 28 28"],
    key
):
    x = jnp.reshape(x, (x.shape[0], 28*28))
    (loss_value, (reconstruction_loss, size_loss, mean_loss, variance_loss)), grads = update_GMM(model, x, key)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value

opt_state = optim.init(eqx.filter(model, eqx.is_array))

    # Always wrap everything -- computing gradients, running the optimiser, updating
    # the model -- into a single JIT region. This ensures things run as fast as
    # possible.
    

# Loop over our training dataset as many times as we need.
def infinite_trainloader():
    while True:
        yield from trainloader

for step, (x, y) in zip(range(STEPS), infinite_trainloader()):
    key, subkey = jax.random.split(key, 2)
    # PyTorch dataloaders give PyTorch tensors by default,
    # so convert them to NumPy arrays.
    x = x.numpy()
    y = y.numpy()
    model, opt_state, train_loss = make_step(model, opt_state, x, subkey)
    if (step % 200) == 0 or (step == STEPS - 1):
        print(
            f"{step=}, train_loss={train_loss.item()}, "
        )
        # print(jnp.count_nonzero(jnp.isnan(model.means.weight)))
        # distribution = distrax.MultivariateNormalFullCovariance(model.means.weight, model.covariances)
        # print(jnp.count_nonzero(jnp.isnan(distribution.sample(seed=subkey))))
        # print(jnp.min(model.variances))
        # print(jnp.max(model.variances))
        # print(jnp.min(model.means.weight))
        # print(jnp.max(model.means.weight))
        # print(jnp.min(model.sizes))
        # print(jnp.max(model.sizes))
        x = jnp.reshape(x, (x.shape[0], 28*28))
        (loss_value, (reconstruction_loss, size_loss, mean_loss, variance_loss)), grads = update_GMM(model, x, subkey)
        print((reconstruction_loss.item(), size_loss.item(), mean_loss.item(), variance_loss.item()))
        print(jnp.min(jnp.linalg.eigvalsh(model.covariances)))

    if ((step % 2000) == 0 or (step == STEPS - 1)):
        subkeys = jax.random.split(key, x.shape[0])
        encodings = eqx.filter_vmap(model.encode)(x)
        quantized, log_probs, indices = eqx.filter_vmap(model.quantize)(encodings, subkeys)
        colors = cm.rainbow(jnp.linspace(0, 1, NUM_CLUSTERS))

        # ax = plt.subplot(111, aspect='equal')
        eigenvalues, eigenvectors = jnp.linalg.eigh(model.covariances)
        for i in range(NUM_CLUSTERS):
            theta = jnp.linspace(0, 2*jnp.pi, 1000)
            ellipsis = (jnp.sqrt(eigenvalues[i][None,:]) * eigenvectors[i]) @ jnp.asarray([jnp.sin(theta), jnp.cos(theta)])
            # ellipsis *= model.sizes[i] * NUM_CLUSTERS
            ellipsis += model.means.weight[i][:,None]
            plt.plot(ellipsis[0,:], ellipsis[1,:], color=colors[i])
        plt.scatter(encodings[:,0], encodings[:,1], c=colors[indices])
        plt.show()
        for i in range(5):
            key, subkey = jax.random.split(key, 2)
            dummy_x, dummy_y = next(iter(testloader))
            dummy_x = dummy_x[0].numpy()
            pixels = dummy_x.reshape((28, 28))
            plt.imshow(pixels, cmap='gray')
            plt.show()
            encoded = model.encode(dummy_x.reshape(-1))
            quantized, log_prob, index = model.quantize(encoded, subkey)
            reconstruction = model.decode(encoded)
            pixels = reconstruction.reshape((28, 28))
            print(index)
            plt.imshow(pixels, cmap='gray')
            plt.show()



for i in range(20):
    key, subkey = jax.random.split(key, 2)
    dummy_x, dummy_y = next(iter(testloader))
    dummy_x = dummy_x[0].numpy()
    pixels = dummy_x.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()
    encoded = model.encode(dummy_x.reshape(-1))
    quantized, log_prob, index = model.quantize(encoded, subkey)
    reconstruction = model.decode(encoded)
    pixels = reconstruction.reshape((28, 28))
    print(index)
    plt.imshow(pixels, cmap='gray')
    plt.show()