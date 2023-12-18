import torch  # https://pytorch.org
import torchvision  # https://pytorch.org
from network import VQVAE, CVQVAE, AE
from gradients import update_VQ, update_AE
import optax
import equinox as eqx
from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping
import jax.numpy as jnp
import jax
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse

BATCH_SIZE = 512
LEARNING_RATE = 3e-4
TOTAL_STEPS = 5120000
STEPS = int(TOTAL_STEPS/BATCH_SIZE)
NUM_CLUSTERS = 24
VISUALIZE = False
EMBEDDING_DIM = 8

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
subkey, key = jax.random.split(key)
model = VQVAE(28*28, EMBEDDING_DIM, NUM_CLUSTERS, subkey)
# model = AE(28*28, 8, jax.random.PRNGKey(42))
optim = optax.adamw(LEARNING_RATE)

@eqx.filter_jit
def make_step(
    model: VQVAE,
    opt_state: PyTree,
    x: Float[Array, "batch 1 28 28"],
):
    x = jnp.reshape(x, (x.shape[0], 28*28))
    loss_value, grads = update_VQ(model, x)
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
    # PyTorch dataloaders give PyTorch tensors by default,
    # so convert them to NumPy arrays.
    x = x.numpy()
    y = y.numpy()
    model, opt_state, train_loss = make_step(model, opt_state, x)
    if (step % 200) == 0 or (step == STEPS - 1):
        print(
            f"{step=}, train_loss={train_loss.item()}, "
        )
    if VISUALIZE and ((step % 200) == 0 or (step == STEPS - 1)):
        x = jnp.reshape(x, (x.shape[0], 28*28))
        subkeys = jax.random.split(key, x.shape[0])
        encodings = eqx.filter_vmap(model.encode)(x)
        quantized, indices = eqx.filter_vmap(model.quantize)(encodings)
        colors = cm.rainbow(jnp.linspace(0, 1, 10))
        # colors = cm.rainbow(jnp.linspace(0, 1, NUM_CLUSTERS))

        # ax = plt.subplot(111, aspect='equal')
        # plt.scatter(encodings[:,0], encodings[:,1], c=colors[y], label = y)
        
        #plotting the results:
        u_labels = jnp.unique(y)
        for i in u_labels:
            plt.scatter(encodings[y == i, 0] , encodings[y == i , 1] , label = i)
        plt.scatter(model.embedding[:,0] , model.embedding[:,1] , s = 80, color = 'black')
        plt.legend()
        plt.show()

for i in range(20):
    dummy_x, dummy_y = next(iter(testloader))
    dummy_x = dummy_x[0].numpy()
    pixels = dummy_x.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()
    encoded = model.encode(dummy_x.reshape(-1))
    quantized, index = model.quantize(encoded)
    reconstruction = model.decode(encoded)
    pixels = reconstruction.reshape((28, 28))
    print(index)
    plt.imshow(pixels, cmap='gray')
    plt.show()