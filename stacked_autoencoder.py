import jax
import jax.numpy as jnp
import optax  # https://github.com/deepmind/optax
import torch  # https://pytorch.org
import torchvision  # https://pytorch.org
from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping
from typing import List

import equinox as eqx
import distrax
from matplotlib import pyplot as plt


# Hyperparameters

BATCH_SIZE = 64
LEARNING_RATE = 3e-4
STEPS = 30000
PRINT_EVERY = 1000
SEED = 5678

key = jax.random.PRNGKey(SEED)

class Node(eqx.Module):
    encoder: list
    decoder: list

    def encode(self, x, key):
        for layer in self.encoder:
            x = layer(x)
        return x

    def decode(self, x):
        for layer in self.decoder:
            x = layer(x)
        return x
        
class ConvNode(Node):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, key):
        key1, key2 = jax.random.split(key)
        self.encoder = [eqx.nn.Conv2d(in_channels, out_channels, kernel_size, stride, key=key1)]
        self.decoder = [eqx.nn.ConvTranspose2d(out_channels, in_channels, kernel_size, stride, key=key2)]
    
class MaxPoolConvNode(ConvNode):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, pool_size, pool_stride, key):
        super().__init__(in_channels, out_channels, kernel_size, stride, key)
        self.encoder.append(eqx.nn.MaxPool2d(pool_size, pool_stride))

class DiscreteMaxPoolConvNode(MaxPoolConvNode):
    discrete_size: int
    num_discrete: int
    
    def __init__(self, in_channels, num_discrete, kernel_size, stride, pool_size, pool_stride, discrete_size, key):
        super().__init__(in_channels, num_discrete*discrete_size, kernel_size, stride, pool_size, pool_stride, key)
        self.discrete_size = discrete_size
        self.num_discrete = num_discrete

    def encode(self, x, key):
        logits = MaxPoolConvNode.encode(self, x)
        old_shape = logits.shape
        new_shape = (self.num_discrete, self.discrete_size) + old_shape[1:]
        logits = jnp.reshape(logits, new_shape)

        probs = jax.nn.softmax(logits, axis=1)

        # Make it so that discrete_size is at the end
        probs = jnp.swapaxes(probs, axis1=1, axis2=-1)

        distribution = distrax.OneHotCategorical(probs=probs)

        sample = distribution.sample(seed=key)

        # Straight-through gradient
        sample += probs - jax.lax.stop_gradient(probs)

        # Swap back
        sample = jnp.swapaxes(sample, axis1=1, axis2=-1)
        sample = jnp.reshape(sample, old_shape)

        return sample
    
class DiscreteConvNode(ConvNode):
    discrete_size: int
    num_discrete: int
    
    def __init__(self, in_channels, num_discrete, discrete_size, kernel_size, stride, key):
        super().__init__(in_channels, num_discrete*discrete_size, kernel_size, stride, key)
        self.discrete_size = discrete_size
        self.num_discrete = num_discrete

    def encode(self, x, key):
        key, subkey = jax.random.split(key)
        logits = ConvNode.encode(self, x, subkey)
        old_shape = logits.shape
        new_shape = (self.num_discrete, self.discrete_size) + old_shape[1:]
        logits = jnp.reshape(logits, new_shape)
        probs = jax.nn.softmax(logits, axis=1)
        indices = jnp.argmax(logits, axis=1)
        onehots = jax.nn.one_hot(indices, self.discrete_size, axis=1)


        # Straight-through gradient
        onehots += probs - jax.lax.stop_gradient(probs)

        onehots = jnp.reshape(onehots, old_shape)

        return onehots
    
    # def encode(self, x, key):
    #     key, subkey = jax.random.split(key)
    #     logits = ConvNode.encode(self, x, subkey)
    #     old_shape = logits.shape
    #     new_shape = (self.num_discrete, self.discrete_size) + old_shape[1:]
    #     logits = jnp.reshape(logits, new_shape)

    #     probs = jax.nn.softmax(logits, axis=1)

    #     # Make it so that discrete_size is at the end
    #     probs = jnp.swapaxes(probs, axis1=1, axis2=-1)

    #     distribution = distrax.OneHotCategorical(probs=probs)

    #     sample = distribution.sample(seed=key)

    #     # Straight-through gradient
    #     sample += probs - jax.lax.stop_gradient(probs)

    #     # Swap back
    #     sample = jnp.swapaxes(sample, axis1=1, axis2=-1)
    #     sample = jnp.reshape(sample, old_shape)

    #     return sample

class TwohotLayer(eqx.Module):
    size: int
    axis: int
    centers: jnp.array
    def __init__(self, size, input_channels, input_axis=0):
        self.size = size
        self.axis = input_axis
        self.centers = jnp.ones((input_channels, self.size)) * jnp.arange(self.size)
    
    def __call__(self, inputs):
        normalized = ((self.size - 1) * jnp.arctan(inputs) / jnp.pi) + ((self.size - 1) / 2)
        normalized = jnp.moveaxis(normalized, self.axis, -1)

        output_shape = normalized.shape + (self.size, )
        #stop grad?
        centers = jnp.ones(output_shape) * jax.lax.stop_gradient(self.centers)
        squared_distances = (centers - jnp.expand_dims(normalized, -1))**2

        #play around with coefficient here
        encodings = jnp.exp(-squared_distances)

        encodings = jnp.moveaxis(encodings, -2, 0)

        encodings = jnp.concatenate(encodings, axis=-1)
        encodings = jnp.moveaxis(encodings, -1, self.axis)

        # concatenate
        return encodings


class TwohotConvNode(ConvNode):
    discrete_size: int
    num_discrete: int
    
    def __init__(self, in_channels, num_discrete, discrete_size, kernel_size, stride, key):
        key1, key2 = jax.random.split(key)
        self.encoder = [eqx.nn.Conv2d(in_channels, num_discrete, kernel_size, stride, key=key1), TwohotLayer(discrete_size, num_discrete)]
        self.decoder = [eqx.nn.ConvTranspose2d(num_discrete*discrete_size, in_channels, kernel_size, stride, key=key2)]
        self.discrete_size = discrete_size
        self.num_discrete = num_discrete

    

class DenseNode(Node):
    
    def __init__(self, input_size, output_size, key):
        x = jnp.ravel(x)
        key1, key2 = jax.random.split(key)
        self.encoder = [eqx.nn.Linear(input_size, output_size, key=key1), jax.nn.elu]
        self.decoder = [eqx.nn.Linear(output_size, input_size, key=key2)]
    
class DiscreteDenseNode(Node):
    discrete_size: int
    num_discrete: int

    def __init__(self, input_size, discrete_size, num_discrete, key):
        key1, key2 = jax.random.split(key)
        self.encoder = [eqx.nn.Linear(input_size, discrete_size*num_discrete, key=key1)]
        self.decoder = [eqx.nn.Linear(discrete_size*num_discrete, input_size, key=key2)]

    def encode(self, x: jnp.array, key):
        x = jnp.ravel(x)
        logits = Node.encode(self, x, key)
        old_shape = logits.shape
        logits = jnp.reshape(self.num_discrete, self.discrete_size)
        probs = jax.nn.softmax(logits, axis=1)
        distribution = distrax. OneHotCategorical(probs=probs)
        sample = distribution.sample(seed=key)
        sample += probs - jax.lax.stop_gradient(probs)
        sample = jnp.reshape(sample, old_shape)
        return sample

    
    
class ThinkerNetwork(eqx.Module):
    nodes: List[Node]
    
    def __init__(self, nodes):
        self.nodes = nodes

    def encode(self, x, key):
        outputs = []
        for node in self.nodes:
            # TODO: stop gradient after appending?
            key, subkey = jax.random.split(key)
            output = node.encode(x, subkey)
            outputs.append(output)
            # comment this or not
            x = jax.lax.stop_gradient(output)
            # x = output

        return outputs

    def decode(self, encodings):
        outputs = []
        for encoding, node in zip(encodings, self.nodes):
            output = node.decode(encoding)
            outputs.append(output)
        
        return outputs
    
    def reconstruct(self, encoding):

        for node in reversed(self.nodes):
            encoding = node.decode(encoding)

        return encoding
    
class DecisionHead(eqx.Module):
    layers: List[eqx.Module]

    def __init__(self, input_size, output_size, key):
        key1, key2, key3 = jax.random.split(key, 3)
        # Standard CNN setup: convolutional layer, followed by flattening,
        # with a small MLP on top.
        self.layers = [
            jnp.ravel,
            eqx.nn.Linear(input_size, 512, key=key1),
            jax.nn.sigmoid,
            eqx.nn.Linear(512, 256, key=key2),
            jax.nn.relu,
            eqx.nn.Linear(256, output_size, key=key3),
            jax.nn.log_softmax,
        ]

    def __call__(self, x: Float[Array, "1 28 28"]) -> Float[Array, "10"]:
        for layer in self.layers:
            x = layer(x)
        return x
    

@eqx.filter_jit
def unsupervised_loss(
    network: ThinkerNetwork, x: Float[Array, "batch 1 28 28"], key
) -> Float[Array, ""]:
    # Our input has the shape (BATCH_SIZE, 1, 28, 28), but our model operations on
    # a single input input image of shape (1, 28, 28).
    #
    # Therefore, we have to use jax.vmap, which in this case maps our model over the
    # leading (batch) axis.
    total_loss = 0
    keys = jax.random.split(key, x.shape[0])
    encodings = eqx.filter_vmap(network.encode)(x, keys)
    decodings = eqx.filter_vmap(network.decode)(encodings)
    actuals = [x] + encodings[:-1]
    for decoding, actual in zip(decodings, actuals):
        total_loss += jnp.sum((decoding - actual) ** 2)
    return total_loss, encodings[-1]

def cross_entropy(
    y: Int[Array, " batch"], pred_y: Float[Array, "batch 10"]
) -> Float[Array, ""]:
    # y are the true targets, and should be integers 0-9.
    # pred_y are the log-softmax'd predictions.
    pred_y = jnp.take_along_axis(pred_y, jnp.expand_dims(y, 1), axis=1)
    return -jnp.mean(pred_y)

@eqx.filter_jit
def supervised_loss(
    model: DecisionHead, x, y: Int[Array, " batch"]
) -> Float[Array, ""]:
    # Our input has the shape (BATCH_SIZE, 1, 28, 28), but our model operations on
    # a single input input image of shape (1, 28, 28).
    #
    # Therefore, we have to use jax.vmap, which in this case maps our model over the
    # leading (batch) axis.
    predictions = jax.vmap(model)(x)
    return cross_entropy(y, predictions)


@eqx.filter_jit
def compute_accuracy(
    network: DecisionHead, x: Float[Array, "batch 1 28 28"], y: Int[Array, " batch"]
) -> Float[Array, ""]:
    """This function takes as input the current model
    and computes the average accuracy on a batch.
    """
    pred_y = jax.vmap(network)(x)
    pred_y = jnp.argmax(pred_y, axis=1)
    return jnp.mean(y == pred_y)

def evaluate(network: ThinkerNetwork, decision_head: DecisionHead, testloader: torch.utils.data.DataLoader, key):
    """This function evaluates the model on the test dataset,
    computing both the average loss and the average accuracy.
    """
    avg_unsupervised_loss = 0
    avg_supervised_loss = 0
    avg_acc = 0
    for x, y in testloader:
        x = x.numpy()
        y = y.numpy()
        # Note that all the JAX operations happen inside `loss` and `compute_accuracy`,
        # and both have JIT wrappers, so this is fast.
        unsupervised_loss_value, x = unsupervised_loss(network, x, key)
        supervised_loss_value = supervised_loss(decision_head, x, y)
        avg_unsupervised_loss += unsupervised_loss_value
        avg_supervised_loss += supervised_loss_value
        avg_acc += compute_accuracy(decision_head, x, y)
    return avg_unsupervised_loss / len(testloader), avg_supervised_loss / len(testloader), avg_acc / len(testloader)



def train(
    network: ThinkerNetwork,
    decision_head: DecisionHead,
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    optim: optax.GradientTransformation,
    steps: int,
    print_every: int,
    key
) -> ThinkerNetwork:
    # Just like earlier: It only makes sense to train the arrays in our model,
    # so filter out everything else.
    network_opt_state = optim.init(eqx.filter(network, eqx.is_array))
    head_opt_state = optim.init(eqx.filter(decision_head, eqx.is_array))

    # Always wrap everything -- computing gradients, running the optimiser, updating
    # the model -- into a single JIT region. This ensures things run as fast as
    # possible.
    @eqx.filter_jit
    def make_step(
        network: ThinkerNetwork,
        network_opt_state: PyTree,
        decision_head: DecisionHead,
        head_opt_state: PyTree,
        x: Float[Array, "batch 1 28 28"],
        y: Int[Array, " batch"],
        seed
    ):
        (unsupervised_loss_value, x), network_grads = eqx.filter_value_and_grad(unsupervised_loss, has_aux=True)(network, x, seed)
        supervised_loss_value, head_grads = eqx.filter_value_and_grad(supervised_loss)(decision_head, x, y)
        network_updates, network_opt_state = optim.update(network_grads, network_opt_state, network)
        head_updates, head_opt_state = optim.update(head_grads, head_opt_state, decision_head)
        network = eqx.apply_updates(network, network_updates)
        decision_head = eqx.apply_updates(decision_head, head_updates)
        return network, network_opt_state, decision_head, head_opt_state, unsupervised_loss_value, supervised_loss_value

    # Loop over our training dataset as many times as we need.
    def infinite_trainloader():
        while True:
            yield from trainloader

    for step, (x, y) in zip(range(steps), infinite_trainloader()):
        # PyTorch dataloaders give PyTorch tensors by default,
        # so convert them to NumPy arrays.
        x = x.numpy()
        y = y.numpy()
        key, subkey = jax.random.split(key)
        network, network_opt_state, decision_head, head_opt_state, unsupervised_loss_value, supervised_loss_value = make_step(network, network_opt_state, decision_head, head_opt_state, x, y, subkey)
        if (step % print_every) == 0 or (step == steps - 1):
            key, subkey = jax.random.split(key)
            test_unsupervised_loss, test_supervised_loss, test_accuracy = evaluate(network, decision_head, testloader, subkey)
            print(
                f"{step=}, train_unsupervised_loss={unsupervised_loss_value.item()}, train_supervised_loss={supervised_loss_value.item()}"
                f"test_unsupervised_loss={test_unsupervised_loss.item()}, test_supervised_loss={test_supervised_loss.item()}, test_accuracy={test_accuracy.item()}"
            )
            for i in range(10):
                key, subkey = jax.random.split(key, 2)
                pixels = x[i].reshape((28, 28))
                plt.imshow(pixels, cmap='gray')
                plt.show()
                encodings = network.encode(x[i], subkey)
                reconstruction = network.reconstruct(encodings[-1])
                pixels = reconstruction.reshape((28, 28))
                plt.imshow(pixels, cmap='gray')
                plt.show()
    return network, decision_head

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
# train_dataset = torchvision.datasets.CIFAR100(
#     "CIFAR100",
#     train=True,
#     download=True,
#     transform=normalise_data,
# )
# test_dataset = torchvision.datasets.CIFAR100(
#     "CIFAR100",
#     train=False,
#     download=True,
#     transform=normalise_data,
# )
trainloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
testloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=True
)

key, subkey1, subkey2, subkey3, subkey4, subkey5, subkey6 = jax.random.split(key, 7)
optim = optax.adamw(LEARNING_RATE)
layers = []
# layers.append(ConvNode(in_channels=3, out_channels=32, kernel_size=3, stride=1, key=subkey1))
# layers.append(ConvNode(in_channels=32, out_channels=64, kernel_size=4, stride=2, key=subkey2))
# layers.append(ConvNode(in_channels=64, out_channels=128, kernel_size=3, stride=1, key=subkey3))
# layers.append(ConvNode(in_channels=128, out_channels=256, kernel_size=4, stride=2, key=subkey4))
# layers.append(ConvNode(in_channels=256, out_channels=512, kernel_size=3, stride=1, key=subkey5))

layers.append(ConvNode(in_channels=1, out_channels=4, kernel_size=3, stride=1, key=subkey1))
layers.append(ConvNode(in_channels=4, out_channels=8, kernel_size=4, stride=2, key=subkey2))
layers.append(ConvNode(in_channels=8, out_channels=8, kernel_size=3, stride=1, key=subkey3))
layers.append(ConvNode(in_channels=8, out_channels=16, kernel_size=4, stride=2, key=subkey4))
layers.append(ConvNode(in_channels=16, out_channels=16, kernel_size=3, stride=1, key=subkey5))

# layers.append(TwohotConvNode(in_channels=1, num_discrete=8, discrete_size=10, kernel_size=3, stride=1, key=subkey1))
# layers.append(TwohotConvNode(in_channels=8*10, num_discrete=16, discrete_size=10, kernel_size=4, stride=2, key=subkey2))
# layers.append(TwohotConvNode(in_channels=16*10, num_discrete=16, discrete_size=10, kernel_size=3, stride=1, key=subkey3))
# layers.append(TwohotConvNode(in_channels=16*10, num_discrete=32, discrete_size=10, kernel_size=4, stride=2, key=subkey4))
# layers.append(TwohotConvNode(in_channels=32*10, num_discrete=32, discrete_size=10, kernel_size=3, stride=1, key=subkey5))

unsupervised_network = ThinkerNetwork(layers)
# decision_head = DecisionHead(3*3*512, 100, subkey6)
# decision_head = DecisionHead(2*2*10*32, 100, subkey6)
decision_head = DecisionHead(2*2*16, 100, subkey6)

train(unsupervised_network, decision_head, trainloader, testloader, optim, STEPS, PRINT_EVERY, key)
