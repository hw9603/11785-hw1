"""
Follow the instructions provided in the writeup to completely
implement the class specifications for a basic MLP, optimizer, .
You will be able to test each section individually by submitting
to autolab after implementing what is required for that section
-- do not worry if some methods required are not implemented yet.

Notes:

The __call__ method is a special reserved method in
python that defines the behaviour of an object when it is
used as a function. For example, take the Linear activation
function whose implementation has been provided.

# >>> activation = Identity()
# >>> activation(3)
# 3
# >>> activation.forward(3)
# 3
"""

# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)
import numpy as np
import os


class Activation(object):

    """
    Interface for activation functions (non-linearities).

    In all implementations, the state attribute must contain the result, i.e. the output of forward (it will be tested).
    """

    # No additional work is needed for this class, as it acts like an abstract base class for the others

    def __init__(self):
        self.state = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class Identity(Activation):

    """
    Identity function (already implemented).
    """

    # This class is a gimme as it is already implemented for you as an example

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        self.state = x
        return x

    def derivative(self):
        return 1.0


class Sigmoid(Activation):

    """
    Sigmoid non-linearity
    """

    # Remember do not change the function signatures as those are needed to stay the same for AL

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        # Might we need to store something before returning?
        self.state = 1 / (1 + np.exp(-x))
        return self.state

    def derivative(self):
        # Maybe something we need later in here...
        return self.state * (1 - self.state)


class Tanh(Activation):

    """
    Tanh non-linearity
    """

    # This one's all you!

    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        self.state = np.tanh(x)
        return self.state

    def derivative(self):
        return 1 - np.power(self.state, 2)


class ReLU(Activation):

    """
    ReLU non-linearity
    """

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        self.state = np.maximum(x, 0)
        return self.state

    def derivative(self):
        return np.where(self.state != 0, 1.0, 0.0)

# Ok now things get decidedly more interesting. The following Criterion class
# will be used again as the basis for a number of loss functions (which are in the
# form of classes so that they can be exchanged easily (it's how PyTorch and other
# ML libraries do it))


class Criterion(object):

    """
    Interface for loss functions.
    """

    # Nothing needs done to this class, it's used by the following Criterion classes

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class SoftmaxCrossEntropy(Criterion):

    """
    Softmax loss
    """

    def __init__(self):

        super(SoftmaxCrossEntropy, self).__init__()
        self.sm = None

    def forward(self, x, y):

        self.logits = x
        self.labels = y
        # Solution 1 (optimal): Using LogSumExp trick
        M = np.tile(np.max(x, axis=1).reshape(-1, 1), 10)
        logsumexp = M + np.log(np.sum(np.exp(x - M), axis=1)).reshape(-1, 1)
        self.sm = x - logsumexp  # log of softmax
        self.loss = -np.sum(np.multiply(y, self.sm), axis=1)

        # Solution 2 (DANGER!): Not using LogSumExp trick
        # self.sm = np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1, 1)
        # self.loss = -np.sum(np.multiply(y, np.log(self.sm)), axis=1)
        return self.loss

    def derivative(self):

        return np.exp(self.sm) - self.labels


class BatchNorm(object):

    def __init__(self, fan_in, alpha=0.9):

        # You shouldn't need to edit anything in init

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested
        self.var = np.ones((1, fan_in))
        self.mean = np.zeros((1, fan_in))

        self.gamma = np.ones((1, fan_in))
        self.dgamma = np.zeros((1, fan_in))

        self.beta = np.zeros((1, fan_in))
        self.dbeta = np.zeros((1, fan_in))

        # inference parameters
        self.running_mean = np.zeros((1, fan_in))
        self.running_var = np.ones((1, fan_in))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):
        self.x = x
        if eval:
            self.norm = (self.x - self.running_mean) / np.sqrt(self.running_var + self.eps)
        else:
            self.mean = np.mean(self.x, axis=0)
            self.var = np.var(self.x, axis=0)
            self.norm = (self.x - self.mean) / np.sqrt(self.var + self.eps)
            # update running batch statistics
            self.running_mean = self.alpha * self.running_mean + (1 - self.alpha) * self.mean
            self.running_var = self.alpha * self.running_var + (1 - self.alpha) * self.var

        self.out = self.gamma * self.norm + self.beta
        return self.out

    def backward(self, delta):
        dLdnorm = delta * self.gamma
        self.dbeta = np.sum(delta, axis=0)
        self.dgamma = np.sum(delta * self.norm, axis=0)

        dnormdvar = -0.5 * (self.x - self.mean) * np.power(self.var + self.eps, -1.5)
        dLdvar = np.sum(dLdnorm * dnormdvar, axis=0)
        m = len(self.x)
        dLdmean = -np.sum(dLdnorm * np.power(self.var + self.eps, -0.5), axis=0) \
                  - 2 / m * dLdvar * np.sum(self.x - self.mean, axis=0)
        dLdx = dLdnorm * np.power(self.var + self.eps, -0.5) + dLdvar * 2 / m * (self.x - self.mean) + dLdmean / m
        return dLdx


# These are both easy one-liners, don't over-think them
def random_normal_weight_init(d0, d1):
    return np.random.normal(0, 1, (d0, d1))


def zeros_bias_init(d):
    return np.zeros(d)


class MLP(object):

    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn, bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0):

        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly
        self.W = []
        self.b = []
        self.dW = None
        self.db = None
        self.dy = None
        if self.nlayers == 1:
            self.W.append(weight_init_fn(input_size, output_size))  # (784, 10) (input_size, output_size)
            self.b.append(bias_init_fn(output_size))  # (10, )
        else:
            for i in range(self.nlayers):
                if i == 0:
                    self.W.append(weight_init_fn(input_size, hiddens[0]))
                    self.b.append(bias_init_fn(hiddens[0]))
                elif i == self.nlayers - 1:
                    self.W.append(weight_init_fn(hiddens[-1], output_size))
                    self.b.append(bias_init_fn(output_size))
                else:
                    self.W.append(weight_init_fn(hiddens[i - 1], hiddens[i]))
                    self.b.append(bias_init_fn(hiddens[i]))

        # HINT: self.foo = [ bar(???) for ?? in ? ]
        self.deltaW = [w - w for w in self.W]
        self.deltab = [b - b for b in self.b]

        # if batch norm, add batch norm parameters
        if self.bn:
            self.bn_layers = []
            for i in range(self.num_bn_layers):
                self.bn_layers.append(BatchNorm(hiddens[i]))

        # Feel free to add any other attributes useful to your implementation (input, output, ...)
        self.input = None
        self.output = None
        self.batch_size = None

    def forward(self, x):
        self.input = x
        self.batch_size = x.shape[0]
        # print(x.shape)  # (20, 784) (batch_size, input_size)
        # (20, 10) (batch_size, output_size) + (20, 10)
        self.neuron_outputs = []
        self.neuron_outputs.append(x)
        for i in range(self.nlayers):
            input = self.neuron_outputs[i]
            z = np.matmul(input, self.W[i]) + np.tile(self.b[i].reshape(1, -1), (input.shape[0], 1))
            # add batch norm
            if i < self.num_bn_layers:
                z = self.bn_layers[i].forward(z, not self.train_mode)
            output = self.activations[i](z)
            self.neuron_outputs.append(output)
        return self.neuron_outputs[-1]

    def zero_grads(self):
        self.dW = None
        self.db = None
        self.dy = None

    def step(self):
        for i in range(self.nlayers):
            self.deltaW[i] = self.deltaW[i] * self.momentum - self.lr * self.dW[i]
            self.deltab[i] = self.deltab[i].flatten() * self.momentum - self.lr * self.db[i]
            self.W[i] = self.W[i] + self.deltaW[i]
            self.b[i] = self.b[i] + self.deltab[i]
        if self.bn:
            for i in range(self.num_bn_layers):
                self.bn_layers[i].gamma = self.bn_layers[i].gamma - self.lr * self.bn_layers[i].dgamma
                self.bn_layers[i].beta = self.bn_layers[i].beta - self.lr * self.bn_layers[i].dbeta

    def backward(self, labels):
        loss = self.criterion(self.neuron_outputs[-1], labels).mean()
        error = np.sum(self.neuron_outputs[-1].argmax(axis=1) != labels.argmax(axis=1)) / len(self.input)
        self.dW = []
        self.db = []
        self.dy = []
        self.dy.insert(0, 1)  # insert to the front
        for i in range(self.nlayers, 0, -1):
            if i == self.nlayers:
                dErrdZ = self.criterion.derivative() * self.dy[0]
            else:
                if i - 1 < self.num_bn_layers:
                    dErrdZ = self.bn_layers[i - 1].backward(self.activations[i - 1].derivative() * self.dy[0])
                else:
                    dErrdZ = self.activations[i - 1].derivative() * self.dy[0]
            self.dW.insert(0, np.matmul(self.neuron_outputs[i - 1].T, dErrdZ) / self.batch_size)
            self.db.insert(0, np.sum(dErrdZ, axis=0) / self.batch_size)
            self.dy.insert(0, np.matmul(dErrdZ, self.W[i - 1].T))
        return loss, error

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False


def get_training_stats(mlp, dset, nepochs, batch_size):
    train, val, test = dset
    trainx, trainy = train
    valx, valy = val
    testx, testy = test

    idxs = np.arange(len(trainx))

    training_losses = []
    training_errors = []
    validation_losses = []
    validation_errors = []

    # Setup ...

    for e in range(nepochs):
        print("Epoch %d" % e)
        # shuffle the data
        train_indices = np.arange(trainx.shape[0])
        val_indices = np.arange(valx.shape[0])
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        trainx = trainx[train_indices]
        trainy = trainy[train_indices]
        valx = valx[val_indices]
        valy = valy[val_indices]

        train_loss = 0
        train_error = 0
        mlp.train()
        for b in range(0, len(trainx), batch_size):
            # Train ...
            inputs = trainx[b:batch_size + b]
            labels = trainy[b:batch_size + b]
            mlp.zero_grads()
            mlp.forward(inputs)
            loss, error = mlp.backward(labels)
            train_loss += loss
            train_error += error
            mlp.step()
        train_loss /= (len(trainx) / batch_size)
        train_error /= (len(trainx) / batch_size)

        mlp.eval()
        val_loss = 0
        val_error = 0
        for b in range(0, len(valx), batch_size):
            # Val ...
            inputs = valx[b:batch_size + b]
            labels = valy[b:batch_size + b]
            mlp.zero_grads()
            mlp.forward(inputs)
            loss, error = mlp.backward(labels)
            val_loss += loss
            val_error += error
            mlp.step()
        val_loss /= (len(valx) / batch_size)
        val_error /= (len(valx) / batch_size)

        training_losses.append(train_loss)
        training_errors.append(train_error)
        validation_losses.append(val_loss)
        validation_errors.append(val_error)

    return training_losses, training_errors, validation_losses, validation_errors
