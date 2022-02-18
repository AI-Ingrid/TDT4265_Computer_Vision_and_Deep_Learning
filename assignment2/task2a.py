import numpy as np
import utils
import typing
import copy
np.random.seed(1)


def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


def sigmoid_improved(z):
    """The improved sigmoid function."""
    return 1.7159 * np.tanh(2/3*z)


def sigmoid_improved_prime(z):
    """Derivative of the improved sigmoid function."""
    # 1.17267 * (1/(np.cosh(2/3*z)))**2
    return 1.7259 * 2/3 * (1 - (np.tanh(2/3*z))**2)


def activation_func(z, use_improved):
    if use_improved:
        return sigmoid_improved(z)
    else:
        return sigmoid(z)


def activation_func_prime(z, use_improved):
    if use_improved:
        return sigmoid_improved_prime(z)
    else:
        return sigmoid_prime(z)


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)


def pre_process_images(X: np.ndarray, X_std, X_mean):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] normalized as described in task2a
    """
    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"

    X_norm = (X - X_mean)/X_std

    # Add bias
    x_vector = np.ones((X.shape[0], 1))
    X = np.concatenate((X_norm, x_vector), axis=1)
    return X


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """

    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
    # Task 2 Implementation of one-hot-encode
    cross_entropy_error = - np.sum(targets * np.log(outputs), axis=1)
    return cross_entropy_error.mean()


class SoftmaxModel:

    def __init__(self,
                 neurons_per_layer: typing.List[int],   # Number of neurons per layer
                 use_improved_sigmoid: bool,            # Task 3a hyperparameter
                 use_improved_weight_init: bool         # Task 3c hyperparameter
                 ):
        
        np.random.seed(1)                               # Always reset random seed before weight init to get comparable results.
        self.I = 785                                    # Define number of input nodes
        self.use_improved_sigmoid = use_improved_sigmoid # Initializing z_j for usage in backward after forward prop
        self.z_j = []
        self.outputs = 10                               # Define number of output nodes
        self.use_improved_weight_init = use_improved_weight_init
        self.hidden_layer_output = []

        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer

        # TASK 3A) set weights improved or not
        self.ws = self.set_weights(self.use_improved_weight_init)

        prev = self.I
        for size in self.neurons_per_layer:
            w_shape = (prev, size)
            print("Initializing weight to shape:", w_shape)
            w = np.zeros(w_shape)
            prev = size
        self.grads = [None for i in range(len(self.ws))]

    def set_weights(self, use_improved):
        ws = []
        # Create 'not' improved weights
        if not use_improved:
            # Initialize weights from input nodes
            ws.append(np.random.uniform(-1, 1, (785, self.neurons_per_layer[0])))

            # Initialize weights for the rest of the layers
            for index, layer in enumerate(self.neurons_per_layer):

                # For the last element in neurons_per_layer
                if layer == self.neurons_per_layer[-1]:
                    break
                else:
                    ws.append(np.random.uniform(-1, 1, (layer, self.neurons_per_layer[index+1])))
            return ws

        # Create the improved weights
        else:
            # Initialize weights from input nodes
            ws.append(np.random.normal(0, 1/np.sqrt(785), self.neurons_per_layer[0]))

            # Initialize weights for the rest of the layers
            for index, layer in enumerate(self.neurons_per_layer):

                # For the last element in neurons_per_layer
                if layer == self.neurons_per_layer[-1]:
                    ws.append(np.random.normal(0, 1/np.sqrt(layer), (self.neurons_per_layer[index-1], layer)))
                else:
                    ws.append(np.random.normal(0, 1/np.sqrt(layer), (layer, self.neurons_per_layer[index+1])))
            return ws

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """
        self.z_j = []
        self.hidden_layer_output = []

        # For the first forward pass from the input nodes
        self.hidden_layer_output.append(X)
        # Do forward pass through every layer expect the last layer
        for i in range(len(self.neurons_per_layer)-1):
            w_j = copy.deepcopy(self.ws[i])
            self.z_j.append(self.hidden_layer_output[i] @ w_j)
            self.hidden_layer_output.append(activation_func(self.z_j[i], self.use_improved_sigmoid))

        # Do forward pass for the last layer
        w_k = self.ws[-1]
        z_k = np.dot(self.hidden_layer_output[-1], w_k)
        y_hat = softmax(z_k)
        return y_hat

    def backward(self, X: np.ndarray, outputs: np.ndarray,
                 targets: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad

        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"
        batch_size = X.shape[0]
        # Calculate the difference between prediction and the correct value
        delta = -(targets-outputs)  # delta_k

        # Calculate the gradient for the last weight
        first_grad = (self.hidden_layer_output[-1].T @ delta)
        self.grads[-1] = np.divide(first_grad, batch_size)
        # Do backward pass through the network
        for i in range(len(self.neurons_per_layer)-2, -1, -1):
            z = self.z_j[i]
            z_j_prime = activation_func_prime(z, self.use_improved_sigmoid)

            # Update delta
            delta = (delta @ self.ws[i+1].T) * z_j_prime

            # Update the gradient
            self.grads[i] = (self.hidden_layer_output[i].T @ delta)
            self.grads[i] = np.divide(self.grads[i], batch_size)

        for grad, w in zip(self.grads, self.ws):
            assert grad.shape == w.shape,\
                f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."

    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    # Task 2 Implementation of one-hot-encode
    Y_encoded = np.zeros((Y.shape[0], num_classes), dtype=int)

    # Create arrays for one hot encoding
    Y_all_rows = np.arange(Y.shape[0])
    Y_specified_cols = Y.T # we dont know

    # Set value in the right col for all rows to 1
    Y_encoded[Y_all_rows, Y_specified_cols] = 1
    return Y_encoded


def gradient_approximation_test(
        model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited. 
        Details about this test is given in the appendix in the assignment.
    """
    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**2,\
                    f"Calculated gradient is incorrect. " \
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n" \
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n" \
                    f"If this test fails there could be errors in your cross entropy loss function, " \
                    f"forward function or backward function"


if __name__ == "__main__":
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert Y[0, 3] == 1 and Y.sum() == 1, \
        f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist()
    X_std = np.std(X_train)
    X_mean = np.mean(X_train)
    X_train = pre_process_images(X_train, X_std=X_std, X_mean=X_mean)
    Y_train = one_hot_encode(Y_train, 10)
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 64, 10]
    use_improved_sigmoid = False
    use_improved_weight_init = False
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init)

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)
