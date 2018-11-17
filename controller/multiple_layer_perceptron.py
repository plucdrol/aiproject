import numpy as np


def sigmoid(x):
    """ Sigmoid like function using tanh """
    return np.tanh(x)


def dsigmoid(x):
    """ Derivative of sigmoid above """
    return 1.0-x**2


class MultiLayerPerceptron:
    """ Multi-layer perceptron class. """

    def __init__(self, *args):
        """ Initialization of the perceptron with given sizes.  """

        self.shape = args
        n = len(args)

        # Build layers
        self.layers = []
        # Input layer (+1 unit for bias)
        self.layers.append(np.ones(self.shape[0]+1))
        # Hidden layer(s) + output layer
        for i in range(1, n):
            self.layers.append(np.ones(self.shape[i]))

        # Build weights matrix (randomly between -0.25 and +0.25)
        self.weights = []
        for i in range(n-1):
            self.weights.append(np.zeros((self.layers[i].size,
                                         self.layers[i+1].size)))

        # dw will hold last change in weights (for momentum)
        self.dw = [0, ]*len(self.weights)

        # Reset weights
        self.reset()

    def reset(self):
        """ Reset weights """

        for i in range(len(self.weights)):
            z = np.random.random((self.layers[i].size, self.layers[i+1].size))
            self.weights[i][...] = (2*z-1)*0.05

    def propagate_forward(self, data):
        """ Propagate data from input layer to output layer. """

        # Set input layer
        self.layers[0][0:-1] = data

        # Propagate from layer 0 to layer n-1 using sigmoid as activation function
        for i in range(1, len(self.shape)):
            # Propagate activity
            self.layers[i][...] = sigmoid(np.dot(self.layers[i-1], self.weights[i-1]))

        # Return output
        return self.layers[-1]

    def propagate_backward(self, target, lrate=0.1, momentum=0.1):
        """ Back propagate error related to target using lrate. """

        deltas = []

        # Compute error on output layer
        error = target - self.layers[-1]
        delta = error*dsigmoid(self.layers[-1])
        deltas.append(delta)

        # Compute error on hidden layers
        for i in range(len(self.shape)-2, 0, -1):
            delta = np.dot(deltas[0], self.weights[i].T)*dsigmoid(self.layers[i])
            deltas.insert(0, delta)

        # Update weights
        for i in range(len(self.weights)):
            layer = np.atleast_2d(self.layers[i])
            delta = np.atleast_2d(deltas[i])
            dw = np.dot(layer.T, delta)
            self.weights[i] += lrate*dw + momentum*self.dw[i]
            self.dw[i] = dw

        # Return error
        return (error**2).sum()

    def learn_testset(self, samples, training_set_ratio=0.75, threshold=0.0001, lrate=.1, momentum=0.1):
        # Split samples into two sets for training and testing
        training_set_index = int(training_set_ratio * samples.size)
        randomized_samples = np.split(np.random.permutation(samples), [training_set_index, samples.size])
        training_samples = randomized_samples[0]
        test_samples = randomized_samples[1]

        # Train
        mean_squared_error, nb_iterations = self._train(training_samples, threshold, lrate, momentum)

        # Test
        outputs = self._test(test_samples)

        return self._results(test_samples, outputs, mean_squared_error, nb_iterations)

    def learn_kfold(self, samples, k, threshold=0.0001, lrate=.1, momentum=0.1):
        # Split samples into k sets
        randomized_samples = np.array_split(np.random.permutation(samples), samples.size / k)

        # Each of the k sets is used as the test sample while the k-1 other sets are the training sample
        mean_squared_errors = []
        for i in range(k):
            test_sample = randomized_samples[i]

            training_samples = None
            for j in range(k):
                if not np.array_equal(randomized_samples[j], test_sample):
                    if training_samples is None:
                        training_samples = randomized_samples[j]
                    else:
                        training_samples = np.concatenate((training_samples, randomized_samples[j]))

            mean_squared_error, nb_iterations = self._train(training_samples, threshold, lrate, momentum)

            outputs = self._test(test_sample)
            mean_squared_errors.append(self._mse_outputs(test_sample, outputs))

        return sum(mean_squared_errors) / len(mean_squared_errors)

    def _mse_outputs(self, sample, outputs):
        mean_squared_errors = []
        for x in range(len(outputs)):
            error = sample['output'][x] - outputs[x]
            squared_error = error**2
            mean_squared_errors.append(sum(squared_error) / len(outputs[x]))
        return sum(mean_squared_errors) / len(mean_squared_errors)

    def _train(self, training_samples, threshold, lrate, momentum):
        mean_squared_error = 10000
        nb_iterations = 0
        while True:
            nb_iterations += 1
            mse_learn = []
            for i in range(training_samples.size):
                self.propagate_forward(training_samples['input'][i])
                mse_learn.append(self.propagate_backward(training_samples['output'][i], lrate, momentum) /
                                 self.layers[-1].size)
            mse_learn_mean = sum(mse_learn) / len(mse_learn)

            if abs(mean_squared_error - mse_learn_mean) > threshold:
                mean_squared_error = mse_learn_mean
            else:
                mean_squared_error = mse_learn_mean
                break
        return mean_squared_error, nb_iterations

    def _test(self, test_samples):
        outputs = []
        for i in range(test_samples.size):
            outputs.append(list(self.propagate_forward(test_samples['input'][i])))
        return outputs

    def _results(self, test_samples, outputs, mean_squared_error, nb_iterations):
        results = [[], {'nb_iterations': nb_iterations}]
        for i in range(test_samples.size):
            results[0].append({'input': test_samples['input'][i],
                               'output': outputs[i],
                               'expected': test_samples['output'][i],
                               'error': self._error(test_samples['output'][i], outputs[i])})

        results[1]['error_rate'] = self._error_rate(results[0])
        results[1]['mean_squared_error'] = self._mse_outputs(test_samples, outputs)

        return results

    def _error(self, sample, output):
        return sum([abs(sample[x] - output[x]) for x in range(sample.size)]) / sample.size

    def _error_rate(self, results):
        return sum([x['error'] for x in results]) / len(results)

