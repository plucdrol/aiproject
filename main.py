from utility.zoo_sample import ZooSample
from controller.multiple_layer_perceptron import MultiLayerPerceptron


def print_results_testset(exp_nb, results):
    print "\tTestset:"
    print "\t\tError rate:\t\t\t\t%.5f" % results[1]['error_rate']
    print "\t\tMean squared error:\t\t%.5f" % results[1]['mean_squared_error']
    print "\t\tNumber of iterations:\t", results[1]['nb_iterations']


def print_results_kfold(exp_nb, mean_mse):
    print "\tK-Fold:"
    print "\t\tMean squared error: %.5f" % mean_mse


def experiment(exp_nb, zoo_sample, nb_hidden_neurons, training_set_ratio, threshold, lrate, momentum, k):
    print "Experimentation ", exp_nb
    mlp = MultiLayerPerceptron(zoo_sample.nb_inputs, nb_hidden_neurons, zoo_sample.nb_outputs)
    print_results_testset(exp_nb, mlp.learn_testset(zoo_sample.samples(), training_set_ratio, threshold, lrate, momentum))
    mlp.reset()
    print_results_kfold(exp_nb, mlp.learn_kfold(zoo_sample.samples(), k, threshold, lrate, momentum))
    print

zoo_sample = ZooSample()

threshold = 0.0001
k = 10

experiment(1, zoo_sample, 20, 0.75, threshold, 0.15, 0.6, k)
experiment(2, zoo_sample, 20, 0.7, threshold, 0.2, 0.1, k)
experiment(3, zoo_sample, 15, 0.75, threshold, 0.4, 0.01, k)
experiment(4, zoo_sample, 15, 0.7, threshold, 0.08, 0.6, k)
experiment(5, zoo_sample, 10, 0.85, threshold, 0.025, 0.72, k)
experiment(6, zoo_sample, 10, 0.8, threshold, 0.01, 0.5, k)
experiment(7, zoo_sample, 10, 0.75, threshold, 0.4, 0.5, k)
experiment(8, zoo_sample, 10, 0.7, threshold, 0.4, 0.5, k)
experiment(9, zoo_sample, 5, 0.85, threshold, 0.15, 0.65, k)
experiment(10, zoo_sample, 5, 0.8, threshold, 0.2, 0.7, k)
