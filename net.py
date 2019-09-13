import random
import math

def random_matrix(a, b):
    return [[random.uniform(-0.5, 0.5) for _ in range(a)] for _ in range(b)]

def sigmoid(a): return 1 / (1 + math.exp(-a))

def activation(output, reliable_limit, unknown_value=None):
    is_output_unknown = reliable_limit < output < 1 - reliable_limit
    res_output = unknown_value if is_output_unknown else round(output)
    return res_output

def count_neuron(weights, layer):
    return sigmoid(sum([weight * data for weight, data in zip(weights, layer)]))

def count_layer(weights_layer, layer):
    return [count_neuron(weights, layer) for weights in weights_layer]

def create_network(layers_size):
    return [random_matrix(prev_layer, layer)
            for prev_layer, layer in zip(layers_size[:-1], layers_size[1:])]

def predict(weights, layer, reliable_limit):
    for weight_layer in weights: layer = count_layer(weight_layer, layer)
    return [activation(output, reliable_limit) for output in layer]


def train(weights, datasets, iterations, learning_rate):
    for _ in range(iterations):
        for input_data, expected_data in datasets:
            correct_weights(weights, input_data, expected_data, learning_rate)

def correct_weights(weights, layer, expected_data, learning_rate):
    outputs = []
    for weight_layer in weights:
        outputs.append(layer)
        layer = count_layer(weight_layer, layer)

    errors = [output - expected for output, expected in zip(layer, expected_data)]
    for layer_weights in reversed(weights):
        weight_deltas = [error * output * (1 - output) for error, output in zip(errors, layer)]
        layer = outputs.pop()
        errors = [0] * len(layer)
        for parent, weight_delta in zip(layer_weights, weight_deltas):
            for child_num, output_data in enumerate(layer):
                parent[child_num] -= weight_delta * output_data * learning_rate
                errors[child_num] += weight_delta * parent[child_num]
