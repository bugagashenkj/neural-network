import random
import math

def random_matrix(diff, a, b):
    return [[random.uniform(-diff, diff) for _ in range(a)] for _ in range(b)]

def sigmoid(a):
    return 1 / (1 + math.exp(-a))

def activation(output, reliable_limit, unknown_value=None):
    is_output_unknown = reliable_limit < output < 1 - reliable_limit 
    res_output = unknown_value if is_output_unknown else round(output)
    return res_output 

def count_neuron(weights, layer):
    new_data = sigmoid(sum([weight * data for weight, data in zip(weights, layer)]))
    return new_data

def count_layer(weights, layer):
    new_layer = [count_neuron(weights, layer) for weights in weights]
    return new_layer

def create_weights(layers_size):
    weights = []
    for prev_layer, layer in zip(layers_size[:-1], layers_size[1:]):
        diff = 0.5
        weights.append(random_matrix(diff, prev_layer, layer)) 
    return weights 

def predict(weights, layer, percent):
    for weight_layer in weights:
        layer = count_layer(weight_layer, layer)
    result_output = [activation(output, percent) for output in layer]
    return result_output 

def train(weights, datasets, iterations, learning_rate):
    for _ in range(iterations):
        for input_data, expected_data in datasets:
            outputs = train_predict(weights, input_data)
            correct_weights(weights, outputs, expected_data, learning_rate)
    
def train_predict(weights, layer):
    outputs = [layer]
    for weight_layer in weights:
        layer = count_layer(weight_layer, layer)
        outputs.append(layer)
    return outputs

def correct_weights(weights, outputs, expected_data, learning_rate):
    layer = outputs.pop()
    errors = [output - expected for output, expected in zip(layer, expected_data)]
    for layer_weights in reversed(weights):
        weight_deltas = [error * output * (1 - output) for error, output in zip(errors, layer)]
        layer = outputs.pop()
        errors = [0] * len(layer)
        for parent, weight_delta in zip(layer_weights, weight_deltas):
            for child_num, output_data in enumerate(layer):
                parent[child_num] -= weight_delta * output_data * learning_rate
                errors[child_num] += weight_delta * parent[child_num] 
