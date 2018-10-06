import random
import math

def random_matrix(diff, a, b):
    return [[random.uniform(-diff, diff) for _ in range(a)] for _ in range(b)]

def sigmoid(x): return 1 / (1 + math.exp(-x)) 

def next_round(weight_layer, layer):
    return [sigmoid(sum(weight * data for weight, data in zip(weights, layer)))
            for weights in weight_layer]

def create_weights(network_struct):
    weights = []
    prev_layer_size = network_struct.pop(0) 
    for layer_size in network_struct:
        weight_layer = random_matrix(0.5, prev_layer_size, layer_size)
        weights.append(weight_layer)
        prev_layer_size = layer_size
    return weights
    
def predict(weights, layer, percent):
    for weight_layer in weights:
        layer = next_round(weight_layer, layer)
    activation = lambda output: None if percent < output < 1 - percent else round(output) 
    result = [activation(output) for output in layer]
    return result 

def train(weights, datasets, iterations, learning_rate):
    for iteration in range(iterations):
        for input_data, expected_data in datasets:
            outputs = train_predict(weights, input_data)
            correct_weights(weights, outputs, expected_data, learning_rate)
    
def train_predict(weights, layer):
    outputs = [layer]
    for weight_layer in weights:
        layer = next_round(weight_layer, layer)
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
