import random
import math

def random_matrix(diff, a, b):
    return [[random.uniform(-diff, diff) for _ in range(a)] for _ in range(b)]

def next_round(weight_layer, layer):
    sigmoid = lambda x: 1 / (1 + math.exp(-x))
    array_dot = lambda a, b: [a1 * b1 for a1, b1 in zip(a, b)]
    inputs = [sum(array_dot(weights, layer)) for weights in weight_layer]
    layer_data = [sigmoid(input_data) for input_data in inputs]
    return layer_data 

class Net:
    def __init__(self, network_struct):
        self.weights = []
        prev_layer_size = network_struct.pop(0) 
        for layer_size in network_struct:
            weight_layer = random_matrix(0.5, layer_size, prev_layer_size)
            self.weights.append(weight_layer)
            prev_layer_size = layer_size
    
    def predict(self, layer):
        for weight_layer in reversed(self.weights):
            layer = next_round(weight_layer, layer)
        activation_func = lambda x: round(x)
        return list(map(activation_func, layer))

    def train(self, datasets, iterations, learning_rate):
        for iteration in range(iterations):
            for input_data, expected_data in datasets:
                self.train_round(input_data, expected_data, learning_rate)
    
    def train_round(self, layer, expected_data, learning_rate):
        outputs = []
        for weight_layer in reversed(self.weights):
            outputs.append(layer)
            layer = next_round(weight_layer, layer) 
        
        errors = [output - expected for output, expected in zip(layer, expected_data)]
        for layer_weights in self.weights:
            weight_deltas = [error * output * (1 - output) for error, output in zip(errors, layer)]
            layer = outputs.pop()
            errors = [0] * len(layer)
            for parent, weight_delta in zip(layer_weights, weight_deltas):
                for child_num, output_data in enumerate(layer):
                    parent[child_num] -= weight_delta * output_data * learning_rate
                    errors[child_num] += weight_delta * parent[child_num] 
