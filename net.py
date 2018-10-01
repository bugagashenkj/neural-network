import numpy as np

class Net:
    def __init__(self, network_struct):
        self.weights = []
        self.weight_layers_num = len(network_struct) - 1;
        for layer_num in range(self.weight_layers_num):
            layer_index = tuple(network_struct[layer_num:layer_num + 2])
            layer = np.random.normal(0.0, 1, layer_index)
            self.weights.append(layer)
        self.sigmoid_mapper = np.vectorize(self.sigmoid)
    
    def round(self, weight_layer, layer):
        return self.sigmoid_mapper(np.dot(weight_layer, layer))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, layer):
        for weight_layer in reversed(self.weights):
            layer = self.round(weight_layer, layer)
        return layer 

    def train_round(self, layer, expected_data, learning_rate):
        outputs = [layer]
        for weight_layer in reversed(self.weights):
            layer = self.round(weight_layer, layer) 
            outputs.append(layer)
        
        layer_output = outputs.pop()
        errors = [output - expected for output, expected in zip(layer_output, expected_data)]
        for layer_weights in self.weights:
            weight_deltas = [error * output * (1 - output) for error, output in zip(errors, layer_output)]
            layer_output = outputs.pop()
            errors = [0] * len(layer_output)
            for parent_num, weight_delta in enumerate(weight_deltas):
                for child_num, output_data in enumerate(layer_output):
                    layer_weights[parent_num][child_num] -= output_data * weight_delta * learning_rate
                    errors[child_num] += layer_weights[parent_num][child_num] * weight_delta
            

    def train(self, datasets, iterations, learning_rate):
        for iteration in range(iterations):
            for input_data, expected_data in datasets:
                self.train_round(input_data, expected_data, learning_rate)
                
datasets = [
        ([0, 0, 1], [1]),
        ([0, 1, 0], [0]),
        ([0, 1, 1], [0]),
        ([1, 0, 0], [1]),
        ([1, 0, 1], [1]),
        ([1, 1, 0], [0]),
        ([1, 1, 1], [1]),
        ([0, 0, 0], [0])
        ]

network = Net([1, 2, 3])
network.train(datasets, 5000, 0.05)

for input_data, expected_data in datasets:
    print('{}, {}, {}'.format(str(input_data), str(network.predict(input_data) > .5), str(expected_data[0] == 1)))

