import numpy as np

class Net:
    def __init__(self, network_struct, learning_rate=0.1):
        self.weights = []
        self.weight_layers_num = len(network_struct) - 1;
        for layer_num in range(self.weight_layers_num):
            layer_index = tuple(network_struct[layer_num:layer_num + 2])
            layer = np.random.normal(0.0, 1, layer_index)
            self.weights.append(layer)
        self.sigmoid_mapper = np.vectorize(self.sigmoid)
        self.learning_rate = np.array([learning_rate])

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, inputs):
        layer = inputs
        for weight_layer in reversed(self.weights):
            layer = np.dot(weight_layer, layer)
            layer = self.sigmoid_mapper(layer)
        return layer 

    def train(self, inputs, expected):
        layer = inputs
        outputs = [layer]
        for weight_layer in reversed(self.weights):
            layer = np.dot(weight_layer, layer)
            layer = self.sigmoid_mapper(layer)
            outputs.append(layer)
        
        output = outputs.pop()
        errors = [o - e for o,e in zip(output, expected)]
        for num in range(self.weight_layers_num):
            weight_deltas = [e * o * (1 - o) for e,o in zip(errors, output)]
            output = outputs.pop()
            errors = [0] * len(output)
            for parent_i in range(len(self.weights[num])):
                for child_i in range(len(self.weights[num][parent_i])):
                    self.weights[num][parent_i][child_i] -= output[child_i] * weight_deltas[parent_i] * self.learning_rate
                    errors[child_i] += self.weights[num][parent_i][child_i] * weight_deltas[parent_i]
            

train = [
        ([0, 0, 1], [1]),
        ([0, 1, 0], [0]),
        ([0, 1, 1], [0]),
        ([1, 0, 0], [1]),
        ([1, 0, 1], [1]),
        ([1, 1, 0], [0]),
        ([1, 1, 1], [1]),
        ([0, 0, 0], [0])
        ]

network = Net([1, 2, 3], learning_rate = 0.05)

for e in range(5000):
   for input_stat, correct_predict in train:
        network.train(np.array(input_stat), correct_predict)

for input_stat, correct_predict in train:
    print('{}, {}, {}'.format(str(input_stat), str(network.predict(np.array(input_stat)) > .5), str(correct_predict[0] == 1)))

