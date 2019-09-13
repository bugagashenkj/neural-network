from net import *

train_datasets = [
        ([0, 0, 1], [1]),
        ([0, 1, 1], [0]),
        ([1, 0, 1], [1]),
        ([1, 1, 1], [1]),
        ([0, 0, 0], [0])
        ]

test_datasets = [
        ([0, 1, 0], [0]),
        ([1, 0, 0], [1]),
        ([1, 1, 0], [0]),
        ]

weights = create_network([3, 2, 1])
train(weights, train_datasets, 15000, 0.05)

unknown_values = 0
max_unknown_values = 1
for inputs, expected in test_datasets:
    result = (predict(weights, inputs, 0.25))
    if result == None and ++unknown_values > max_unknown_values:
        raise Exception('Too many unknown values!')
    elif result != expected:
        raise Exception('Uncorrect value!')
