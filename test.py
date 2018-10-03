from net import Net

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
network.train(datasets, 3000, 0.05)

for inputs, expected in datasets:
    if (network.predict(inputs) != expected):
        raise Exception('Test failed')

print('Test passed!')
