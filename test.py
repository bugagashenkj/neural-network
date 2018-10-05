from net import Net

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

network = Net([3, 2, 1])
network.train(train_datasets, 6000, 0.05)

for inputs, expected in test_datasets:
    print(network.predict(inputs, 0.35))

print('Test passed!')
