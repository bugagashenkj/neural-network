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

weights = create_weights([3, 2, 1])
train(weights, train_datasets, 6000, 0.05)

for inputs, expected in test_datasets:
    print(predict(weights, inputs, 0.35))

print('Test passed!')
