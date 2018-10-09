# Backpropagation neural network prototype 

`bugaga-net-prototype` is a library that allows you to create and train a neural network of a given configuration.

## Create network weights 
`create_weights(layers_size)`
- `layers_size` - list of the number of neurons on each layer (starting from the input layer)

## Train network
`train(weights, test_datasets, iterations, learning_rate)`
- `weights` - network weights 
- `test_datasets` - list of typles of input datasets and expected datasets 
- `iterations` - number of interations 
- `learning_rate` - (0, 1) neural network learning speed / accuracy 

## Predict 
`predict(weights, input_datasets, reliable_limit)`
- `weights` - network weights 
- `input_datasets` - list of input datasets 
- `reliable_limit` - (0, 0.5) limit of unknown results 

Example:

```Python
train_datasets = [
        ([0, 0, 1], [1]),
        ([0, 1, 1], [0]),
        ([1, 0, 1], [1]),
        ([1, 1, 1], [1]),
        ([0, 0, 0], [0])
        ]

weights = create_weights([3, 2, 1])
train(weights, train_datasets, 6000, 0.05)
assert predict(weights, [0, 1, 0], 0.35) == [0] 
```
## Contributors
  - See github for full [contributors list](https://github.com/bugagashenkj/bugaga-net-proto/graphs/contributors)
