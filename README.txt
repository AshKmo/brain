# brain
This is a simple library for Golang that provides a simple implementation of an artificial neural network of any arbitrary size, and provides basic functions for training the network.
## Functions
- `Sigmoid(x float64) float64`: the sigmoid function
- `DSigmoid(x float64) float64`: the sigmoid function's derivative
## Types
### `NeuralNet`
A NeuralNet represents a neural network. A new NeuralNet can be conveniently created using the function `NewNeuralNet`, which accepts the following parameters:
- `squisher func(float64) float64`: the function to use for "squishing" the inputs of neurons to fit within the domain [0, 1]
- `dsquisher func(float64) float64`: the derivative of the squisher function, used for training the model
- `nodeCounts ...int`: a series of integers specifying the widths, in nodes, of each layer in the network, excluding the input layer

A NeuralNet can be manipulated by the following functions:
#### `(nn NeuralNet) Feed(inputs ...float64) ([][]float64, []float64)`
Accepts a series of input values and produces the raw and final activations (respectively) of the output layer of the network when fed these inputs.
#### `(nn) NeuralNet Train(dataset [][][]float64, wildness float64) float64`
Accepts a list whose elements are each a pair of lists containing the input and output values (respectively) upon which the neural network is to be trained. Also accepts a "wildness" value dictating the coefficient of the magnitude of the amount of change to be made to the model at each backpropagation step.
