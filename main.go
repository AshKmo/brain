package brain

import (
	"math"
)

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func DSigmoid(x float64) float64 {
	return Sigmoid(x) * (1 - Sigmoid(x))
}

type NeuralNet struct {
	nodeCounts []int

	weights [][][]float64
	biases [][]float64

	squisher func(float64) float64
	dsquisher func(float64) float64
}

func weightsBiasesLists(initialWeights float64, initialBiases float64, nodeCounts []int) ([][][]float64, [][]float64) {
	layers := len(nodeCounts)

	weights := make([][][]float64, layers - 1)
	biases := make([][]float64, layers - 1)

	for nc := 1; nc < layers; nc++ {
		biases[nc - 1] = make([]float64, nodeCounts[nc])

		for i := 0; i < len(biases[nc - 1]); i++ {
			biases[nc - 1][i] = initialBiases
		}

		weights[nc - 1] = make([][]float64, nodeCounts[nc])

		for i := 0; i < len(weights[nc - 1]); i++ {
			weights[nc - 1][i] = make([]float64, nodeCounts[nc - 1])

			for x := 0; x < len(weights[nc - 1][i]); x++ {
				weights[nc - 1][i][x] = initialWeights
			}
		}
	}

	return weights, biases
}

func NewNeuralNet(squisher func(float64) float64, dsquisher func(float64) float64, nodeCounts ...int) *NeuralNet {
	nn := new(NeuralNet)

	nn.nodeCounts = nodeCounts

	nn.weights, nn.biases = weightsBiasesLists(0.5, 0, nodeCounts)

	nn.squisher = squisher
	nn.dsquisher = dsquisher
	
	return nn
}

func (nn NeuralNet) Feed(inputs ...float64) ([][]float64, []float64) {
	rawActivations := make([][]float64, len(nn.nodeCounts) - 1)

	lastResults := inputs

	for layer := 0; layer < len(nn.nodeCounts) - 1; layer++ {
		nodeCount := len(nn.biases[layer])

		rawActivations[layer] = make([]float64, nodeCount)

		newResults := make([]float64, nodeCount)

		for node := 0; node < nodeCount; node++ {
			var total float64 = 0

			for lastNode := 0; lastNode < len(nn.weights[layer][node]); lastNode++ {
				total += lastResults[lastNode] * nn.weights[layer][node][lastNode] + nn.biases[layer][node]
			}

			rawActivations[layer][node] = total
			newResults[node] = nn.squisher(total)
		}

		lastResults = newResults
	}

	return rawActivations, lastResults
}

func (nn NeuralNet) Train(dataset [][][]float64, wildness float64) float64 {
	var totalCost float64 = 0

	totalWeightAdjustments, totalBiasAdjustments := weightsBiasesLists(0, 0, nn.nodeCounts)

	for _, pair := range dataset {
		input := pair[0]
		expectation := pair[1]

		rawActivations, results := nn.Feed(input...)

		differences := make([]float64, len(results))

		for i, r := range results {
			d := expectation[i] - r
			differences[i] = d
			totalCost += math.Pow(d, 2)
		}

		for layer := len(nn.nodeCounts) - 2; layer >= 0; layer-- {
			newDifferences := make([]float64, len(nn.weights[layer][0]))

			for node := 0; node < len(nn.biases[layer]); node++ {
				dcdz := nn.dsquisher(rawActivations[layer][node]) * differences[node] * wildness

				totalBiasAdjustments[layer][node] += dcdz

				for lastNode := 0; lastNode < len(nn.weights[layer][node]); lastNode++ {
					var dcdw float64 = dcdz

					if (layer == 0) {
						dcdw *= input[lastNode]
					} else {
						dcdw *= nn.squisher(rawActivations[layer - 1][lastNode])
					}

					totalWeightAdjustments[layer][node][lastNode] += dcdw

					dcda := nn.weights[layer][node][lastNode] * dcdz
					newDifferences[lastNode] += dcda
				}
			}

			differences = newDifferences
		}
	}

	for layer := 0; layer < len(nn.nodeCounts) - 1; layer++ {
		for node := 0; node < nn.nodeCounts[layer + 1]; node++ {
			for weight := 0; weight < len(nn.weights[layer][node]); weight++ {
				nn.weights[layer][node][weight] += totalWeightAdjustments[layer][node][weight] / float64(len(dataset))
			}

			nn.biases[layer][node] += totalBiasAdjustments[layer][node] / float64(len(dataset))
		}
	}

	return totalCost
}
