package nn

import "math"

type ActivationFunction struct {
	Primary    func(v float64) float64
	Derivative func(v float64) float64
}

// sigmoid implents sigmoid function
// our activation function
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// sigmoigPrime implements the derivative
// of sigmoid func for backpropagation
func sigmoidPrime(x float64) float64 {
	return x * (1.0 - x)
}