package nn

import (
	"errors"
	"gonum.org/v1/gonum/mat"
	"math/rand"
	"time"
)

// NeuralNet stores  info on trained nn network
type NeuralNet struct {
	config  NeuralNetConfig
	wHidden *mat.Dense
	bHidden *mat.Dense
	wOut    *mat.Dense
	bOut    *mat.Dense
}

// NeuralNetConfig stores nn`s architecture and learning parameters
type NeuralNetConfig struct {
	inputNeurons  int
	outputNeurons int
	hiddenNeurons int
	numEpochs     int
	learningRate  float64
}

// NewNetwork initializaes a new nn network
func NewNetwork(config NeuralNetConfig) *NeuralNet {
	return &NeuralNet{config: config}
}

// Train trains a nn network using backpropagation
func (nn *NeuralNet) Train(x, y *mat.Dense) error {
	randSource := rand.NewSource(time.Now().UnixNano())
	randGen := rand.New(randSource)

	// initialize Hidden and Output Weight and Bias with random values
	wHiddenRaw := make([]float64, nn.config.hiddenNeurons*nn.config.inputNeurons)
	bHiddenRaw := make([]float64, nn.config.hiddenNeurons)
	wOutRaw := make([]float64, nn.config.outputNeurons*nn.config.hiddenNeurons)
	bOutRaw := make([]float64, nn.config.outputNeurons)
	for _, param := range [][]float64{wHiddenRaw, bHiddenRaw, wOutRaw, bOutRaw} {
		for i := range param {
			param[i] = randGen.Float64()
		}
	}

	// put them into matrices
	wHidden := mat.NewDense(nn.config.inputNeurons, nn.config.hiddenNeurons, wHiddenRaw)
	bHidden := mat.NewDense(1, nn.config.hiddenNeurons, bHiddenRaw)
	wOut := mat.NewDense(nn.config.hiddenNeurons, nn.config.outputNeurons, wOutRaw)
	bOut := mat.NewDense(1, nn.config.outputNeurons, bOutRaw)

	output := new(mat.Dense)

	// begin forward propagation
	for i := 0; i < nn.config.numEpochs; i++ {
		// EPOCH STARTED

		// multiply input layer by hidden layer
		hiddenLayerInput := new(mat.Dense)
		hiddenLayerInput.Mul(x, wHidden)
		// add Bias to input
		addBHidden := func(_, col int, v float64) float64 {
			return v + bHidden.At(0, col)
		}
		hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)
		// perform activation
		hiddenLayerActivations := new(mat.Dense)
		applySigmoid := func(_, _ int, v float64) float64 {
			return sigmoid(v)
		}
		hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

		// derive output weights
		outputLayerInput := new(mat.Dense)
		outputLayerInput.Mul(hiddenLayerActivations, wOut)
		// add Bias to output
		addBOut := func(_, col int, v float64) float64 {
			return v + bOut.At(0, col)
		}
		//apply activation on output
		outputLayerInput.Apply(addBOut, outputLayerInput)
		output.Apply(applySigmoid, outputLayerInput)

		// AT THIS POINT, WE HAVE PREDICTIONS IN 'output'. NOW, WE NEED TO PERFORM BACKPROPAGATION
		// TO UPDATE WEIGHTS AND BIASES ACCORDING TO THE ERROR WE DERIVE FROM 'output - label'
		networkError := new(mat.Dense)
		networkError.Sub(y, output)

		slopeOutputLayer := new(mat.Dense)

		// derivative of activation on output layer
		applySigmoidPrime := func(_, _ int, v float64) float64 {
			return sigmoidPrime(v)
		}
		slopeOutputLayer.Apply(applySigmoidPrime, output)

		// derivative of activation on hidden layer
		slopeHiddenLayer := new(mat.Dense)
		slopeHiddenLayer.Apply(applySigmoidPrime, hiddenLayerActivations)

		// update Output layer with error
		dOutput := new(mat.Dense)
		dOutput.MulElem(networkError, slopeOutputLayer)

		// derive error at hidden layer
		errorAtHiddenLayer := new(mat.Dense)
		errorAtHiddenLayer.Mul(dOutput, wOut.T())

		// update Hidden layer
		dHiddenLayer := new(mat.Dense)
		dHiddenLayer.MulElem(errorAtHiddenLayer, slopeHiddenLayer)

		// WE APPLIED THE ERROR TO DATA ON LAYERS.
		// NOW WE NEED TO UPDATE PARAMETERS

		// update weight for output layer
		wOutAdj := new(mat.Dense)
		wOutAdj.Mul(hiddenLayerActivations.T(), dOutput)
		wOutAdj.Scale(nn.config.learningRate, wOutAdj) // scale according to learning rate
		wOut.Add(wOut, wOutAdj)

		// adjust bias for output layer
		bOutAdj, err := sumAlongAxis(0, dOutput)
		if err != nil {
			return err
		}
		bOutAdj.Scale(nn.config.learningRate, bOutAdj) // scale according to learning rate
		bOut.Add(bOut, bOutAdj)

		// update weight for hidden layer (only take x.T() because there is no activations on input layer)
		wHiddenAdj := new(mat.Dense)
		wHiddenAdj.Mul(x.T(), dHiddenLayer)
		wHiddenAdj.Scale(nn.config.learningRate, wHiddenAdj) // scale according to learning rate
		wHidden.Add(wHidden, wHiddenAdj)

		// adjust bias for hidden layer
		bHiddenAdj, err := sumAlongAxis(0, dHiddenLayer)
		if err != nil {
			return err
		}
		bHiddenAdj.Scale(nn.config.learningRate, bHiddenAdj) // scale according to learning rate
		bHidden.Add(bHidden, bHiddenAdj)

		// EPOCH FINISHED, COMPLETED FORWARD AND BACK PROPAGATIONS
	}

	// after completing the epochs, update nn network with found weights and biases
	nn.wHidden = wHidden
	nn.bHidden = bHidden
	nn.wOut = wOut
	nn.bOut = bOut
	return nil
}

// Predict makes predictions based on trained NN.
// Essentially the feed-forward part without back propagation.
func (nn *NeuralNet) Predict(x *mat.Dense) (*mat.Dense, error) {
	// check to see if NeuralNet was trained
	if nn.wHidden == nil || nn.wOut == nil || nn.bHidden == nil || nn.bOut == nil {
		return nil, errors.New("supplied nn net weights and biases are empty")
	}

	output := new(mat.Dense)
	// now we feed forward data throught our trained nn network
	hiddenLayerInput := new(mat.Dense)
	hiddenLayerInput.Mul(x, nn.wHidden)
	addBHidden := func(_, col int, v float64) float64 {
		return v + nn.bHidden.At(0, col)
	}
	hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

	hiddenLayerActivations := new(mat.Dense)
	applySigmoid := func(_, _ int, v float64) float64 {
		return sigmoid(v)
	}
	hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

	outputLayerInput := new(mat.Dense)
	outputLayerInput.Mul(hiddenLayerActivations, nn.wOut)
	addBOut := func(_, col int, v float64) float64 {
		return v + nn.bOut.At(0, col)
	}
	outputLayerInput.Apply(addBOut, outputLayerInput)
	output.Apply(applySigmoid, outputLayerInput)

	return output, nil
}