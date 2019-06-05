package nn

import (
	"fmt"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"log"
	"os"
	"testing"
)

func TestTraining(t *testing.T) {
	file, err := os.Open("test_resources/train.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	inputs, labels, _ := loadDataFromCSV(file)

	// CONFIGURE NEURAL NETWORK
	// 4 INPUT NEURONS BECAUSE 4 FEATURES
	// 3 OUTPUT NEURONS BECAUSE 3 TARGET VARIABLES {(1,0,0), (0,1,0), (0,0,1)}
	// HIDDEN NEURONS - ACTUALLY DONT KNOW, MAYBE CHECK FOR OTHER STUFF?
	// NUM-EPOCH / LEARNING RATE - DON'T REALLY KNOW, WOULD HAVE TO GUESS THAT
	config := &NeuralNetConfig{
		inputNeurons:  4,
		outputNeurons: 3,
		hiddenNeurons: 3,
		numEpochs:     100,
		learningRate:  0.1,
		activationFunction: ActivationFunction{
			sigmoid,
			sigmoidPrime,
		},
	}
	network := NewNetwork(*config)
	if err := network.Train(inputs, labels); err != nil {
		log.Fatal(err)
	}

	//printTrained(network)
	validate(network)
}

// validate gives predictions on validation data set using trained nn network.
func validate(network *NeuralNet) {
	f, err := os.Open("test_resources/valid.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	testInputs, testLabels, err := loadDataFromCSV(f)
	if err != nil {
		log.Fatal(err)
	}

	predictions, err := network.Predict(testInputs)
	if err != nil {
		log.Fatal(err)
	}

	// calculate accuracy
	var truePosNeg int
	numPreds, _ := predictions.Dims()
	for i := 0; i < numPreds; i++ {
		// Get the label
		labelRow := mat.Row(nil, i, testLabels)
		var species int
		for idx, label := range labelRow {
			if label == 1.0 {
				species = idx
				break
			}
		}

		// accumulate true positives / negatives
		if predictions.At(i, species) == floats.Max(mat.Row(nil, i, predictions)) {
			truePosNeg++
		}
	}
	accuracy := float64(truePosNeg) / float64(numPreds)
	fmt.Printf("\nAccuracy = %0.2f\n\n", accuracy)
}

