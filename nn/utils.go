package nn

import (
	"encoding/csv"
	"errors"
	"fmt"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"log"
	"os"
	"strconv"
)

// UNEXPORTED
// I think this helps with getting the Bias,
// sums bias values from 1st column of matrix a.k.a. X0. THETA 0
func sumAlongAxis(axis int, m *mat.Dense) (*mat.Dense, error) {
	numRows, numCols := m.Dims()

	var output *mat.Dense

	switch axis {
	case 0:
		data := make([]float64, numCols)
		for i := 0; i < numCols; i++ {
			col := mat.Col(nil, i, m)
			data[i] = floats.Sum(col)
		}
		output = mat.NewDense(1, numCols, data)
	case 1:
		data := make([]float64, numRows)
		for i := 0; i < numRows; i++ {
			row := mat.Row(nil, i, m)
			data[i] = floats.Sum(row)
		}
		output = mat.NewDense(numRows, 1, data)
	default:
		return nil, errors.New("bad axis, must be 0 or 1")
	}
	return output, nil
}

// loadDataFromCSV splits CSV-formatted dataset into inputs and lables matrices.
func loadDataFromCSV(file *os.File) (*mat.Dense, *mat.Dense, error) {
	// fetch data from CSV file
	// data has 4 features and is one-hot encoded for flower family type

	reader := csv.NewReader(file)
	reader.FieldsPerRecord = 7

	rawCSVData, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
		return nil, nil, err
	}

	inputsData := make([]float64, 4*len(rawCSVData))
	labelsData := make([]float64, 3*len(rawCSVData))

	var inputsIndex, labelsIndex int

	for idx, record := range rawCSVData {
		if idx == 0 {
			continue
		}

		for i, val := range record {
			parsedVal, err := strconv.ParseFloat(val, 64)
			if err != nil {
				log.Fatal(err)
				return nil, nil, err
			}

			// adding labels
			if i == 4 || i == 5 || i == 6 {
				labelsData[labelsIndex] = parsedVal
				labelsIndex++
				continue
			}

			inputsData[inputsIndex] = parsedVal
			inputsIndex++
		}
	}

	// make matrices
	inputs := mat.NewDense(len(rawCSVData), 4, inputsData)
	labels := mat.NewDense(len(rawCSVData), 3, labelsData)

	return inputs, labels, nil
}

func printTrained(network *NeuralNet) {
	// Output the weights that define our network
	f := mat.Formatted(network.wHidden, mat.Prefix(" "))
	fmt.Printf("\nwHidden = % v\n\n", f)

	f = mat.Formatted(network.bHidden, mat.Prefix(" "))
	fmt.Printf("\nbHidden = % v\n\n", f)

	f = mat.Formatted(network.wOut, mat.Prefix(" "))
	fmt.Printf("\nwOut = % v\n\n", f)

	f = mat.Formatted(network.bOut, mat.Prefix(" "))
	fmt.Printf("\nbOut = % v\n\n", f)
}
