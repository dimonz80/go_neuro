package main

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"
)

func loadMnist(filePath string) []TrainSample {
	var result = []TrainSample{}

	var file, err = os.OpenFile(filePath, os.O_RDONLY, 0666)
	if err != nil {
		panic(err)
	}

	defer file.Close()

	var scanner = bufio.NewScanner(file)
	var lineNum = 0

	for scanner.Scan() {
		var line = scanner.Text()
		var arr = strings.Split(line, ",")
		var category, _ = strconv.Atoi(arr[0])
		var in = make([]float64, 28*28)
		var goal = make([]float64, 10)
		goal[category] = 1.0

		for i := 1; i < len(in); i++ {
			var v, _ = strconv.Atoi(arr[i])
			in[i] = float64(v) / 255.0
		}

		result = append(result, TrainSample{in, goal})

		lineNum += 1
	}
	return result
}

func argmax(a []float64) int {

	var max = a[0]
	var maxI = 0
	for i := range len(a) {
		if a[i] > max {
			max = a[i]
			maxI = i
		}
	}
	return maxI
}

func test(trainSet []TrainSample, net *Net) float64 {
	var err = 0.0
	l := len(trainSet)
	for i := range l {
		sample := trainSet[i]
		out := net.Query(sample.Input)
		if !net.Accuracy(out, sample.Goal) {
			err += 1.0
		}
	}
	err /= float64(l)
	var acc = 1.0 - err
	return acc
}

func showSample(data []float64) string {

	var result = strings.Builder{}
	for row := range 28 {
		for col := range 28 {
			var b byte = ' '
			var v = data[col+row*28]
			switch {
			case v < 0.2:
				b = '.'
			case v < 0.5:
				b = ';'
			case v < 0.8:
				b = '~'
			default:
				b = '#'
			}
			result.WriteByte(b)
		}
		result.WriteByte('\n')
	}
	return result.String()
}

func testMnist() {

	var rate = 0.1
	var epochs = 500
	var batchSize = 8
	var iters = 60000

	fmt.Println("Loading samples...")

	var trainSet = loadMnist("../../MNIST/MNIST_CSV/mnist_train.csv")
	fmt.Println("Loading test samples...")
	var testTrainSet = loadMnist("../../MNIST/MNIST_CSV/mnist_test.csv")

	var net = NewNet(28*28, []LayerDescription{
		{48, relu},
		{128, relu},
		{10, NewParamLinear(1, 0)},
	})

	net.Accuracy = func(out, goal []float64) bool {
		return argmax(out) == argmax(goal)
	}

	var trainSetSize = len(trainSet)

	fmt.Println("Start train...")
	prevAcc := 0.0
	dAcc := 0.0
	for epoch := range epochs {
		t1 := time.Now()
		if dAcc < 0 {
			rate *= 0.9
		} else if dAcc < 0.001 {
			rate *= 1.1
		}
		stats := net.BatchTrain(trainSet, rate, batchSize, iters)
		dAcc = stats.accuracy - prevAcc
		prevAcc = stats.accuracy
		t2 := time.Now()
		dt := t2.UnixMilli() - t1.UnixMilli()
		//acc := test(trainSet, net)
		testAcc := test(testTrainSet, net)
		fmt.Printf("epoch=%v acc=%.4f%% testAcc=%.4f%% dt=%v rate=%.4f\n", epoch, stats.accuracy*100.0, testAcc*100.0, dt, rate)
	}

	var randSample = trainSet[rand.Intn(trainSetSize)]
	var result = argmax(net.Query(randSample.Input))
	var goal = argmax(randSample.Goal)

	fmt.Printf("goal=%v result=%v\n", goal, result)
	fmt.Println(showSample(randSample.Input))

}
