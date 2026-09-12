package main

import (
	"math"
	"math/rand"
)

// Обобщенная ф-ия активации
type ActivationFunction struct {
	F  func(x float64) float64            // функция активации
	Df func(x float64) float64            // производная ф-ии активации
	Ff func(src []float64, dst []float64) // функция активации над массивом
}

// Слой нейронной сети
type Layer struct {
	numIn   int                // к-во входов
	numOut  int                // к-во выходов
	af      ActivationFunction // ф-ия активации
	w       []float64          // веса
	outputs []float64          // рассчитанные значения выходов слоя
	dw      []float64          // накопленные поправки для весов
	bias    []float64          // веса смещения
	errors  []float64          // рассчитанные ошибки для выходов слоя
	sums    []float64          // рассчитанные суммы перед применением функции активации
}

func NewLayer(numIns, numOuts int, af ActivationFunction) *Layer {
	layer := &Layer{
		numIn:   numIns,
		numOut:  numOuts,
		af:      af,
		w:       make([]float64, numIns*numOuts),
		dw:      make([]float64, (numIns+1)*numOuts), // для весов и смещений (+1)
		outputs: make([]float64, numOuts),
		bias:    make([]float64, numOuts),
		errors:  make([]float64, numOuts),
		sums:    make([]float64, numOuts),
	}

	for i := range numIns * numOuts {
		layer.w[i] = rand.NormFloat64() / float64(numOuts)
	}

	return layer
}

// Расчет прямого распространения
func (l *Layer) Forward(ins []float64) {

	for j := range l.numOut {
		l.sums[j] = 0
	}
	for i := range l.numIn {
		offset := i * l.numOut
		for j := range l.numOut {
			l.sums[j] += l.w[offset+j] * ins[i]
		}
	}
	F := l.af.F
	for j := range l.numOut {
		l.sums[j] += l.bias[j]
		l.outputs[j] = F(l.sums[j]) // bias + активация в одном шаге
	}

}

// Обратное распространение ошибки и расчет поправки для весов
func (l *Layer) Backward(layerIn []float64, nextLayer *Layer) {
	for i := range nextLayer.numIn {
		errSum := 0.0
		df := l.af.Df(l.sums[i])
		offset := i * nextLayer.numOut
		for j := range nextLayer.numOut {
			errSum += nextLayer.errors[j] * nextLayer.w[j+offset]
		}
		errI := errSum * df
		l.errors[i] = errI

		for j := range l.numIn {
			l.dw[i+j*l.numOut] += errI * layerIn[j]
		}

		l.dw[l.numIn*l.numOut+i] += errI
	}
}

// Применение поправки для весов
var momentum = 1.9

func (l *Layer) UpdateWeights(rate float64, batchSize int) {
	k := momentum / float64(batchSize)
	rk := rate * k
	for i := range l.numIn {
		offset := i * l.numOut
		for j := range l.numOut {
			idx := j + offset
			l.w[idx] += l.dw[idx] * rk
			l.dw[idx] = 0
		}
	}

	biasOffset := l.numIn * l.numOut
	for j := range l.numOut {
		biasIdx := biasOffset + j
		l.bias[j] += l.dw[biasIdx] * rk
		l.dw[biasIdx] = 0
	}

}

// Структура для упрощения описания слоя при инициализации сети
type LayerDescription struct {
	NumIn int
	AF    ActivationFunction
}

// Сюда упаковываем обучающий пример и ожидаемый результат
type TrainSample struct {
	Input []float64
	Goal  []float64
}

type TrainStats struct {
	loss     float64
	accuracy float64
}

// Собственно сеть
type Net struct {
	numIn           int                                         // кло-во входов
	layers          []*Layer                                    // слои
	globalIteration int                                         // счетчик нужен для пакетного обучения
	Loss            func(out []float64, goal []float64) float64 // ф-ия потерь
	Accuracy        func(out []float64, goal []float64) bool    // ф-ия проверки корректности результата
}

// среднеквадратическая ошибка
func mse(out []float64, goal []float64) float64 {
	result := 0.0
	for i := range out {
		diff := out[i] - goal[i]
		result += (diff * diff) / float64(len(out))
	}
	return result
}

// Корректность по умолчанию - просто равенство выходов сети и ожидаемого результата
func defaultAccuracy(out []float64, goal []float64) bool {
	for i := range out {
		if goal[i] != out[i] {
			return false
		}
	}
	return true
}

// Конструктор сети
func NewNet(numIns int, layersDescr []LayerDescription) *Net {
	if numIns <= 0 {
		panic("numIn should be greater than 0")
	}
	if len(layersDescr) < 2 {
		panic("number of layers should be greater than 1")
	}

	net := &Net{
		numIn:  numIns,
		layers: make([]*Layer, len(layersDescr)),
	}

	net.layers[0] = NewLayer(numIns, layersDescr[0].NumIn, layersDescr[0].AF)

	// "Сцепляем" слои - выходы предыдущего = входам последующего
	for i := 1; i < len(layersDescr); i++ {
		net.layers[i] = NewLayer(
			net.layers[i-1].numOut,
			layersDescr[i].NumIn,
			layersDescr[i].AF,
		)
	}
	net.Loss = mse
	net.Accuracy = defaultAccuracy

	return net
}

// Прямое распространение по слоям
func (n *Net) Forward(input []float64) {
	n.layers[0].Forward(input)
	for i := 1; i < len(n.layers); i++ {
		n.layers[i].Forward(n.layers[i-1].outputs)
	}
}

// Обратное распространение по слоям
func (n *Net) BatchBackPropagation(inputs, sample []float64, rate float64) {
	outLayer := n.layers[len(n.layers)-1]
	preOutLayer := n.layers[len(n.layers)-2]
	nOuts := float64(len(outLayer.outputs)) // число выходов выходного слоя
	df := outLayer.af.Df
	for i := range outLayer.numOut {

		outLayer.errors[i] = (2.0 * (sample[i] - outLayer.outputs[i]) / nOuts) * df(outLayer.sums[i])
		tmp := outLayer.errors[i]
		for j := range outLayer.numIn {
			outLayer.dw[i+j*outLayer.numOut] += tmp * preOutLayer.outputs[j]
		}
		outLayer.dw[outLayer.numIn*outLayer.numOut+i] += tmp
	}

	l := len(n.layers) - 1
	for l > 0 {
		currentLayer := n.layers[l]
		prevLayer := n.layers[l-1]

		var layerInputs []float64
		if l == 1 {
			layerInputs = inputs
		} else {
			layerInputs = n.layers[l-2].outputs
		}

		prevLayer.Backward(layerInputs, currentLayer)
		l--
	}
}

// Коррекция весов из накопленных значений поправок dW
func (n *Net) BatchCorrectWeights(rate float64, batchSize int) {
	for _, layer := range n.layers {
		layer.UpdateWeights(rate, batchSize)
	}
}

// Шаг обучения для одного примера
func (n *Net) Train(input, sample []float64, rate float64) {
	n.Forward(input)
	n.BatchBackPropagation(input, sample, rate)
	n.BatchCorrectWeights(rate, 1)
}

// Пакетное обучение.
// trainSet   - весь трайнсет,
// rate       - скорость обучения,
// batchSize  - размер батча,
// iterations - кол-во примеров выбранных для обучения.
func (n *Net) BatchTrain(trainSet []TrainSample, rate float64, batchSize int, iterations int) TrainStats {
	var iters int
	l := len(trainSet)
	if iterations == 0 {
		iters = l
	} else {
		iters = iterations
	}

	loss := 0.0
	acc := 0.0

	for range iters {
		n.globalIteration++
		idx := rand.Intn(l)
		trainSample := trainSet[idx]

		n.Forward(trainSample.Input)
		n.BatchBackPropagation(trainSample.Input, trainSample.Goal, rate)

		loss += n.Loss(n.layers[len(n.layers)-1].outputs, trainSample.Goal)
		if !n.Accuracy(n.layers[len(n.layers)-1].outputs, trainSample.Goal) {
			acc += 1.0
		}

		if n.globalIteration%batchSize == 0 {
			n.BatchCorrectWeights(rate, batchSize)
		}
	}

	loss /= float64(iterations)
	acc = 1.0 - acc/float64(iterations)

	if n.globalIteration%batchSize != 0 {
		n.BatchCorrectWeights(rate, batchSize)
	}

	return TrainStats{loss, acc}
}

// Получить копию выходов сети
func (n *Net) Outputs() []float64 {
	lastLayer := n.layers[len(n.layers)-1]
	output := make([]float64, len(lastLayer.outputs))
	copy(output, lastLayer.outputs)
	return output
}

// Получить результат
func (n *Net) Query(inputs []float64) []float64 {
	n.Forward(inputs)
	return n.Outputs()
}

func newRelu() ActivationFunction {
	F := func(x float64) float64 {
		if x >= 0 {
			return x
		}
		return 0
	}
	Df := func(x float64) float64 {
		if x >= 0 {
			return 1
		}
		return 0
	}

	Ff := func(src, dst []float64) {
		for i := range src {
			dst[i] = F(src[i])
		}
	}
	return ActivationFunction{F, Df, Ff}
}

func newParamLinear(k1 float64, k2 float64) ActivationFunction {

	var x1 float64 = -0.5 / k1
	var x2 float64 = 0.5 / k1
	var b1 float64 = -k2 * x1
	var b2 float64 = 1.0 - k2*x2

	F := func(x float64) float64 {
		if x < x1 {
			return k2*x + b1
		} else if x > x2 {
			return k2*x + b2
		} else {
			return k1*x + 0.5
		}
	}

	Df := func(x float64) float64 {
		if x < x1 || x > x2 {
			return k2
		} else {
			return k1
		}
	}

	Ff := func(src, dst []float64) {
		for i := range src {
			dst[i] = F(src[i])
		}
	}

	return ActivationFunction{F, Df, Ff}

}

func newSigmoid() ActivationFunction {
	F := func(x float64) float64 {
		return 1.0 / (1.0 + math.Exp(-x))
	}
	Df := func(x float64) float64 {
		fx := F(x)
		return fx * (1.0 - fx)
	}
	Ff := func(src, dst []float64) {
		for i := range src {
			dst[i] = F(src[i])
		}
	}
	return ActivationFunction{F, Df, Ff}
}

func newLinear() ActivationFunction {
	F := func(x float64) float64 {
		return x
	}
	Df := func(x float64) float64 {
		return 1
	}
	Ff := func(src, dst []float64) {
		for i := range src {
			dst[i] = F(src[i])
		}
	}
	return ActivationFunction{F, Df, Ff}
}

func newWavelet() ActivationFunction {
	F := func(x float64) float64 {
		x2 := x * x
		return (1 - x2) * math.Exp(-(x2 / 2.0))
	}
	Df := func(x float64) float64 {
		x2 := x * x
		e := math.Exp(-(x2 / 2))
		return e * x * (x2 - 3)
	}
	Ff := func(src, dst []float64) {
		for i := range src {
			dst[i] = F(src[i])
		}
	}
	return ActivationFunction{F, Df, Ff}
}

func newSine() ActivationFunction {
	F := func(x float64) float64 {
		return math.Sin(x)
	}
	Df := func(x float64) float64 {
		return math.Cos(x)
	}
	Ff := func(src, dst []float64) {
		for i := range src {
			dst[i] = F(src[i])
		}
	}
	return ActivationFunction{F, Df, Ff}
}

// Export activation functions
var (
	relu    = newRelu()
	sigmoid = newSigmoid()
	linear  = newLinear()
	wavelet = newWavelet()
	sine    = newSine()
)
