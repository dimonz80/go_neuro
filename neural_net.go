package main

import (
	"math"
	"math/rand"
)

// Обобщенная ф-ия активации
type ActivationFunction interface {
	F(x float64) float64             // функция активации
	Df(x float64) float64            // производная ф-ии активации
	Ff(src []float64, dst []float64) // функция активации над массивом
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
		sum := 0.0
		for i := range l.numIn {
			idx := j + i*l.numOut
			sum += l.w[idx] * ins[i]
		}
		l.sums[j] = sum + l.bias[j]
	}
	l.af.Ff(l.sums, l.outputs)
}

// Обратное распространение ошибки и расчет поправки для весов
func (l *Layer) Backward(layerIn []float64, nextLayer *Layer) {
	for i := range nextLayer.numIn {
		errSum := 0.0
		for j := range nextLayer.numOut {
			errSum += nextLayer.errors[j] * nextLayer.w[j+i*nextLayer.numOut]
		}
		errI := errSum * l.af.Df(l.sums[i])
		l.errors[i] = errI

		for j := range l.numIn {
			l.dw[i+j*l.numOut] += errI * layerIn[j]
		}

		l.dw[l.numIn*l.numOut+i] += errI
	}
}

// Применение поправки для весов
func (l *Layer) UpdateWeights(rate float64, batchSize int) {
	for j := range l.numOut {
		for i := range l.numIn {
			idx := j + i*l.numOut
			l.w[idx] += rate * l.dw[idx] / float64(batchSize)
			l.dw[idx] = 0
		}
		// bias
		biasIdx := l.numIn*l.numOut + j
		l.bias[j] += rate * l.dw[biasIdx] / float64(batchSize)
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

	for i := range outLayer.numOut {
		outLayer.errors[i] = sample[i] - outLayer.outputs[i]
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

// Реализация некоторых функция активации
type Relu struct{}

func (r *Relu) F(x float64) float64 {
	if x >= 0 {
		return x
	}
	return 0
}

func (r *Relu) Df(x float64) float64 {
	if x >= 0 {
		return 1
	}
	return 0
}

func (r *Relu) Ff(src, dst []float64) {
	for i := range src {
		dst[i] = r.F(src[i])
	}
}

type Sigmoid struct{}

func (s *Sigmoid) F(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func (s *Sigmoid) Df(x float64) float64 {
	fx := s.F(x)
	return fx * (1.0 - fx)
}

func (s *Sigmoid) Ff(src, dst []float64) {
	for i := range src {
		dst[i] = s.F(src[i])
	}
}

type Linear struct{}

func (l *Linear) F(x float64) float64 {
	return x
}

func (l *Linear) Df(x float64) float64 {
	return 1
}

func (l *Linear) Ff(src, dst []float64) {
	for i := range src {
		dst[i] = l.F(src[i])
	}
}

// Ломаная прямая
type ParamLinear struct {
	k1 float64
	k2 float64
	x1 float64
	x2 float64
	b1 float64
	b2 float64
}

func NewParamLinear(k1 float64, k2 float64) *ParamLinear {
	var x1 = -0.5 / k1
	var x2 = 0.5 / k1
	return &ParamLinear{k1, k2, x1, x2, -k2 * x1, 1.0 - k2*x2}
}

func (l *ParamLinear) F(x float64) float64 {
	if x < l.x1 {
		return l.k2*x + l.b1
	} else if x > l.x2 {
		return l.k2*x + l.b2
	} else {
		return l.k1*x + 0.5
	}

}

func (l *ParamLinear) Df(x float64) float64 {
	if x < l.x1 || x > l.x2 {
		return l.k2
	} else {
		return l.k1
	}
}

func (l *ParamLinear) Ff(src, dst []float64) {
	for i := range src {
		dst[i] = l.F(src[i])
	}
}

// Вейвлет "самбреро"
type Wavelet struct{}

func (w *Wavelet) F(x float64) float64 {
	x2 := x * x
	return (1 - x2) * math.Exp(-(x2 / 2.0))
}

func (w *Wavelet) Df(x float64) float64 {
	x2 := x * x
	e := math.Exp(-(x2 / 2))
	return e * x * (x2 - 3)
}

func (w *Wavelet) Ff(src, dst []float64) {
	for i := range src {
		dst[i] = w.F(src[i])
	}
}

type Sine struct{}

func (w *Sine) F(x float64) float64 {
	return math.Sin(x)
}

func (w *Sine) Df(x float64) float64 {
	return math.Cos(x)
}

func (w *Sine) Ff(src, dst []float64) {
	for i := range src {
		dst[i] = w.F(src[i])
	}
}

// Export activation functions
var (
	relu    = &Relu{}
	sigmoid = &Sigmoid{}
	linear  = &Linear{}
	wavelet = &Wavelet{}
	sine    = &Sine{}
)
