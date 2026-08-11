package matrix

import (
	"math"
	"testing"
)

func TestMatMul(t *testing.T) {
	a := NewWithData(2, 3, []float64{
		1, 2, 3,
		4, 5, 6,
	})
	b := NewWithData(3, 2, []float64{
		7, 8,
		9, 1,
		2, 3,
	})

	c := MatMul(a, b)
	expected := []float64{
		31, 19,
		85, 55,
	}

	if c.Rows != 2 || c.Cols != 2 {
		t.Fatalf("esperado 2x2, obteve %dx%d", c.Rows, c.Cols)
	}

	for i, val := range c.Data {
		if val != expected[i] {
			t.Errorf("índice %d: esperado %f, obteve %f", i, expected[i], val)
		}
	}
}

func TestSoftmax(t *testing.T) {
	m := NewWithData(1, 3, []float64{1.0, 2.0, 3.0})
	sm := Softmax(m)

	sum := 0.0
	for _, v := range sm.Data {
		sum += v
	}

	if math.Abs(sum-1.0) > 1e-6 {
		t.Errorf("soma da softmax deve ser 1.0, obteve %f", sum)
	}

	if sm.Data[2] <= sm.Data[1] || sm.Data[1] <= sm.Data[0] {
		t.Errorf("Softmax deve preservar a ordem dos valores")
	}
}

func TestLayerNorm(t *testing.T) {
	m := NewWithData(1, 4, []float64{2.0, 4.0, 4.0, 6.0})
	gamma := []float64{1, 1, 1, 1}
	beta := []float64{0, 0, 0, 0}

	ln := LayerNorm(m, gamma, beta, 1e-5)

	// Média de [2, 4, 4, 6] é 4.0
	// Variância é ((4+0+0+4)/4) = 2.0 -> desvio padrão ~ sqrt(2) = 1.4142
	// (2-4)/1.4142 = -1.4142, (6-4)/1.4142 = 1.4142
	if math.Abs(ln.Data[0]-(-1.4142)) > 1e-2 {
		t.Errorf("LayerNorm incorreto no elemento 0: %f", ln.Data[0])
	}
}

func TestArgmax(t *testing.T) {
	slice := []float64{0.1, 0.8, 0.3, 0.5}
	idx := Argmax(slice)
	if idx != 1 {
		t.Errorf("esperado índice 1 para argmax, obteve %d", idx)
	}
}
