package matrix

import (
	"fmt"
	"math"
	"math/rand"
)

// Matrix representa uma matriz 2D com layout de memória continuo em 1D (row-major).
type Matrix struct {
	Rows int
	Cols int
	Data []float64
}

// New cria uma nova matriz preenchida com zeros.
func New(rows, cols int) *Matrix {
	return &Matrix{
		Rows: rows,
		Cols: cols,
		Data: make([]float64, rows*cols),
	}
}

// NewWithData cria uma matriz utilizando o slice fornecido.
func NewWithData(rows, cols int, data []float64) *Matrix {
	if len(data) != rows*cols {
		panic(fmt.Sprintf("tamanho dos dados (%d) incompatível com dimensões %dx%d", len(data), rows, cols))
	}
	return &Matrix{
		Rows: rows,
		Cols: cols,
		Data: data,
	}
}

// Random cria uma matriz preenchida com distribuição normal N(0, stddev).
func Random(rows, cols int, stddev float64) *Matrix {
	m := New(rows, cols)
	for i := range m.Data {
		m.Data[i] = rand.NormFloat64() * stddev
	}
	return m
}

// Get retorna o elemento da linha r e coluna c.
func (m *Matrix) Get(r, c int) float64 {
	return m.Data[r*m.Cols+c]
}

// Set define o elemento na linha r e coluna c.
func (m *Matrix) Set(r, c int, val float64) {
	m.Data[r*m.Cols+c] = val
}

// Copy faz uma cópia profunda da matriz.
func (m *Matrix) Copy() *Matrix {
	c := New(m.Rows, m.Cols)
	copy(c.Data, m.Data)
	return c
}

// RowSlice retorna uma fatia da linha r como um slice 1D.
func (m *Matrix) RowSlice(r int) []float64 {
	start := r * m.Cols
	return m.Data[start : start+m.Cols]
}

// SetRow define os valores da linha r a partir de um slice.
func (m *Matrix) SetRow(r int, row []float64) {
	start := r * m.Cols
	copy(m.Data[start:start+m.Cols], row)
}

// MatMul realiza a multiplicação de matrizes: C = A * B.
// A é (rowsA x colsA), B é (colsA x colsB) => C é (rowsA x colsB).
func MatMul(a, b *Matrix) *Matrix {
	if a.Cols != b.Rows {
		panic(fmt.Sprintf("dimensões incompatíveis para MatMul: (%dx%d) * (%dx%d)", a.Rows, a.Cols, b.Rows, b.Cols))
	}

	c := New(a.Rows, b.Cols)
	for i := 0; i < a.Rows; i++ {
		aRow := i * a.Cols
		cRow := i * c.Cols
		for k := 0; k < a.Cols; k++ {
			aVal := a.Data[aRow+k]
			bRow := k * b.Cols
			for j := 0; j < b.Cols; j++ {
				c.Data[cRow+j] += aVal * b.Data[bRow+j]
			}
		}
	}
	return c
}

// Transpose retorna a matriz transposta de m (Cols x Rows).
func Transpose(m *Matrix) *Matrix {
	t := New(m.Cols, m.Rows)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			t.Data[j*m.Rows+i] = m.Data[i*m.Cols+j]
		}
	}
	return t
}

// Add realiza a soma elemento a elemento: C = A + B.
func Add(a, b *Matrix) *Matrix {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		panic("dimensões incompatíveis para Add")
	}
	c := New(a.Rows, a.Cols)
	for i := range a.Data {
		c.Data[i] = a.Data[i] + b.Data[i]
	}
	return c
}

// AddBias adiciona um vetor de viés (1D de tamanho Cols) a cada linha da matriz.
func AddBias(m *Matrix, bias []float64) *Matrix {
	if len(bias) != m.Cols {
		panic("tamanho do viés incompatível com colunas da matriz")
	}
	out := New(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		rowOffset := i * m.Cols
		for j := 0; j < m.Cols; j++ {
			out.Data[rowOffset+j] = m.Data[rowOffset+j] + bias[j]
		}
	}
	return out
}

// Scale multiplica todos os elementos da matriz por um escalar.
func Scale(m *Matrix, scalar float64) *Matrix {
	out := New(m.Rows, m.Cols)
	for i := range m.Data {
		out.Data[i] = m.Data[i] * scalar
	}
	return out
}

// Softmax aplica a função Softmax numericamente estável linha por linha.
func Softmax(m *Matrix) *Matrix {
	out := New(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		rowOffset := i * m.Cols
		// Encontra o valor máximo para estabilidade numérica
		maxVal := m.Data[rowOffset]
		for j := 1; j < m.Cols; j++ {
			if m.Data[rowOffset+j] > maxVal {
				maxVal = m.Data[rowOffset+j]
			}
		}

		// Calcula exp(x - max) e a soma
		sumExp := 0.0
		for j := 0; j < m.Cols; j++ {
			expVal := math.Exp(m.Data[rowOffset+j] - maxVal)
			out.Data[rowOffset+j] = expVal
			sumExp += expVal
		}

		// Normaliza dividindo pela soma
		for j := 0; j < m.Cols; j++ {
			out.Data[rowOffset+j] /= sumExp
		}
	}
	return out
}

// ApplyCausalMask aplica a máscara causal em m. Zera (ou coloca -1e9) nas posições onde col > row.
func ApplyCausalMask(m *Matrix) {
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			if j > i {
				m.Set(i, j, -1e9)
			}
		}
	}
}

// GELU aplica a função de ativação Gaussian Error Linear Unit elemento a elemento.
func GELU(m *Matrix) *Matrix {
	out := New(m.Rows, m.Cols)
	const sqrt2OverPi = 0.7978845608028654
	for i, x := range m.Data {
		// Aproximação estatística da GELU usada no GPT-2
		cube := 0.044715 * x * x * x
		out.Data[i] = 0.5 * x * (1.0 + math.Tanh(sqrt2OverPi*(x+cube)))
	}
	return out
}

// LayerNorm aplica Layer Normalization linha a linha.
func LayerNorm(m *Matrix, gamma, beta []float64, eps float64) *Matrix {
	out := New(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		rowOffset := i * m.Cols

		// Média da linha
		mean := 0.0
		for j := 0; j < m.Cols; j++ {
			mean += m.Data[rowOffset+j]
		}
		mean /= float64(m.Cols)

		// Variância da linha
		variance := 0.0
		for j := 0; j < m.Cols; j++ {
			diff := m.Data[rowOffset+j] - mean
			variance += diff * diff
		}
		variance /= float64(m.Cols)

		stdDev := math.Sqrt(variance + eps)

		// Normalização e escala (gamma, beta)
		for j := 0; j < m.Cols; j++ {
			norm := (m.Data[rowOffset+j] - mean) / stdDev
			if len(gamma) > 0 && len(beta) > 0 {
				norm = norm*gamma[j] + beta[j]
			}
			out.Data[rowOffset+j] = norm
		}
	}
	return out
}

// Argmax retorna o índice do maior elemento em um slice.
func Argmax(slice []float64) int {
	if len(slice) == 0 {
		return -1
	}
	maxIdx := 0
	maxVal := slice[0]
	for i := 1; i < len(slice); i++ {
		if slice[i] > maxVal {
			maxVal = slice[i]
			maxIdx = i
		}
	}
	return maxIdx
}
