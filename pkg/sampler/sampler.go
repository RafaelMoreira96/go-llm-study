package sampler

import (
	"math"
	"math/rand"
	"sort"

	"go-llm/pkg/matrix"
)

// Sampler encapsula opções de amostragem de tokens durante a geração de texto.
type Sampler struct {
	Temperature float64 // Controle de aleatoriedade/criatividade (ex: 0.7 - 1.0)
	TopK        int     // Restringe a amostragem aos K tokens com maiores probabilidades
}

func New(temperature float64, topK int) *Sampler {
	return &Sampler{
		Temperature: temperature,
		TopK:        topK,
	}
}

// SampleNextToken seleciona o próximo token ID com base no vetor de Logits da última posição.
func (s *Sampler) SampleNextToken(logits []float64) int {
	if len(logits) == 0 {
		return 0
	}

	// 1. Amostragem Greedy se Temperature for 0 ou negativa
	if s.Temperature <= 0.0 {
		return matrix.Argmax(logits)
	}

	// 2. Aplica a escala de Temperatura aos logits: z = z / T
	scaledLogits := make([]float64, len(logits))
	for i, v := range logits {
		scaledLogits[i] = v / s.Temperature
	}

	// 3. Aplica Top-K se especificado
	if s.TopK > 0 && s.TopK < len(scaledLogits) {
		type pair struct {
			idx int
			val float64
		}
		pairs := make([]pair, len(scaledLogits))
		for i, v := range scaledLogits {
			pairs[i] = pair{idx: i, val: v}
		}

		// Ordena em ordem decrescente de valor
		sort.Slice(pairs, func(i, j int) bool {
			return pairs[i].val > pairs[j].val
		})

		// Define o corte (threshold) no valor do K-ésimo elemento
		threshold := pairs[s.TopK-1].val
		for i := range scaledLogits {
			if scaledLogits[i] < threshold {
				scaledLogits[i] = -1e9
			}
		}
	}

	// 4. Calcula Softmax numericamente estável
	maxLogit := scaledLogits[0]
	for _, v := range scaledLogits {
		if v > maxLogit {
			maxLogit = v
		}
	}

	sumExp := 0.0
	probs := make([]float64, len(scaledLogits))
	for i, v := range scaledLogits {
		expVal := math.Exp(v - maxLogit)
		probs[i] = expVal
		sumExp += expVal
	}

	for i := range probs {
		probs[i] /= sumExp
	}

	// 5. Amostragem Aleatória Categórica baseada na Distribuição de Probabilidades
	r := rand.Float64()
	cumulative := 0.0
	for i, p := range probs {
		cumulative += p
		if r <= cumulative {
			return i
		}
	}

	return len(logits) - 1
}
