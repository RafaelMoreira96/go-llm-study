package sampler

import (
	"testing"
)

func TestGreedySampling(t *testing.T) {
	s := New(0.0, 0)
	logits := []float64{0.1, 2.5, 0.3, 1.2}

	tokenID := s.SampleNextToken(logits)
	if tokenID != 1 {
		t.Errorf("esperado tokenID 1 na amostragem greedy, obteve %d", tokenID)
	}
}

func TestTopKSampling(t *testing.T) {
	s := New(1.0, 1) // TopK = 1 equivale a escolher o maior elemento
	logits := []float64{0.1, 0.5, 9.9, 0.2}

	tokenID := s.SampleNextToken(logits)
	if tokenID != 2 {
		t.Errorf("esperado tokenID 2 com TopK=1, obteve %d", tokenID)
	}
}
