package model

import (
	"path/filepath"
	"testing"
)

func TestGPTForwardAndLoss(t *testing.T) {
	cfg := Config{
		VocabSize: 20,
		DModel:    16,
		NumHeads:  2,
		NumLayers: 1,
		SeqLen:    8,
	}

	gpt := NewGPT(cfg)

	inputTokens := []int{1, 5, 8, 12, 3}
	targetTokens := []int{5, 8, 12, 3, 2}

	logits := gpt.Forward(inputTokens)

	if logits.Rows != len(inputTokens) {
		t.Fatalf("esperado %d linhas nos logits, obteve %d", len(inputTokens), logits.Rows)
	}
	if logits.Cols != cfg.VocabSize {
		t.Fatalf("esperado %d colunas nos logits, obteve %d", cfg.VocabSize, logits.Cols)
	}

	loss := ComputeLoss(logits, targetTokens)
	if loss <= 0 || mathIsNaN(loss) {
		t.Errorf("Loss inválido: %f", loss)
	}
}

func TestGPTSaveAndLoadWeights(t *testing.T) {
	cfg := Config{
		VocabSize: 10,
		DModel:    8,
		NumHeads:  2,
		NumLayers: 1,
		SeqLen:    4,
	}

	gpt := NewGPT(cfg)

	tmpDir := t.TempDir()
	weightsPath := filepath.Join(tmpDir, "weights.json")

	err := gpt.SaveWeights(weightsPath)
	if err != nil {
		t.Fatalf("falha ao salvar pesos: %v", err)
	}

	loadedGPT, err := LoadGPT(weightsPath)
	if err != nil {
		t.Fatalf("falha ao carregar pesos: %v", err)
	}

	if loadedGPT.Config.VocabSize != gpt.Config.VocabSize {
		t.Errorf("configuração do modelo difere após reload")
	}

	tokens := []int{1, 2, 3}
	originalLogits := gpt.Forward(tokens)
	loadedLogits := loadedGPT.Forward(tokens)

	for i := range originalLogits.Data {
		if originalLogits.Data[i] != loadedLogits.Data[i] {
			t.Fatalf("logits divergiram após reload do modelo na posição %d", i)
		}
	}
}

func mathIsNaN(f float64) bool {
	return f != f
}
