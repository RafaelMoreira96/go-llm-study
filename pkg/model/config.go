package model

// Config define os hiperparâmetros da arquitetura Transformer/GPT.
type Config struct {
	VocabSize int `json:"vocab_size"` // Tamanho total do vocabulário de tokens
	DModel    int `json:"d_model"`    // Dimensão das representações de embedding (ex: 64, 128)
	NumHeads  int `json:"num_heads"`  // Número de cabeças no Multi-Head Attention (ex: 4)
	NumLayers int `json:"num_layers"` // Número de blocos Transformer (ex: 2 ou 3)
	SeqLen    int `json:"seq_len"`    // Tamanho do contexto/janela de sequência (ex: 32 ou 64)
}

// DefaultConfig retorna uma configuração padrão eficiente para treino local rápido.
func DefaultConfig(vocabSize int) Config {
	return Config{
		VocabSize: vocabSize,
		DModel:    64,
		NumHeads:  4,
		NumLayers: 2,
		SeqLen:    32,
	}
}
