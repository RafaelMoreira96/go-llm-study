package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"go-llm/pkg/model"
	"go-llm/pkg/tokenizer"
)

func main() {
	inputFile := flag.String("input", "data/input.txt", "Caminho para o arquivo de texto de treinamento")
	weightsFile := flag.String("out-weights", "weights.json", "Caminho para salvar os pesos treinados (.json)")
	vocabFile := flag.String("out-vocab", "vocab.json", "Caminho para salvar o vocabulário (.json)")
	epochs := flag.Int("epochs", 80, "Número de épocas de treinamento")
	lr := flag.Float64("lr", 0.03, "Taxa de aprendizado (Learning Rate)")
	seqLen := flag.Int("seq-len", 32, "Comprimento máximo da janela de contexto")
	dModel := flag.Int("d-model", 64, "Dimensão da representação de embedding")
	layers := flag.Int("layers", 2, "Número de blocos Transformer")
	heads := flag.Int("heads", 4, "Número de cabeças de atenção")
	flag.Parse()

	fmt.Println("=========================================================")
	fmt.Println("🚀 Iniciando Treinamento da LLM em Go do Zero")
	fmt.Println("=========================================================")

	// 1. Carregar Dataset
	content, err := os.ReadFile(*inputFile)
	if err != nil {
		log.Fatalf("❌ Erro ao ler arquivo de entrada '%s': %v", *inputFile, err)
	}
	text := string(content)
	fmt.Printf("📖 Dataset carregado: %d caracteres do arquivo '%s'\n", len(text), *inputFile)

	// 2. Construir Vocabulário e Tokenizer
	tok := tokenizer.New()
	tok.BuildFromText(text)
	vocabSize := tok.VocabSize()
	fmt.Printf("🔤 Vocabulário construído: %d tokens únicos\n", vocabSize)

	// 3. Codificar texto do dataset em IDs de tokens
	tokenIDs := tok.Encode(text)
	fmt.Printf("🔢 Dataset codificado em %d tokens\n", len(tokenIDs))

	if len(tokenIDs) <= *seqLen+1 {
		log.Fatalf("❌ O dataset é muito pequeno para a janela de sequência configurada (%d tokens vs seq-len %d)", len(tokenIDs), *seqLen)
	}

	// 4. Configurar e Instanciar Modelo GPT
	cfg := model.Config{
		VocabSize: vocabSize,
		DModel:    *dModel,
		NumHeads:  *heads,
		NumLayers: *layers,
		SeqLen:    *seqLen,
	}

	gpt := model.NewGPT(cfg)
	fmt.Printf("⚙️ Modelo instanciado (d_model=%d, heads=%d, layers=%d, seq_len=%d)\n\n",
		cfg.DModel, cfg.NumHeads, cfg.NumLayers, cfg.SeqLen)

	// 5. Loop de Treinamento
	fmt.Println("🏋️ Treinando a rede neural...")
	startTime := time.Now()

	numSamples := len(tokenIDs) - cfg.SeqLen - 1
	stepCount := 0

	for epoch := 1; epoch <= *epochs; epoch++ {
		totalEpochLoss := 0.0
		batchCount := 0

		// Percorre o dataset em janelas deslizantes (sliding window)
		for i := 0; i < numSamples; i += cfg.SeqLen / 2 {
			inputSeq := tokenIDs[i : i+cfg.SeqLen]
			targetSeq := tokenIDs[i+1 : i+cfg.SeqLen+1]

			loss := gpt.TrainStep(inputSeq, targetSeq, *lr)
			totalEpochLoss += loss
			batchCount++
			stepCount++
		}

		avgLoss := totalEpochLoss / float64(batchCount)

		if epoch == 1 || epoch%10 == 0 || epoch == *epochs {
			fmt.Printf("   📌 Época %3d/%d | Steps: %4d | Avg Loss: %.4f | Tempo decorrido: %v\n",
				epoch, *epochs, stepCount, avgLoss, time.Since(startTime).Truncate(time.Millisecond))
		}
	}

	fmt.Println("\n✅ Treinamento concluído com sucesso!")

	// 6. Salvar Vocabulário e Pesos do Modelo
	if err := tok.Save(*vocabFile); err != nil {
		log.Fatalf("❌ Erro ao salvar vocabulário: %v", err)
	}
	fmt.Printf("💾 Vocabulário salvo em '%s'\n", *vocabFile)

	if err := gpt.SaveWeights(*weightsFile); err != nil {
		log.Fatalf("❌ Erro ao salvar pesos do modelo: %v", err)
	}
	fmt.Printf("💾 Pesos do modelo salvos em '%s'\n", *weightsFile)
	fmt.Println("=========================================================")
}
