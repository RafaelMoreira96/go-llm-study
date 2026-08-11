package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"go-llm/pkg/model"
	"go-llm/pkg/sampler"
	"go-llm/pkg/tokenizer"
)

func generateCompletion(gpt *model.GPT, tok *tokenizer.Tokenizer, prompt string, maxTokens int, temp float64, topK int) {
	s := sampler.New(temp, topK)

	// Encode do prompt inicial
	inputTokens := tok.Encode(prompt)
	if len(inputTokens) == 0 {
		inputTokens = []int{tokenizer.BosID}
	}

	fmt.Printf("\n📝 Prompt: \"%s\"\n", prompt)
	fmt.Print("🤖 Geração do Modelo: ")
	fmt.Print(prompt)

	generatedCount := 0

	for generatedCount < maxTokens {
		// Mantém a janela de contexto dentro do limite SeqLen do modelo
		currInput := inputTokens
		if len(currInput) > gpt.Config.SeqLen {
			currInput = currInput[len(currInput)-gpt.Config.SeqLen:]
		}

		// Passagem Direta (Forward Pass)
		logits := gpt.Forward(currInput)

		// Pega os Logits da última posição prevista
		lastPosLogits := logits.RowSlice(logits.Rows - 1)

		// Amostragem do próximo token
		nextTokenID := s.SampleNextToken(lastPosLogits)

		// Para se gerar token EOS (End of Sequence)
		if nextTokenID == tokenizer.EosID {
			break
		}

		// Adiciona o token gerado aos tokens de entrada para o próximo passo auto-regressivo
		inputTokens = append(inputTokens, nextTokenID)

		// Decodifica o novo token e faz o streaming no terminal
		nextTokenStr := tok.Decode([]int{nextTokenID})
		fmt.Print(nextTokenStr)
		os.Stdout.Sync()

		generatedCount++
	}

	fmt.Println()
}

func main() {
	weightsFile := flag.String("weights", "weights.json", "Caminho para o arquivo de pesos (.json)")
	vocabFile := flag.String("vocab", "vocab.json", "Caminho para o arquivo de vocabulário (.json)")
	promptFlag := flag.String("prompt", "Go é", "Prompt inicial para geração de texto")
	maxTokens := flag.Int("max-tokens", 120, "Número máximo de tokens a gerar")
	temp := flag.Float64("temp", 0.7, "Temperatura de amostragem (0.0 = determinístico/greedy)")
	topK := flag.Int("top-k", 5, "Top-K amostragem de tokens")
	interactive := flag.Bool("interactive", false, "Iniciar modo interativo no terminal")
	flag.Parse()

	// 1. Carregar Vocabulário
	tok, err := tokenizer.Load(*vocabFile)
	if err != nil {
		log.Fatalf("❌ Erro ao carregar vocabulário de '%s': %v (Dica: rode 'go run ./cmd/train' primeiro!)", *vocabFile, err)
	}

	// 2. Carregar Modelo e Pesos
	gpt, err := model.LoadGPT(*weightsFile)
	if err != nil {
		log.Fatalf("❌ Erro ao carregar modelo de '%s': %v (Dica: rode 'go run ./cmd/train' primeiro!)", *weightsFile, err)
	}

	fmt.Println("=========================================================")
	fmt.Println("🤖 LLM em Go - Gerador de Texto Auto-Regressivo")
	fmt.Printf("⚙️ Parâmetros: temp=%.2f | top_k=%d | max_tokens=%d\n", *temp, *topK, *maxTokens)
	fmt.Println("=========================================================")

	if *interactive {
		fmt.Println("💬 Modo Interativo ativado. Digite seu prompt ou 'sair' para encerrar.")
		scanner := bufio.NewScanner(os.Stdin)

		for {
			fmt.Print("\n👉 Digite seu prompt: ")
			if !scanner.Scan() {
				break
			}
			input := strings.TrimSpace(scanner.Text())
			if input == "sair" || input == "exit" || input == "quit" {
				fmt.Println("Até logo! 👋")
				break
			}
			if input == "" {
				continue
			}

			generateCompletion(gpt, tok, input, *maxTokens, *temp, *topK)
		}
	} else {
		generateCompletion(gpt, tok, *promptFlag, *maxTokens, *temp, *topK)
	}
}
