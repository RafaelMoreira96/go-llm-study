package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
	"unicode"
)

const (
	UnkToken = "<UNK>"
	PadToken = "<PAD>"
	BosToken = "<BOS>"
	EosToken = "<EOS>"

	UnkID = 0
	PadID = 1
	BosID = 2
	EosID = 3
)

// Tokenizer faz a conversão entre texto e IDs numéricos de tokens (suporta palavras e pontuação).
type Tokenizer struct {
	TokenToID map[string]int `json:"token_to_id"`
	IDToToken map[int]string `json:"id_to_token"`
}

// New cria um novo Tokenizer pré-inicializado com tokens especiais.
func New() *Tokenizer {
	t := &Tokenizer{
		TokenToID: make(map[string]int),
		IDToToken: make(map[int]string),
	}

	t.addToken(UnkToken, UnkID)
	t.addToken(PadToken, PadID)
	t.addToken(BosToken, BosID)
	t.addToken(EosToken, EosID)

	return t
}

func (t *Tokenizer) addToken(token string, id int) {
	t.TokenToID[token] = id
	t.IDToToken[id] = token
}

// SplitIntoTokens divide o texto em palavras, pontuações e colapsa espaços múltiplos.
func SplitIntoTokens(text string) []string {
	var tokens []string
	var current strings.Builder
	inSpace := false

	for _, r := range text {
		if unicode.IsSpace(r) {
			if current.Len() > 0 {
				tokens = append(tokens, current.String())
				current.Reset()
			}
			if !inSpace {
				tokens = append(tokens, " ")
				inSpace = true
			}
		} else if unicode.IsPunct(r) {
			inSpace = false
			if current.Len() > 0 {
				tokens = append(tokens, current.String())
				current.Reset()
			}
			tokens = append(tokens, string(r))
		} else {
			inSpace = false
			current.WriteRune(r)
		}
	}
	if current.Len() > 0 {
		tokens = append(tokens, current.String())
	}
	return tokens
}

// BuildFromText constrói o vocabulário a partir de palavras e pontuações do corpus.
func (t *Tokenizer) BuildFromText(text string) {
	rawTokens := SplitIntoTokens(text)
	seen := make(map[string]bool)
	for _, tok := range rawTokens {
		if !seen[tok] {
			seen[tok] = true
		}
	}

	var uniqueTokens []string
	for tok := range seen {
		uniqueTokens = append(uniqueTokens, tok)
	}
	sort.Strings(uniqueTokens)

	nextID := len(t.TokenToID)
	for _, tok := range uniqueTokens {
		if _, exists := t.TokenToID[tok]; !exists {
			t.addToken(tok, nextID)
			nextID++
		}
	}
}

// VocabSize retorna a quantidade total de tokens no vocabulário.
func (t *Tokenizer) VocabSize() int {
	return len(t.TokenToID)
}

// Encode converte uma string de texto em uma fatia de IDs de tokens.
func (t *Tokenizer) Encode(text string) []int {
	rawTokens := SplitIntoTokens(text)
	var ids []int
	for _, tok := range rawTokens {
		if id, found := t.TokenToID[tok]; found {
			ids = append(ids, id)
		} else {
			ids = append(ids, UnkID)
		}
	}
	return ids
}

// Decode converte uma fatia de IDs de tokens de volta para string de texto.
func (t *Tokenizer) Decode(ids []int) string {
	var builder strings.Builder
	for _, id := range ids {
		if token, found := t.IDToToken[id]; found {
			if token == UnkToken || token == PadToken || token == BosToken || token == EosToken {
				continue
			}
			builder.WriteString(token)
		}
	}
	return builder.String()
}

// Save salva o vocabulário em JSON.
func (t *Tokenizer) Save(filePath string) error {
	data, err := json.MarshalIndent(t, "", "  ")
	if err != nil {
		return fmt.Errorf("falha ao serializar vocabulário: %w", err)
	}
	return os.WriteFile(filePath, data, 0644)
}

// Load carregar a estrutura de vocabulário de um arquivo JSON.
func Load(filePath string) (*Tokenizer, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("falha ao ler arquivo de vocabulário: %w", err)
	}

	t := &Tokenizer{
		TokenToID: make(map[string]int),
		IDToToken: make(map[int]string),
	}

	type tempStruct struct {
		TokenToID map[string]int    `json:"token_to_id"`
		IDToToken map[string]string `json:"id_to_token"`
	}

	var temp tempStruct
	if err := json.Unmarshal(data, &temp); err != nil {
		return nil, fmt.Errorf("falha ao deserializar JSON de vocabulário: %w", err)
	}

	t.TokenToID = temp.TokenToID
	for token, id := range temp.TokenToID {
		t.IDToToken[id] = token
	}

	return t, nil
}
