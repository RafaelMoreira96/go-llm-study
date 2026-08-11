package tokenizer

import (
	"path/filepath"
	"testing"
)

func TestTokenizerBuildEncodeDecode(t *testing.T) {
	tok := New()
	text := "Olá , modelo LLM em Go !"
	tok.BuildFromText(text)

	if tok.VocabSize() <= 4 {
		t.Fatalf("esperado vocabulário > 4, obteve %d", tok.VocabSize())
	}

	encoded := tok.Encode(text)
	decoded := tok.Decode(encoded)
	if decoded != text {
		t.Errorf("esperado '%s', obteve '%s'", text, decoded)
	}
}

func TestTokenizerSaveLoad(t *testing.T) {
	tok := New()
	tok.BuildFromText("Teste de serialização de vocabulário Go LLM")

	tmpDir := t.TempDir()
	filePath := filepath.Join(tmpDir, "vocab.json")

	err := tok.Save(filePath)
	if err != nil {
		t.Fatalf("falha ao salvar vocabulário: %v", err)
	}

	loadedTok, err := Load(filePath)
	if err != nil {
		t.Fatalf("falha ao carregar vocabulário: %v", err)
	}

	if loadedTok.VocabSize() != tok.VocabSize() {
		t.Errorf("tamanho do vocabulário carregado (%d) difere do original (%d)", loadedTok.VocabSize(), tok.VocabSize())
	}

	sample := "Teste"
	if loadedTok.Decode(loadedTok.Encode(sample)) != sample {
		t.Errorf("falha no encode/decode com vocabulário carregado")
	}
}
