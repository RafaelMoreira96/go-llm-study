package model

import (
	"encoding/json"
	"fmt"
	"os"

	"go-llm/pkg/matrix"
)

// SerializableGPT é a estrutura intermediária para exportar e carregar o modelo em JSON.
type SerializableGPT struct {
	Config Config `json:"config"`

	TokenEmbeddingData []float64 `json:"token_embedding_data"`
	PosEmbeddingData   []float64 `json:"pos_embedding_data"`

	HeadWData []float64 `json:"head_w_data"`
	HeadBData []float64 `json:"head_b_data"`

	// Dados dos blocos
	BlocksData []SerializableBlock `json:"blocks_data"`
}

type SerializableBlock struct {
	WqData []float64 `json:"wq_data"`
	WkData []float64 `json:"wk_data"`
	WvData []float64 `json:"wv_data"`
	WoData []float64 `json:"wo_data"`
	BqData []float64 `json:"bq_data"`
	BkData []float64 `json:"bk_data"`
	BvData []float64 `json:"bv_data"`
	BoData []float64 `json:"bo_data"`

	W1Data []float64 `json:"w1_data"`
	B1Data []float64 `json:"b1_data"`
	W2Data []float64 `json:"w2_data"`
	B2Data []float64 `json:"b2_data"`

	LN1Gamma []float64 `json:"ln1_gamma"`
	LN1Beta  []float64 `json:"ln1_beta"`
	LN2Gamma []float64 `json:"ln2_gamma"`
	LN2Beta  []float64 `json:"ln2_beta"`
}

// SaveWeights exporta as configurações e pesos do modelo em um arquivo JSON.
func (gpt *GPT) SaveWeights(filePath string) error {
	s := SerializableGPT{
		Config:             gpt.Config,
		TokenEmbeddingData: gpt.TokenEmbedding.Data,
		PosEmbeddingData:   gpt.PosEmbedding.Data,
		HeadWData:          gpt.HeadW.Data,
		HeadBData:          gpt.HeadB,
	}

	for _, b := range gpt.Blocks {
		sb := SerializableBlock{
			WqData:   b.Attn.Wq.Data,
			WkData:   b.Attn.Wk.Data,
			WvData:   b.Attn.Wv.Data,
			WoData:   b.Attn.Wo.Data,
			BqData:   b.Attn.Bq,
			BkData:   b.Attn.Bk,
			BvData:   b.Attn.Bv,
			BoData:   b.Attn.Bo,
			W1Data:   b.FFN.W1.Data,
			B1Data:   b.FFN.B1,
			W2Data:   b.FFN.W2.Data,
			B2Data:   b.FFN.B2,
			LN1Gamma: b.LN1Gamma,
			LN1Beta:  b.LN1Beta,
			LN2Gamma: b.LN2Gamma,
			LN2Beta:  b.LN2Beta,
		}
		s.BlocksData = append(s.BlocksData, sb)
	}

	data, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return fmt.Errorf("falha ao serializar pesos do modelo: %w", err)
	}

	return os.WriteFile(filePath, data, 0644)
}

// LoadGPT carregar o modelo e restore seus pesos a partir de um arquivo JSON.
func LoadGPT(filePath string) (*GPT, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("falha ao ler arquivo de pesos do modelo: %w", err)
	}

	var s SerializableGPT
	if err := json.Unmarshal(data, &s); err != nil {
		return nil, fmt.Errorf("falha ao deserializar JSON dos pesos do modelo: %w", err)
	}

	gpt := NewGPT(s.Config)
	gpt.TokenEmbedding = matrix.NewWithData(s.Config.VocabSize, s.Config.DModel, s.TokenEmbeddingData)
	gpt.PosEmbedding = matrix.NewWithData(s.Config.SeqLen, s.Config.DModel, s.PosEmbeddingData)
	gpt.HeadW = matrix.NewWithData(s.Config.DModel, s.Config.VocabSize, s.HeadWData)
	gpt.HeadB = s.HeadBData

	for i, sb := range s.BlocksData {
		b := gpt.Blocks[i]
		b.Attn.Wq = matrix.NewWithData(s.Config.DModel, s.Config.DModel, sb.WqData)
		b.Attn.Wk = matrix.NewWithData(s.Config.DModel, s.Config.DModel, sb.WkData)
		b.Attn.Wv = matrix.NewWithData(s.Config.DModel, s.Config.DModel, sb.WvData)
		b.Attn.Wo = matrix.NewWithData(s.Config.DModel, s.Config.DModel, sb.WoData)
		b.Attn.Bq = sb.BqData
		b.Attn.Bk = sb.BkData
		b.Attn.Bv = sb.BvData
		b.Attn.Bo = sb.BoData

		dFF := 4 * s.Config.DModel
		b.FFN.W1 = matrix.NewWithData(s.Config.DModel, dFF, sb.W1Data)
		b.FFN.B1 = sb.B1Data
		b.FFN.W2 = matrix.NewWithData(dFF, s.Config.DModel, sb.W2Data)
		b.FFN.B2 = sb.B2Data

		b.LN1Gamma = sb.LN1Gamma
		b.LN1Beta = sb.LN1Beta
		b.LN2Gamma = sb.LN2Gamma
		b.LN2Beta = sb.LN2Beta
	}

	return gpt, nil
}
