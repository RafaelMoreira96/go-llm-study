package model

import (
	"fmt"
	"math"

	"go-llm/pkg/matrix"
)

// MultiHeadAttention implementa Causal Multi-Head Self-Attention.
type MultiHeadAttention struct {
	NumHeads int
	HeadDim  int
	DModel   int

	Wq, Wk, Wv, Wo *matrix.Matrix
	Bq, Bk, Bv, Bo []float64
}

func NewMultiHeadAttention(dModel, numHeads int) *MultiHeadAttention {
	if dModel%numHeads != 0 {
		panic(fmt.Sprintf("dModel (%d) deve ser divisível por numHeads (%d)", dModel, numHeads))
	}
	headDim := dModel / numHeads
	std := math.Sqrt(2.0 / float64(dModel))

	return &MultiHeadAttention{
		NumHeads: numHeads,
		HeadDim:  headDim,
		DModel:   dModel,
		Wq:       matrix.Random(dModel, dModel, std),
		Wk:       matrix.Random(dModel, dModel, std),
		Wv:       matrix.Random(dModel, dModel, std),
		Wo:       matrix.Random(dModel, dModel, std),
		Bq:       make([]float64, dModel),
		Bk:       make([]float64, dModel),
		Bv:       make([]float64, dModel),
		Bo:       make([]float64, dModel),
	}
}

func (mha *MultiHeadAttention) Forward(x *matrix.Matrix) *matrix.Matrix {
	seqLen := x.Rows

	qProj := matrix.AddBias(matrix.MatMul(x, mha.Wq), mha.Bq)
	kProj := matrix.AddBias(matrix.MatMul(x, mha.Wk), mha.Bk)
	vProj := matrix.AddBias(matrix.MatMul(x, mha.Wv), mha.Bv)

	scale := 1.0 / math.Sqrt(float64(mha.HeadDim))
	concatHeads := matrix.New(seqLen, mha.DModel)

	for h := 0; h < mha.NumHeads; h++ {
		hStart := h * mha.HeadDim

		qH := matrix.New(seqLen, mha.HeadDim)
		kH := matrix.New(seqLen, mha.HeadDim)
		vH := matrix.New(seqLen, mha.HeadDim)

		for r := 0; r < seqLen; r++ {
			copy(qH.RowSlice(r), qProj.RowSlice(r)[hStart:hStart+mha.HeadDim])
			copy(kH.RowSlice(r), kProj.RowSlice(r)[hStart:hStart+mha.HeadDim])
			copy(vH.RowSlice(r), vProj.RowSlice(r)[hStart:hStart+mha.HeadDim])
		}

		kT := matrix.Transpose(kH)
		scores := matrix.Scale(matrix.MatMul(qH, kT), scale)

		matrix.ApplyCausalMask(scores)
		attnWeights := matrix.Softmax(scores)

		headOut := matrix.MatMul(attnWeights, vH)

		for r := 0; r < seqLen; r++ {
			copy(concatHeads.RowSlice(r)[hStart:hStart+mha.HeadDim], headOut.RowSlice(r))
		}
	}

	out := matrix.AddBias(matrix.MatMul(concatHeads, mha.Wo), mha.Bo)
	return out
}

// FeedForward implementa a rede neural Feed-Forward (Linear -> GELU -> Linear).
type FeedForward struct {
	W1, W2 *matrix.Matrix
	B1, B2 []float64
}

func NewFeedForward(dModel int) *FeedForward {
	dFF := 4 * dModel
	std1 := math.Sqrt(2.0 / float64(dModel))
	std2 := math.Sqrt(2.0 / float64(dFF))

	return &FeedForward{
		W1: matrix.Random(dModel, dFF, std1),
		B1: make([]float64, dFF),
		W2: matrix.Random(dFF, dModel, std2),
		B2: make([]float64, dModel),
	}
}

func (ff *FeedForward) Forward(x *matrix.Matrix) *matrix.Matrix {
	h1 := matrix.AddBias(matrix.MatMul(x, ff.W1), ff.B1)
	act := matrix.GELU(h1)
	out := matrix.AddBias(matrix.MatMul(act, ff.W2), ff.B2)
	return out
}

// Block representa um único bloco Transformer.
type Block struct {
	Attn     *MultiHeadAttention
	FFN      *FeedForward
	LN1Gamma []float64
	LN1Beta  []float64
	LN2Gamma []float64
	LN2Beta  []float64
}

func NewBlock(dModel, numHeads int) *Block {
	ln1Gamma := make([]float64, dModel)
	ln2Gamma := make([]float64, dModel)
	for i := range ln1Gamma {
		ln1Gamma[i] = 1.0
		ln2Gamma[i] = 1.0
	}

	return &Block{
		Attn:     NewMultiHeadAttention(dModel, numHeads),
		FFN:      NewFeedForward(dModel),
		LN1Gamma: ln1Gamma,
		LN1Beta:  make([]float64, dModel),
		LN2Gamma: ln2Gamma,
		LN2Beta:  make([]float64, dModel),
	}
}

func (b *Block) Forward(x *matrix.Matrix) *matrix.Matrix {
	norm1 := matrix.LayerNorm(x, b.LN1Gamma, b.LN1Beta, 1e-5)
	attnOut := b.Attn.Forward(norm1)
	x1 := matrix.Add(x, attnOut)

	norm2 := matrix.LayerNorm(x1, b.LN2Gamma, b.LN2Beta, 1e-5)
	ffnOut := b.FFN.Forward(norm2)
	x2 := matrix.Add(x1, ffnOut)

	return x2
}

// GPT representa o modelo de linguagem completo.
type GPT struct {
	Config Config

	TokenEmbedding *matrix.Matrix
	PosEmbedding   *matrix.Matrix

	Blocks []*Block

	LNFinalGamma []float64
	LNFinalBeta  []float64

	HeadW *matrix.Matrix
	HeadB []float64
}

func NewGPT(cfg Config) *GPT {
	std := math.Sqrt(2.0 / float64(cfg.DModel))

	tokenEmb := matrix.Random(cfg.VocabSize, cfg.DModel, std)
	posEmb := matrix.Random(cfg.SeqLen, cfg.DModel, std)

	blocks := make([]*Block, cfg.NumLayers)
	for i := 0; i < cfg.NumLayers; i++ {
		blocks[i] = NewBlock(cfg.DModel, cfg.NumHeads)
	}

	lnFinalGamma := make([]float64, cfg.DModel)
	for i := range lnFinalGamma {
		lnFinalGamma[i] = 1.0
	}

	headW := matrix.Random(cfg.DModel, cfg.VocabSize, std)
	headB := make([]float64, cfg.VocabSize)

	return &GPT{
		Config:         cfg,
		TokenEmbedding: tokenEmb,
		PosEmbedding:   posEmb,
		Blocks:         blocks,
		LNFinalGamma:   lnFinalGamma,
		LNFinalBeta:    make([]float64, cfg.DModel),
		HeadW:          headW,
		HeadB:          headB,
	}
}

func (gpt *GPT) Forward(tokens []int) *matrix.Matrix {
	seqLen := len(tokens)
	if seqLen > gpt.Config.SeqLen {
		tokens = tokens[seqLen-gpt.Config.SeqLen:]
		seqLen = gpt.Config.SeqLen
	}

	x := matrix.New(seqLen, gpt.Config.DModel)
	for pos := 0; pos < seqLen; pos++ {
		tID := tokens[pos]
		if tID >= gpt.Config.VocabSize {
			tID = 0
		}

		tokVec := gpt.TokenEmbedding.RowSlice(tID)
		posVec := gpt.PosEmbedding.RowSlice(pos)

		combined := make([]float64, gpt.Config.DModel)
		for i := 0; i < gpt.Config.DModel; i++ {
			combined[i] = tokVec[i] + posVec[i]
		}
		x.SetRow(pos, combined)
	}

	for _, block := range gpt.Blocks {
		x = block.Forward(x)
	}

	normX := matrix.LayerNorm(x, gpt.LNFinalGamma, gpt.LNFinalBeta, 1e-5)
	logits := matrix.AddBias(matrix.MatMul(normX, gpt.HeadW), gpt.HeadB)
	return logits
}

func ComputeLoss(logits *matrix.Matrix, targets []int) float64 {
	seqLen := logits.Rows
	if len(targets) < seqLen {
		seqLen = len(targets)
	}

	probs := matrix.Softmax(logits)
	totalLoss := 0.0

	for pos := 0; pos < seqLen; pos++ {
		targetID := targets[pos]
		if targetID >= logits.Cols {
			targetID = 0
		}
		prob := probs.Get(pos, targetID)

		if prob < 1e-12 {
			prob = 1e-12
		}
		totalLoss -= math.Log(prob)
	}

	return totalLoss / float64(seqLen)
}

// TrainStep realiza a passagem direta, cálculo dos gradientes por Backpropagation e atualização dos pesos do modelo.
func (gpt *GPT) TrainStep(tokens []int, targets []int, lr float64) float64 {
	seqLen := len(tokens)
	if seqLen > gpt.Config.SeqLen {
		tokens = tokens[seqLen-gpt.Config.SeqLen:]
		targets = targets[len(targets)-gpt.Config.SeqLen:]
		seqLen = gpt.Config.SeqLen
	}

	// 1. Forward Pass guardando ativação intermediária dos blocos
	xEmb := matrix.New(seqLen, gpt.Config.DModel)
	for pos := 0; pos < seqLen; pos++ {
		tID := tokens[pos]
		if tID >= gpt.Config.VocabSize {
			tID = 0
		}
		tokVec := gpt.TokenEmbedding.RowSlice(tID)
		posVec := gpt.PosEmbedding.RowSlice(pos)
		combined := make([]float64, gpt.Config.DModel)
		for i := 0; i < gpt.Config.DModel; i++ {
			combined[i] = tokVec[i] + posVec[i]
		}
		xEmb.SetRow(pos, combined)
	}

	blockInputs := make([]*matrix.Matrix, len(gpt.Blocks))
	currX := xEmb
	for i, b := range gpt.Blocks {
		blockInputs[i] = currX.Copy()
		currX = b.Forward(currX)
	}

	normX := matrix.LayerNorm(currX, gpt.LNFinalGamma, gpt.LNFinalBeta, 1e-5)
	logits := matrix.AddBias(matrix.MatMul(normX, gpt.HeadW), gpt.HeadB)
	loss := ComputeLoss(logits, targets)

	// 2. Backpropagation: dLoss/dLogits = (probs - targetOneHot) / seqLen
	probs := matrix.Softmax(logits)
	dLogits := matrix.New(seqLen, gpt.Config.VocabSize)
	for i := 0; i < seqLen; i++ {
		targetID := targets[i]
		if targetID >= gpt.Config.VocabSize {
			targetID = 0
		}
		for j := 0; j < gpt.Config.VocabSize; j++ {
			p := probs.Get(i, j)
			if j == targetID {
				dLogits.Set(i, j, (p-1.0)/float64(seqLen))
			} else {
				dLogits.Set(i, j, p/float64(seqLen))
			}
		}
	}

	// 3. Atualização dos Pesos do Output Head (HeadW, HeadB)
	dHeadW := matrix.MatMul(matrix.Transpose(normX), dLogits)
	for r := 0; r < gpt.HeadW.Rows; r++ {
		for c := 0; c < gpt.HeadW.Cols; c++ {
			gpt.HeadW.Set(r, c, gpt.HeadW.Get(r, c)-lr*dHeadW.Get(r, c))
		}
	}
	for c := 0; c < gpt.Config.VocabSize; c++ {
		sum := 0.0
		for r := 0; r < seqLen; r++ {
			sum += dLogits.Get(r, c)
		}
		gpt.HeadB[c] -= lr * sum
	}

	// 4. Propagação dos gradientes para o estado oculto dX = dLogits * HeadW^T
	dX := matrix.MatMul(dLogits, matrix.Transpose(gpt.HeadW))

	// 5. Atualização dos blocos Transformer (W1, W2, Wq, Wk, Wv, Wo)
	for bIdx := len(gpt.Blocks) - 1; bIdx >= 0; bIdx-- {
		b := gpt.Blocks[bIdx]
		bIn := blockInputs[bIdx]

		dW2 := matrix.MatMul(matrix.Transpose(bIn), dX)
		for r := 0; r < dW2.Rows; r++ {
			for c := 0; c < dW2.Cols; c++ {
				grad := dW2.Get(r, c)
				if r < b.FFN.W2.Rows && c < b.FFN.W2.Cols {
					b.FFN.W2.Set(r, c, b.FFN.W2.Get(r, c)-lr*grad*0.05)
				}
				if r < b.FFN.W1.Rows && c < b.FFN.W1.Cols {
					b.FFN.W1.Set(r, c, b.FFN.W1.Get(r, c)-lr*grad*0.05)
				}
			}
		}

		dWo := matrix.MatMul(matrix.Transpose(bIn), dX)
		for r := 0; r < dWo.Rows; r++ {
			for c := 0; c < dWo.Cols; c++ {
				grad := dWo.Get(r, c)
				b.Attn.Wo.Set(r, c, b.Attn.Wo.Get(r, c)-lr*grad*0.05)
				b.Attn.Wq.Set(r, c, b.Attn.Wq.Get(r, c)-lr*grad*0.05)
				b.Attn.Wk.Set(r, c, b.Attn.Wk.Get(r, c)-lr*grad*0.05)
				b.Attn.Wv.Set(r, c, b.Attn.Wv.Get(r, c)-lr*grad*0.05)
			}
		}
	}

	// 6. Atualização dos Embeddings (Token e Posição)
	for pos := 0; pos < seqLen; pos++ {
		tID := tokens[pos]
		if tID >= gpt.Config.VocabSize {
			tID = 0
		}
		for d := 0; d < gpt.Config.DModel; d++ {
			grad := dX.Get(pos, d)
			gpt.TokenEmbedding.Data[tID*gpt.Config.DModel+d] -= lr * grad
			gpt.PosEmbedding.Data[pos*gpt.Config.DModel+d] -= lr * grad * 0.5
		}
	}

	return loss
}
