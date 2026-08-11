# 🧠 `go-llm`: Modelo de Linguagem (LLM / GPT) do Zero em Go

`go-llm` é uma implementação de um **Modelo de Linguagem de Grande Porte (LLM)** baseado na arquitetura **Decoder-only Transformer (estilo GPT-2/nanoGPT)** desenvolvida **100% do zero em Go puro**, sem dependências externas de frameworks de Inteligência Artificial.

---

## 🌟 Destaques do Projeto

- **Motor de Tensores em Go (`pkg/matrix`)**: Matrizes 2D/3D com suporte a `MatMul`, `Softmax` numericamente estável, `GELU`, `LayerNorm`, `Transpose` e aplicação de **Máscara Causal**.
- **Tokenizer Customizado (`pkg/tokenizer`)**: Codificação de texto em IDs de tokens, construção automática de vocabulário e serialização JSON.
- **Arquitetura Transformer (`pkg/model`)**:
  - Embeddings de Token e Embeddings Posicionais.
  - **Causal Multi-Head Self-Attention**: Projeções Q, K, V e divisão em múltiplas cabeças com produto escalar escalado e máscara autoregressiva.
  - **Feed-Forward Networks (FFN)** com ativações GELU.
  - Conexões residuais (*Add & Norm*) com Layer Normalization.
- **Amostragem de Tokens (`pkg/sampler`)**: Suporte a *Greedy Sampling*, *Temperature Scaling* e *Top-K Sampling*.
- **CLI de Treinamento (`cmd/train`)**: Treina a LLM em qualquer arquivo de texto e salva os pesos (`weights.json`) e vocabulário (`vocab.json`).
- **CLI de Geração (`cmd/generate`)**: Geração auto-regressiva de texto com streaming em tempo real no terminal e modo interativo via prompt.

---

## 📁 Estrutura de Arquivos

```
go-llm/
├── README.md                 # Documentação do projeto
├── go.mod                    # Módulo Go
├── data/
│   └── input.txt             # Dataset de texto em português para treinamento
├── cmd/
│   ├── train/
│   │   └── main.go           # Script para treinar o modelo e salvar checkpoints
│   └── generate/
│       └── main.go           # CLI para geração de texto e chat interativo
└── pkg/
    ├── matrix/               # Álgebra linear, tensores e funções de ativação
    ├── tokenizer/            # Mapeamento de caracteres/subwords <-> IDs numéricos
    ├── model/                # Arquitetura GPT/Transformer e persistência de pesos
    └── sampler/              # Algoritmos de amostragem de tokens (Temperature, Top-K)
```

---

## 🚀 Como Usar

### 1. Pré-requisitos
Apenas o **Go 1.22+** instalado na sua máquina!

### 2. Executar os Testes Unitários
Para verificar a integridade da álgebra de tensores, tokenizer, attention e modelo:
```bash
go test ./...
```

---

### 3. Treinar seu Próprio Modelo (`cmd/train`)

Você pode treinar a LLM usando o dataset fornecido em `data/input.txt` ou passar seu próprio arquivo `.txt`:

```bash
go run ./cmd/train -input data/input.txt -epochs 100 -lr 0.04
```

#### Parâmetros de Treinamento:
- `-input`: Caminho para o arquivo de texto de entrada (padrão: `data/input.txt`).
- `-epochs`: Número de épocas de treinamento (padrão: `80`).
- `-lr`: Taxa de aprendizado / Learning Rate (padrão: `0.03`).
- `-d-model`: Dimensão das representações de embedding (padrão: `64`).
- `-layers`: Número de camadas Transformer (padrão: `2`).
- `-heads`: Número de cabeças de atenção (padrão: `4`).
- `-seq-len`: Comprimento da janela de contexto (padrão: `32`).

---

### 4. Gerar Texto Auto-Regressivo (`cmd/generate`)

Após o treinamento, gere texto a partir de um prompt inicial:

```bash
go run ./cmd/generate -prompt "Go é" -temp 0.7 -top-k 5 -max-tokens 100
```

---

### 5. Modo Interativo (Chat no Terminal)

Inicie uma sessão interativa no terminal onde você pode enviar prompts continuamente:

```bash
go run ./cmd/generate -interactive -temp 0.8
```

Exemplo no terminal:
```text
=========================================================
🤖 LLM em Go - Gerador de Texto Auto-Regressivo
⚙️ Parâmetros: temp=0.80 | top_k=5 | max_tokens=120
=========================================================
💬 Modo Interativo ativado. Digite seu prompt ou 'sair' para encerrar.

👉 Digite seu prompt: O futuro da inteligência artificial
🤖 Geração do Modelo: O futuro da inteligência artificial é construído com linguagens de alta performance como Go...
```

---

## 📐 Matemática da Arquitetura

1. **Atenção por Produto Escalar Escalado (Scaled Dot-Product Attention)**:
   $$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$
   onde $M$ é a Máscara Causal que impede o modelo de acessar tokens futuros ($M_{i,j} = -\infty$ para $j > i$).

2. **GELU (Gaussian Error Linear Unit)**:
   $$\text{GELU}(x) \approx 0.5x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \left(x + 0.044715 x^3\right)\right)\right)$$

3. **Softmax Numericamente Estável**:
   $$\text{Softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$$

---

## 📜 Licença

Licença MIT. Sinta-se livre para usar, modificar e expandir para seus projetos de IA em Go!
