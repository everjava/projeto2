# Inspeção de Qualidade de Peças de Fundição

Mini-Projeto Avaliativo — Módulo 2 (Machine Learning e Visão Computacional).

Pipeline em Python que une **Visão Clássica (OpenCV)** e **CNN (TensorFlow/Keras)**
para classificar peças de fundição metálica como **OK** ou **Defeituosa**, simulando
a inspeção visual automatizada da Indústria 4.0.

## Contextualização

Antes de treinar uma rede neural, usamos ferramentas de processamento clássico
(OpenCV) para realizar uma análise exploratória: limpar ruídos, destacar bordas e
compreender as características físicas do defeito. Somente depois aplicamos uma
**Rede Neural Convolucional (CNN)** para automatizar a classificação em larga escala.

## Dataset

Dataset público *Casting Product Image Data for Quality Inspection*
(≈1300 imagens: 519 OK / 781 defeituosas).

Como o arquivo tem ~30 MB, ele **não é versionado** neste repositório. Ele fica no
Google Drive (link compartilhado no enunciado do projeto) e é extraído dentro do
notebook via `zipfile`.


## Estrutura do repositório

```
projeto2/
├── ok_front/     # peças sem defeito
└── def_front/    # peças com defeito
├── src/
│   └── pipeline_fundicao.py      # Versão script do pipeline
├── requirements.txt
├── .gitignore
└── README.md
```

## Como executar

### Opção 1 — Google Colab (recomendado)

1. Abra `notebooks/inspecao_fundicao.ipynb` no Google Colab.
2. Faça upload do `casting_line_512x512.zip` para o seu Drive (MyDrive).
3. Execute as células em ordem.

### Opção 2 — Script local

```bash
pip install -r requirements.txt
python src/pipeline_fundicao.py --dataset /caminho/para/dataset_pecas --epocas 20

ou
projeto2> python src/pipeline_fundicao.py
#com os diretórios ok_front e def_front dentro de projeto2

```

## Pipeline

### 1. Análise Exploratória (OpenCV) — didática, não entra no treino

- **Escala de cinza** (`cvtColor BGR2GRAY`): reduz a informação de cor.
- **Gaussian Blur (5x5)**: suaviza ruído industrial (reflexos, granulado).
- **Limiarização** (`threshold 127/255`): separa objeto do fundo.
- **Canny (50, 150)**: destaca as descontinuidades (trincas/ranhuras).
- **Morfologia** (`MORPH_CLOSE`, elipse 3x3): limpa ruído e conecta bordas.

### 2. Classificação (CNN)

- **Ingestão:** `image_dataset_from_directory` (Treino 80% / Validação 20%, `seed=123`, 224x224).
- **Data Augmentation:** geométrico (`RandomFlip`, `RandomRotation`, `RandomZoom`) e
  luminoso (`RandomBrightness`, `RandomContrast`) — simula as variações da esteira.
- **Arquitetura:** `Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(128) → MaxPool
  → Flatten → Dense(128, relu) → Dropout(0.5) → Dense(1, sigmoid)`.
- **Compilação:** `adam` + `binary_crossentropy` + `accuracy`.
- **Treino:** `EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)`.

### 3. Auditoria

Gráficos de **Acurácia** e **Loss** (Treino vs. Validação) para diagnosticar
overfitting ("boca do jacaré") e matriz de confusão (bônus).

## Resultados

Treinamento executado com `image_dataset_from_directory` (1040 imagens de treino /
260 de validação). O `EarlyStopping` disparou na época 17 e restaurou os pesos da
melhor época (14).

| Métrica | Valor |
|---|---|
| Melhor época | 14 |
| Acurácia de treino (época 14) | 0,8279 |
| Acurácia de validação (época 14) | **0,8923** |
| Loss de treino (época 14) | 0,3815 |
| Loss de validação (época 14) | **0,2542** |
| Épocas treinadas | 17 (de 20) |

**Diagnóstico de overfitting:** as curvas de treino e validação andaram próximas
durante quase todo o treino. A `val_loss` atingiu o mínimo na época 14 (~0,25) e
depois subiu (0,33 na época 17) enquanto a `loss` de treino continuou caindo —
início de overfitting, corretamente contido pelo `EarlyStopping`
(`restore_best_weights=True`). Resultado **saudável**, com ~89% de acurácia de
validação (acima do mínimo de 60% exigido).

**Artefatos gerados** (pasta `resultados/`, não versionada):
- `eda_pipeline_classico.png` — etapas do processamento clássico (OpenCV)
- `historico_treinamento.png` — curvas de Acurácia e Loss (Treino vs. Validação)
- `inspetor_fundicao.keras` — modelo treinado

## Entregáveis

- Código-fonte (notebook + script).
- `README.md` (este arquivo).
- Vídeo técnico de até 5 minutos.

## Tecnologias

Python, OpenCV, TensorFlow/Keras, NumPy, Matplotlib, Scikit-learn, Seaborn.
