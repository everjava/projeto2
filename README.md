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

Estrutura esperada (o notebook localiza automaticamente a pasta de classes):

```
dataset_pecas/
├── ok_front/     # peças sem defeito
└── def_front/    # peças com defeito
```

## Estrutura do repositório

```
projeto2/
├── notebooks/
│   └── inspecao_fundicao.ipynb   # Notebook principal (EDA + CNN)
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

| Métrica | Valor |
|---|---|
| Acurácia de treino | _(preencher após executar)_ |
| Acurácia de validação | _(preencher após executar)_ |
| Épocas treinadas | _(preencher após executar)_ |

**Diagnóstico de overfitting:** _(preencher após analisar os gráficos — as curvas de
treino e validação andaram lado a lado (aprendizado saudável) ou a `val_loss`
estagnou/subiu enquanto a `loss` caiu (overfitting)?)_

## Entregáveis

- Código-fonte (notebook + script).
- `README.md` (este arquivo).
- Vídeo técnico de até 5 minutos.

## Tecnologias

Python, OpenCV, TensorFlow/Keras, NumPy, Matplotlib, Scikit-learn, Seaborn.
