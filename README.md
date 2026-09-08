# Inspeção de Qualidade de Peças de Fundição

Mini-Projeto Avaliativo — Módulo 2 (Machine Learning e Visão Computacional).

Pipeline em Python que une **Visão Clássica (OpenCV)** e **CNN (TensorFlow/Keras)**
para classificar peças de fundição metálica como **OK** ou **Defeituosa**.

> 🚧 Em desenvolvimento.

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

## Dataset

Dataset público *Casting Product Image Data for Quality Inspection*.
Como o arquivo tem ~30 MB, ele **não é versionado** — fica no Google Drive
(link compartilhado no enunciado) e é extraído dentro do notebook.

## Como executar

1. Abra o notebook `notebooks/inspecao_fundicao.ipynb` no Google Colab.
2. Faça upload do `casting_line_512x512.zip` para o seu Drive (MyDrive).
3. Execute as células em ordem.
