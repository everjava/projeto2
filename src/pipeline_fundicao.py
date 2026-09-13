"""
Pipeline de Inspeção de Qualidade de Peças de Fundição
Mini-Projeto Avaliativo — Módulo 2 (ML e Visão Computacional)

Versão em script (equivalente ao notebook `notebooks/inspecao_fundicao.ipynb`).
Executa a EDA clássica com OpenCV e treina uma CNN binária (OK vs. Defeituosa).

Uso:
    python pipeline_fundicao.py --dataset /caminho/para/dataset_pecas
"""

import argparse
import glob
import os

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

ALTURA_IMG, LARGURA_IMG = 224, 224
TAMANHO_LOTE = 32
SEMENTE = 123
AUTOTUNE = tf.data.AUTOTUNE


def localizar_pasta_classes(raiz):
    """Encontra a pasta que contém uma subpasta por classe.

    Procura recursivamente por 'def_front'/'ok_front' e, se houver mais de
    uma candidata (ex.: train e test), prioriza a pasta chamada 'train'.
    """
    candidatos = []
    for caminho, subpastas, _ in os.walk(raiz):
        if any(pasta in subpastas for pasta in ("def_front", "ok_front")):
            candidatos.append(caminho)

    if not candidatos:
        raise FileNotFoundError(
            f"Não encontrei as pastas 'def_front'/'ok_front' a partir de: {raiz}"
        )

    for caminho in candidatos:
        if os.path.basename(caminho).lower() == "train":
            return caminho
    return candidatos[0]


def _tem_classes_diretas(caminho):
    """Verifica se a pasta contém as subpastas de classe diretamente."""
    return any(
        os.path.isdir(os.path.join(caminho, classe))
        for classe in ("def_front", "ok_front")
    )


def resolver_dataset(caminho):
    """Resolve a pasta do dataset a partir de várias localizações prováveis.

    Aceita um caminho explícito (--dataset) ou tenta automaticamente, nesta
    ordem: a raiz do projeto, 'dataset_pecas' dentro dele, o diretório atual
    e a pasta do script. Pastas com as classes diretamente têm prioridade;
    a busca recursiva fica como último recurso.
    """
    if caminho:
        if not os.path.isdir(caminho):
            raise FileNotFoundError(f"A pasta informada em --dataset não existe: {caminho}")
        return localizar_pasta_classes(caminho)

    pasta_script = os.path.dirname(os.path.abspath(__file__))
    raiz_projeto = os.path.dirname(pasta_script)
    candidatos = [
        raiz_projeto,
        os.path.join(raiz_projeto, "dataset_pecas"),
        os.getcwd(),
        pasta_script,
    ]

    # 1) Preferência: pasta que contém as classes diretamente (sem recursão).
    for candidato in candidatos:
        if os.path.isdir(candidato) and _tem_classes_diretas(candidato):
            return candidato

    # 2) Último recurso: busca recursiva, incluindo a pasta acima do projeto.
    for candidato in candidatos + [os.path.dirname(raiz_projeto)]:
        if os.path.isdir(candidato):
            try:
                return localizar_pasta_classes(candidato)
            except FileNotFoundError:
                continue

    raise FileNotFoundError(
        "Não encontrei as pastas 'def_front'/'ok_front'. Coloque-as junto do "
        "projeto (ex.: projeto2/def_front e projeto2/ok_front) ou use "
        "--dataset CAMINHO."
    )


def analise_exploratoria(pasta_raiz, pasta_saida="resultados"):
    """Aplica o pipeline clássico do OpenCV numa amostra de imagens."""
    os.makedirs(pasta_saida, exist_ok=True)
    imagens_def = sorted(glob.glob(os.path.join(pasta_raiz, "def_front", "*")))[:1]
    if not imagens_def:
        print("Sem imagens defeituosas para a EDA.")
        return

    imagem_bgr = cv2.imread(imagens_def[0])
    imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)
    cinza = cv2.cvtColor(imagem_rgb, cv2.COLOR_RGB2GRAY)
    desfoque = cv2.GaussianBlur(cinza, (5, 5), 0)
    _, binaria = cv2.threshold(desfoque, 127, 255, cv2.THRESH_BINARY)
    bordas = cv2.Canny(desfoque, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morfologia = cv2.morphologyEx(bordas, cv2.MORPH_CLOSE, kernel)

    fig, eixos = plt.subplots(1, 5, figsize=(20, 4))
    for eixo, imagem, titulo in zip(
        eixos,
        [imagem_rgb, cinza, desfoque, bordas, morfologia],
        ["Original", "Escala de cinza", "Blur", "Bordas (Canny)", "Morfologia"],
    ):
        eixo.imshow(imagem, cmap="gray")
        eixo.set_title(titulo)
        eixo.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(pasta_saida, "eda_pipeline_classico.png"))
    print("EDA salva em", os.path.join(pasta_saida, "eda_pipeline_classico.png"))


def detectar_classes(pasta_raiz):
    """Lista apenas as pastas de classe.

    Ignora outras pastas do projeto (src, docs, .git...) que, se fossem
    interpretadas como classes, quebrariam o `label_mode='binary'`.
    """
    classes = [
        classe
        for classe in ("def_front", "ok_front")
        if os.path.isdir(os.path.join(pasta_raiz, classe))
    ]
    if len(classes) != 2:
        raise FileNotFoundError(
            f"Esperava as pastas 'def_front' e 'ok_front' em {pasta_raiz}, "
            f"mas encontrei: {classes}"
        )
    return classes


def carregar_dados(pasta_raiz):
    classes = detectar_classes(pasta_raiz)
    dados_treino = tf.keras.utils.image_dataset_from_directory(
        pasta_raiz,
        validation_split=0.2,
        subset="training",
        seed=SEMENTE,
        image_size=(ALTURA_IMG, LARGURA_IMG),
        batch_size=TAMANHO_LOTE,
        label_mode="binary",
        class_names=classes,
    )
    dados_validacao = tf.keras.utils.image_dataset_from_directory(
        pasta_raiz,
        validation_split=0.2,
        subset="validation",
        seed=SEMENTE,
        image_size=(ALTURA_IMG, LARGURA_IMG),
        batch_size=TAMANHO_LOTE,
        label_mode="binary",
        class_names=classes,
    )

    # Otimização: embaralha e pré-carrega lotes para acelerar o treino.
    # Obs.: após essas transformações o dataset deixa de expor `.class_names`,
    # por isso devolvemos `classes` (obtido antes) separadamente.
    #
    # O `.cache()` foi REMOVIDO porque guarda todas as imagens decodificadas na
    # RAM (~750 MB para 1300 imagens 224x224x3 em float32). Com pouca RAM livre,
    # o sistema entra em swap e o treino fica MAIS LENTO. Em uma máquina com RAM
    # sobrando, basta reativá-lo:
    #   dados_treino = dados_treino.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    #   dados_validacao = dados_validacao.cache().prefetch(buffer_size=AUTOTUNE)
    dados_treino = dados_treino.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    dados_validacao = dados_validacao.prefetch(buffer_size=AUTOTUNE)
    return dados_treino, dados_validacao, classes


def construir_modelo():
    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomBrightness(0.2),
            layers.RandomContrast(0.2),
        ],
        name="data_augmentation",
    )

    modelo = models.Sequential(
        [
            layers.Input(shape=(ALTURA_IMG, LARGURA_IMG, 3)),
            data_augmentation,
            layers.Rescaling(1.0 / 255),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="inspetor_fundicao",
    )
    modelo.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )
    return modelo


def plotar_historico(historico, pasta_saida="resultados"):
    os.makedirs(pasta_saida, exist_ok=True)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(historico.history["accuracy"], label="Treino")
    plt.plot(historico.history["val_accuracy"], label="Validação")
    plt.title("Acurácia por Época")
    plt.xlabel("Época")
    plt.ylabel("Acurácia")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(historico.history["loss"], label="Treino")
    plt.plot(historico.history["val_loss"], label="Validação")
    plt.title("Função de Perda por Época")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    caminho = os.path.join(pasta_saida, "historico_treinamento.png")
    plt.savefig(caminho)
    print("Gráficos salvos em", caminho)


def main():
    parser = argparse.ArgumentParser(description="Pipeline de inspeção de peças.")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Pasta raiz do dataset. Se omitido, detecta automaticamente.",
    )
    parser.add_argument("--epocas", type=int, default=20)
    args = parser.parse_args()

    pasta_raiz = resolver_dataset(args.dataset)
    print("Pasta de classes:", pasta_raiz)

    analise_exploratoria(pasta_raiz)

    dados_treino, dados_validacao, nomes_classes = carregar_dados(pasta_raiz)
    print("Classes:", nomes_classes)

    modelo = construir_modelo()
    modelo.summary()

    parada_antecipada = EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )
    historico = modelo.fit(
        dados_treino,
        validation_data=dados_validacao,
        epochs=args.epocas,
        callbacks=[parada_antecipada],
        verbose=1,
    )

    plotar_historico(historico)
    modelo.save("resultados/inspetor_fundicao.keras")


if __name__ == "__main__":
    main()
