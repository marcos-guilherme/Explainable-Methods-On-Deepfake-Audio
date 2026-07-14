# syntax=docker/dockerfile:1.7
#
# Imagem autossuficiente para rodar os notebooks de deteccao de deepfake (XLS-R + fairseq)
# em GPU. Todas as dependencias sao instaladas no build; nao ha nenhum passo de instalacao
# por fora. A GPU e fornecida pelo host via nvidia-container-toolkit (docker run --gpus all).
# Os wheels do PyTorch cu126 ja trazem as libs CUDA, entao a base pode ser uma imagem Python.
# CUDA 12.6 (nao 13.0): o build cu130 exige driver >= r580; drivers r5xx (ex.: 550/CUDA 12.4)
# rodam o runtime cu126 via minor-version compatibility (mesmo major 12), sem atualizar o host.
# Python 3.12: o requirements.txt foi congelado num ambiente 3.12 (ex.: shap==0.52.0 exige
# Python>=3.12), entao alinhamos a imagem a essa versao validada.
FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    IN_DOCKER=1 \
    HF_HOME=/workspace/.hf-cache \
    HF_HUB_DISABLE_TELEMETRY=1

# Dependencias de sistema:
#   build-essential + git : compilar o fairseq 0.12.2 (extensoes Cython/C++)
#   ffmpeg                : usado pelo pydub (degradacao MP3 no notebook zero-shot)
#   libsndfile1           : backend do soundfile para decodificar audio
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        ffmpeg \
        libsndfile1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copia so o requirements primeiro, para aproveitar o cache de camadas do Docker.
COPY requirements.txt /workspace/requirements.txt

# Instalacao em ordem controlada (o par torch cu126 + fairseq 0.12.2 e a parte fragil).
# 1) pip<24.1: os wheels do omegaconf 2.0.6 (dep do fairseq) tem metadata legada que o
#    resolvedor do pip>=24.1 recusa (ResolutionImpossible).
RUN python -m pip install --upgrade "pip<24.1" "setuptools<70" wheel

# 2) torch/torchaudio do indice cu126 (nao resolvivel a partir do PyPI puro).
RUN pip install \
        torch==2.13.0 \
        torchaudio==2.11.0 \
        --index-url https://download.pytorch.org/whl/cu126

# 3) build-deps do fairseq, com numpy<2 JA presente (o fairseq nao compila com numpy>=2).
RUN pip install \
        numpy==1.26.4 \
        Cython==3.2.8 \
        omegaconf==2.0.6 \
        hydra-core==1.0.7

# 4) fairseq a partir do git na tag v0.12.2, sem build-isolation (reaproveita
#    numpy/cython/torch ja instalados). O sdist do fairseq==0.12.2 no PyPI omite
#    fairseq/clib/libbase/balanced_assignment.cpp, o que quebra a compilacao da
#    extensao libbase ("No such file or directory"); o repositorio git tem o arquivo.
RUN pip install --no-build-isolation \
        "fairseq @ git+https://github.com/facebookresearch/fairseq.git@v0.12.2"

# 5) Instala o ambiente CONGELADO exatamente como pinado, SEM re-resolver dependencias.
#    O requirements.txt fixa numpy==1.26.4 (exigido pelo fairseq, que nao compila com numpy>=2),
#    mas o shap==0.52.0 declara numpy>=2 no metadata -> a resolucao estrita e impossivel.
#    Como no Colab, o shap roda bem em numpy 1.26.4 (o requisito >=2 e conservador), entao:
#      a) instala-se o freeze com --no-deps (mantem as versoes pinadas, sem mexer no numpy);
#      b) preenche-se as deps transitivas que faltam ao shap (slicer/numba/cloudpickle/llvmlite),
#         usando o requirements.txt como constraints para o numpy NAO subir para 2.
#    fairseq/torch/torchaudio saem da lista (fairseq veio do git no passo 4; torch do indice cu126).
#    jupyterlab/papermill/ipywidgets (fora do freeze) idem, com deps e constraints.
RUN grep -ivE '^(fairseq|torch|torchaudio)==' requirements.txt > /tmp/req.frozen.txt \
    && pip install --no-deps -r /tmp/req.frozen.txt \
    && pip install -c requirements.txt slicer numba cloudpickle \
    && pip install -c requirements.txt "jupyterlab>=4,<5" "papermill>=2,<3" "ipywidgets>=8,<9" \
                                       "pytest>=8,<9" "pyyaml>=6,<7" \
    && python -m ipykernel install --name python3 --display-name "Python 3"

# Copia o codigo por ultimo (muda mais que as dependencias). A imagem roda sozinha,
# sem depender de bind-mount do repositorio.
COPY notebooks/ /workspace/notebooks/
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh \
    && mkdir -p /workspace/results /workspace/.hf-cache

EXPOSE 8888

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["jupyter"]
