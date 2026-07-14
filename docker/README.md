# Módulo Docker (execução em GPU)

Imagem **autossuficiente** para rodar os notebooks de detecção de deepfake (XLS-R + fairseq)
numa VM com GPU (ex.: A100). Todas as dependências são instaladas no build — não há nenhum
passo de instalação por fora.

## Pré-requisitos do host

Só o que é inevitável para acessar a GPU:

1. Driver NVIDIA compatível com **CUDA 12.6** (os wheels usados são `torch==2.13.0+cu126`).
   Drivers da série r5xx (ex.: `550` / CUDA 12.4) já servem, pois o runtime cu126 roda neles
   por *minor-version compatibility* (mesmo major 12). Verifique com `nvidia-smi`.
2. [`nvidia-container-toolkit`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
   instalado e configurado (`sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`).
3. Docker Engine com Compose v2 (`docker compose version`).

Nada de Python, CUDA toolkit ou pip no host — está tudo dentro da imagem.

## Uso rápido (via Makefile, a partir da raiz do repo)

```bash
make build      # constroi a imagem (demora: compila o fairseq)
make smoke      # confirma que a GPU esta visivel para o torch
make jupyter    # sobe o Jupyter Lab em :8888
make run        # executa o notebook de explicabilidade de ponta a ponta (headless)
make shell      # abre um shell dentro do container
make clean      # derruba o servico e remove volumes (inclui o cache HF)
```

### Jupyter Lab (interativo)

```bash
make jupyter
# Em outro terminal, na sua maquina local:
ssh -L 8888:localhost:8888 usuario@vm
# Abra http://localhost:8888 no navegador.
```

Sem token por padrão (adequado atrás de túnel SSH). Para exigir token, defina `JUPYTER_TOKEN`
no `docker/.env`.

### Execução headless (papermill)

```bash
make run
# ou, escolhendo o notebook:
docker compose run --rm -e NOTEBOOK=/workspace/notebooks/deepfake_brspeech_zeroshot.ipynb deepfake run
```

Os resultados (notebook executado, `master_table.csv`, `performance_table.csv`,
`shap_importance.csv`, `occlusion_table.csv`, `confirmatory_tests.csv` e `figures/`) aparecem
em `./results` no host.

## Sem Compose / imagem standalone

A imagem roda sozinha (o código é copiado para dentro dela):

```bash
docker build -t brspeech-deepfake:cu126 -f docker/Dockerfile .
docker run --rm --gpus all -p 8888:8888 brspeech-deepfake:cu126            # jupyter
docker run --rm --gpus all -v "$PWD/results:/workspace/results" \
    brspeech-deepfake:cu126 run                                            # headless
```

## Hugging Face (opcional)

```bash
cp docker/.env.example docker/.env
# edite docker/.env e preencha HF_TOKEN=... para evitar rate limit
```

O cache do HF fica no volume `hf-cache` (`HF_HOME=/workspace/.hf-cache`), então modelo e
dataset não são re-baixados a cada execução.

## Detalhes da imagem

- Base `python:3.12-slim-bookworm` (o `requirements.txt` foi congelado num ambiente 3.12 —
  ex.: `shap==0.52.0` exige Python >= 3.12); a GPU vem do host (`--gpus all`), pois os wheels
  `torch cu126` já trazem as libs CUDA.
- Ordem de instalação pensada para o `fairseq 0.12.2` (parte frágil): `pip<24.1` →
  `torch/torchaudio` (índice cu126) → `numpy<2 + cython + omegaconf + hydra` →
  `fairseq` (do **git**, tag `v0.12.2`, `--no-build-isolation`) → `requirements.txt`
  (com `--no-deps`) + deps do shap + `jupyterlab`/`papermill`.
- O fairseq é instalado do GitHub (não do PyPI) porque o *sdist* do `0.12.2` no PyPI
  omite `fairseq/clib/libbase/balanced_assignment.cpp`, quebrando a compilação com
  `No such file or directory`. A tag `v0.12.2` do repositório tem o arquivo.
- O `requirements.txt` é instalado com `--no-deps`: ele fixa `numpy==1.26.4` (o fairseq não
  compila com numpy>=2), mas o `shap==0.52.0` declara `numpy>=2` no metadata — a resolução
  estrita é impossível. Como no Colab, o shap roda bem em numpy 1.26.4, então instala-se o
  *freeze* sem re-resolver e preenchem-se as deps transitivas que faltam ao shap
  (`slicer`/`numba`/`cloudpickle`/`llvmlite`) com o `requirements.txt` como *constraints*
  (para o numpy não subir para 2). `fairseq`/`torch`/`torchaudio` saem da lista do freeze.
- `IN_DOCKER=1` faz os notebooks pularem a célula de `%pip` automaticamente.

## Solução de problemas

- **`torch.cuda.is_available()` é `False`**: confira `nvidia-smi` no host, o
  `nvidia-container-toolkit` e se está usando `--gpus all` (ou o serviço do compose).
- **`torch.cuda` falha com "driver too old"**: o `torch` foi instalado para uma CUDA de major
  maior que a do driver. Use um índice `cu1xx` do mesmo major do driver (ex.: driver 12.4 ->
  `cu126` roda por minor-version compatibility; `cu130` exigiria driver r580+). Ajuste o
  `--index-url` no passo 2 do `Dockerfile`.
- **`fairseq`: `balanced_assignment.cpp: No such file or directory`**: bug do *sdist*
  do PyPI. Já contornado instalando do git (tag `v0.12.2`) no `Dockerfile`.
- **`fairseq`: erro de compilação C++ contra o `torch 2.13`**: se aparecer erro de
  compilação (e não mais "No such file"), o toolchain pode não bater com o torch novo.
  Fallbacks: (a) trocar a base por `nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04` e
  reinstalar o Python; ou (b) fixar o ref do fairseq no commit/fork exato usado quando
  o `requirements.txt` foi gerado.
- **`pip ResolutionImpossible` mencionando `numpy`/`shap`**: é o conflito esperado
  (`shap==0.52.0` pede `numpy>=2`, mas o fairseq exige `numpy<2`). O `Dockerfile` já
  contorna com `--no-deps` + preenchimento das deps do shap; não relaxe o pin do numpy.
- **`no space left on device` ao *exportar* a imagem**: o build terminou mas o disco
  encheu ao gravar as camadas (a imagem final tem ~10 GB). Libere espaço (ex.:
  `rm -rf ~/.cache/pip`, `docker builder prune`) e rode `make build` de novo — as camadas
  já compiladas vêm do cache e só a exportação é refeita.
