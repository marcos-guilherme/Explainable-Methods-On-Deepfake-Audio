# Módulo Docker (execução em GPU)

Imagem **autossuficiente** para rodar os notebooks de detecção de deepfake (XLS-R + fairseq)
numa VM com GPU (ex.: A100). Todas as dependências são instaladas no build — não há nenhum
passo de instalação por fora.

## Pré-requisitos do host

Só o que é inevitável para acessar a GPU:

1. Driver NVIDIA que suporte **CUDA 13.0** (os wheels usados são `torch==2.13.0+cu130`).
   Verifique com `nvidia-smi`.
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
docker build -t brspeech-deepfake:cu130 -f docker/Dockerfile .
docker run --rm --gpus all -p 8888:8888 brspeech-deepfake:cu130            # jupyter
docker run --rm --gpus all -v "$PWD/results:/workspace/results" \
    brspeech-deepfake:cu130 run                                            # headless
```

## Hugging Face (opcional)

```bash
cp docker/.env.example docker/.env
# edite docker/.env e preencha HF_TOKEN=... para evitar rate limit
```

O cache do HF fica no volume `hf-cache` (`HF_HOME=/workspace/.hf-cache`), então modelo e
dataset não são re-baixados a cada execução.

## Detalhes da imagem

- Base `python:3.11-slim-bookworm`; a GPU vem do host (`--gpus all`), pois os wheels
  `torch cu130` já trazem as libs CUDA.
- Ordem de instalação pensada para o `fairseq 0.12.2` (parte frágil): `pip<24.1` →
  `torch/torchaudio` (índice cu130) → `numpy<2 + cython + omegaconf + hydra` →
  `fairseq` (do **git**, tag `v0.12.2`, `--no-build-isolation`) → `requirements.txt` +
  `jupyterlab`/`papermill`.
- O fairseq é instalado do GitHub (não do PyPI) porque o *sdist* do `0.12.2` no PyPI
  omite `fairseq/clib/libbase/balanced_assignment.cpp`, quebrando a compilação com
  `No such file or directory`. A tag `v0.12.2` do repositório tem o arquivo.
- `IN_DOCKER=1` faz os notebooks pularem a célula de `%pip` automaticamente.

## Solução de problemas

- **`torch.cuda.is_available()` é `False`**: confira `nvidia-smi` no host, o
  `nvidia-container-toolkit` e se está usando `--gpus all` (ou o serviço do compose).
- **Driver não suporta CUDA 13.0**: atualize o driver, ou troque as versões de
  `torch`/`torchaudio` (e o índice cu1xx) no `Dockerfile` para uma CUDA compatível com o host.
- **`fairseq`: `balanced_assignment.cpp: No such file or directory`**: bug do *sdist*
  do PyPI. Já contornado instalando do git (tag `v0.12.2`) no `Dockerfile`.
- **`fairseq`: erro de compilação C++ contra o `torch 2.13`**: se aparecer erro de
  compilação (e não mais "No such file"), o toolchain pode não bater com o torch novo.
  Fallbacks: (a) trocar a base por `nvidia/cuda:13.0.1-cudnn-devel-ubuntu22.04` e
  reinstalar o Python; ou (b) fixar o ref do fairseq no commit/fork exato usado quando
  o `requirements.txt` foi gerado.
