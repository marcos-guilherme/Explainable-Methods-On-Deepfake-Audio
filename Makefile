# Atalhos para o modulo Docker (ver docker/README.md).
# Pre-requisitos no host: driver NVIDIA (CUDA 13.0) + nvidia-container-toolkit.

IMAGE   ?= brspeech-deepfake:cu126
COMPOSE ?= docker compose
SERVICE ?= deepfake

.PHONY: help build jupyter run shell smoke clean test run-scripts

help:
	@echo "Alvos disponiveis:"
	@echo "  make build    - constroi a imagem (torch cu130 + fairseq + stack pinado)"
	@echo "  make jupyter  - sobe o Jupyter Lab em :8888 (use tunel SSH)"
	@echo "  make run      - executa o notebook de explicabilidade headless (papermill)"
	@echo "  make shell    - abre um shell dentro do container"
	@echo "  make smoke    - verifica se a GPU esta visivel para o torch"
	@echo "  make test     - roda a suite de testes (pytest) do pacote brspeech_xai"
	@echo "  make run-scripts - executa o pipeline em estagios (CONFIG=<arquivo yaml>)"
	@echo "  make clean    - derruba o servico e remove volumes (cache HF incluso)"

build:
	$(COMPOSE) build

jupyter:
	@echo "Jupyter Lab -> http://localhost:8888"
	@echo "Tunel SSH:    ssh -L 8888:localhost:8888 usuario@vm"
	$(COMPOSE) up $(SERVICE)

run:
	$(COMPOSE) run --rm $(SERVICE) run

shell:
	$(COMPOSE) run --rm $(SERVICE) bash

smoke:
	$(COMPOSE) run --rm $(SERVICE) python -c "import torch; ok=torch.cuda.is_available(); print('CUDA disponivel:', ok); print('GPU:', torch.cuda.get_device_name(0) if ok else 'nenhuma')"

test:
	$(COMPOSE) run --rm $(SERVICE) python -m pytest scripts/tests -v

run-scripts:
	$(COMPOSE) run --rm $(SERVICE) run-scripts $(CONFIG)

clean:
	$(COMPOSE) down --volumes --remove-orphans
