#!/usr/bin/env bash
# Sincroniza o repositório (scripts/ + arquivos do módulo Docker) para a VM de GPU e
# executa um comando dentro do container `deepfake` lá.
#
# Uso:
#   scripts/dev/vm-test.sh python -m pytest scripts/tests -v
#   scripts/dev/vm-test.sh build          # atalho: reconstrói a imagem na VM
#   scripts/dev/vm-test.sh run-scripts /workspace/scripts/configs/smoke.yaml
#
# Config via env (defaults para a VM atual):
#   VM_SSH   destino ssh            (default: daniel.melo@34.71.156.183)
#   VM_KEY   chave privada ssh      (default: ~/.ssh/gcp_key)
#   VM_DIR   pasta remota do projeto(default: exp_docker  -> ~/exp_docker)
set -euo pipefail

VM_SSH="${VM_SSH:-daniel.melo@34.71.156.183}"
VM_KEY="${VM_KEY:-$HOME/.ssh/gcp_key}"
VM_DIR="${VM_DIR:-exp_docker}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SSH_CMD=(ssh -i "$VM_KEY")

if [[ $# -eq 0 ]]; then
    echo "uso: $0 <comando para o entrypoint do container>" >&2
    exit 2
fi

echo ">> sincronizando scripts/ e arquivos Docker para ${VM_SSH}:${VM_DIR}/ ..."
rsync -az --delete -e "ssh -i ${VM_KEY}" \
    --exclude '__pycache__' --exclude '.pytest_cache' \
    "${REPO_ROOT}/scripts/" "${VM_SSH}:${VM_DIR}/scripts/"
rsync -az -e "ssh -i ${VM_KEY}" \
    "${REPO_ROOT}/docker-compose.yml" "${REPO_ROOT}/Makefile" "${VM_SSH}:${VM_DIR}/"
rsync -az -e "ssh -i ${VM_KEY}" \
    "${REPO_ROOT}/docker/" "${VM_SSH}:${VM_DIR}/docker/"

if [[ "${1}" == "build" ]]; then
    echo ">> rebuild da imagem na VM ..."
    exec "${SSH_CMD[@]}" "${VM_SSH}" "cd ${VM_DIR} && docker compose build"
fi

echo ">> executando no container: $*"
exec "${SSH_CMD[@]}" "${VM_SSH}" "cd ${VM_DIR} && docker compose run --rm deepfake $*"
