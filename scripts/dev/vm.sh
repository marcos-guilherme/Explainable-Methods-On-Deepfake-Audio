#!/usr/bin/env bash
# Controle remoto da VM de GPU sem precisar entrar nela manualmente.
#
# Sincroniza o código (scripts/ + Docker) para a VM e roda o pipeline dentro do
# container `deepfake`. Runs longas ficam em segundo plano (nohup), com log em arquivo,
# e os resultados voltam por rsync.
#
# Uso:
#   scripts/dev/vm.sh setup                      # provisiona a VM (Docker + toolkit NVIDIA)
#   scripts/dev/vm.sh sync                       # só envia o código para a VM
#   scripts/dev/vm.sh build                      # envia e reconstrói a imagem
#   scripts/dev/vm.sh test                       # roda a suíte de testes (pytest)
#   scripts/dev/vm.sh smoke                      # run rápido (configs/smoke.yaml), em 1º plano
#   scripts/dev/vm.sh run [CONFIG] [-- --set k=v ...]   # run em 2º plano (nohup + log)
#   scripts/dev/vm.sh run-fg [CONFIG] [-- ...]   # run em 1º plano (bloqueia; para depurar)
#   scripts/dev/vm.sh resume RUN_DIR [-- --from STAGE --force]  # reprocessa estágios de uma run existente
#   scripts/dev/vm.sh logs                       # acompanha o log da última run (tail -f)
#   scripts/dev/vm.sh status                      # containers rodando + últimas linhas do log
#   scripts/dev/vm.sh fetch [DEST]               # baixa results/ da VM (default: ./results)
#   scripts/dev/vm.sh shell                       # shell dentro do container
#   scripts/dev/vm.sh ssh                         # ssh cru na VM
#
# Exemplos:
#   scripts/dev/vm.sh run                                   # default.yaml em 2º plano
#   scripts/dev/vm.sh run configs/default.yaml -- --set bands.n_bands=16
#   scripts/dev/vm.sh logs                                  # acompanha o progresso
#   scripts/dev/vm.sh fetch                                 # traz os resultados
#
# Config via env (defaults = VM atual):
#   VM_SSH  destino ssh        (default: daniel.melo@35.254.89.173)
#   VM_KEY  chave privada ssh  (default: ~/.ssh/gcp_key)
#   VM_DIR  pasta remota       (default: /home/daniel.melo/xai_experiments)
set -euo pipefail

VM_SSH="${VM_SSH:-daniel.melo@35.254.89.173}"
VM_KEY="${VM_KEY:-$HOME/.ssh/gcp_key}"
VM_DIR="${VM_DIR:-/home/daniel.melo/xai_experiments}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SSH=(ssh -i "$VM_KEY" "$VM_SSH")
RSH="ssh -i ${VM_KEY}"
COMPOSE="cd ${VM_DIR} && docker compose"

log() { printf '>> %s\n' "$*" >&2; }

# Envia código-fonte, arquivos Docker e garante as pastas montadas pelo compose.
do_sync() {
    log "sincronizando código para ${VM_SSH}:${VM_DIR}/ ..."
    "${SSH[@]}" "mkdir -p ${VM_DIR}/results ${VM_DIR}/notebooks ${VM_DIR}/logs"
    rsync -az --delete -e "$RSH" \
        --exclude '__pycache__' --exclude '.pytest_cache' \
        "${REPO_ROOT}/scripts/" "${VM_SSH}:${VM_DIR}/scripts/"
    rsync -az -e "$RSH" \
        "${REPO_ROOT}/docker/" "${VM_SSH}:${VM_DIR}/docker/"
    rsync -az -e "$RSH" \
        --exclude '.ipynb_checkpoints' \
        "${REPO_ROOT}/notebooks/" "${VM_SSH}:${VM_DIR}/notebooks/"
    # requirements.txt e docker-compose.yml/Makefile fazem parte do contexto de build
    # (o Dockerfile faz COPY requirements.txt); precisam estar na raiz remota.
    rsync -az -e "$RSH" \
        "${REPO_ROOT}/requirements.txt" \
        "${REPO_ROOT}/docker-compose.yml" \
        "${REPO_ROOT}/Makefile" \
        "${VM_SSH}:${VM_DIR}/"
}

# Config default e normalização: aceita "default.yaml", "configs/x.yaml" ou caminho absoluto.
resolve_config() {
    local cfg="${1:-configs/default.yaml}"
    case "$cfg" in
        /*) printf '%s' "$cfg" ;;                                   # já absoluto (no container)
        scripts/*) printf '/workspace/%s' "$cfg" ;;
        configs/*) printf '/workspace/scripts/%s' "$cfg" ;;
        *) printf '/workspace/scripts/configs/%s' "$cfg" ;;         # só o nome do arquivo
    esac
}

# Provisiona uma VM Ubuntu limpa: Docker Engine + compose + nvidia-container-toolkit.
# Idempotente: pula o que já estiver instalado. Requer sudo sem senha na VM.
do_setup() {
    log "provisionando ${VM_SSH} (Docker + toolkit NVIDIA) ..."
    "${SSH[@]}" 'bash -s' <<'REMOTE'
set -euo pipefail
if ! command -v docker >/dev/null 2>&1; then
    echo ">> instalando Docker Engine + compose ..."
    curl -fsSL https://get.docker.com | sudo sh
    sudo usermod -aG docker "$USER"
else
    echo ">> Docker já instalado: $(docker --version)"
fi
if ! command -v nvidia-ctk >/dev/null 2>&1; then
    echo ">> instalando nvidia-container-toolkit ..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    sudo apt-get update -qq
    sudo apt-get install -y -qq nvidia-container-toolkit
else
    echo ">> nvidia-container-toolkit já presente: $(nvidia-ctk --version | head -1)"
fi
echo ">> configurando runtime NVIDIA no Docker ..."
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
echo ">> verificação:"
sudo docker --version
sudo docker compose version | head -1
echo ">> teste de GPU no container (pode baixar uma imagem CUDA pequena):"
sudo docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi \
    --query-gpu=name,driver_version --format=csv,noheader || echo "!! teste de GPU falhou"
REMOTE
    log "setup concluído. Novas sessões SSH já entram no grupo 'docker' (sem sudo)."
}

cmd="${1:-help}"; shift || true

case "$cmd" in
    setup)
        do_setup
        ;;

    sync)
        do_sync
        ;;

    build)
        do_sync
        log "reconstruindo a imagem na VM ..."
        "${SSH[@]}" "${COMPOSE} build"
        ;;

    test)
        do_sync
        "${SSH[@]}" "${COMPOSE} run --rm -T deepfake python -m pytest scripts/tests -q"
        ;;

    smoke)
        do_sync
        log "smoke run (1º plano) ..."
        "${SSH[@]}" "${COMPOSE} run --rm -T deepfake run-scripts /workspace/scripts/configs/smoke.yaml"
        ;;

    run|run-fg)
        do_sync
        # Separa CONFIG (opcional) dos overrides após '--'.
        config_arg=""
        if [[ "${1:-}" != "" && "${1:-}" != "--" ]]; then config_arg="$1"; shift || true; fi
        [[ "${1:-}" == "--" ]] && shift || true
        cfg="$(resolve_config "$config_arg")"
        overrides="$*"
        if [[ "$cmd" == "run-fg" ]]; then
            log "run em 1º plano: ${cfg} ${overrides}"
            "${SSH[@]}" "${COMPOSE} run --rm -T deepfake run-scripts ${cfg} ${overrides}"
        else
            ts="$(date +%Y%m%d-%H%M%S)"
            remote_log="logs/run-${ts}.log"
            log "run em 2º plano: ${cfg} ${overrides}"
            log "log remoto: ${VM_DIR}/${remote_log}"
            "${SSH[@]}" "cd ${VM_DIR} && nohup docker compose run --rm -T deepfake run-scripts ${cfg} ${overrides} > ${remote_log} 2>&1 & echo \"PID remoto: \$!\""
            log "acompanhe com: $0 logs   |   baixe com: $0 fetch"
        fi
        ;;

    resume)
        # Reprocessa estágios de uma run EXISTENTE (reaproveita embeddings/adapt/features),
        # usando o config.resolved.yaml da própria run para preservar a grade de bandas.
        do_sync
        run_dir_arg="${1:-}"; shift || true
        [[ "${1:-}" == "--" ]] && shift || true
        if [[ -z "$run_dir_arg" ]]; then
            log "uso: $0 resume RUN_DIR [-- --from STAGE --force]"
            log "ex.: $0 resume default-20260823-154103 -- --from association --force"
            exit 2
        fi
        case "$run_dir_arg" in
            /*) rdir="$run_dir_arg" ;;
            results/*) rdir="/workspace/${run_dir_arg}" ;;
            *) rdir="/workspace/results/${run_dir_arg}" ;;
        esac
        extra="${*:---from association --force}"   # default: regenera figuras de association em diante
        ts="$(date +%Y%m%d-%H%M%S)"
        remote_log="logs/resume-${ts}.log"
        log "resume em 2º plano: ${rdir} ${extra}"
        log "log remoto: ${VM_DIR}/${remote_log}"
        "${SSH[@]}" "cd ${VM_DIR} && nohup docker compose run --rm -T deepfake \
            python /workspace/scripts/dev/resume_run.py \
            --config ${rdir}/config.resolved.yaml --run-dir ${rdir} ${extra} \
            > ${remote_log} 2>&1 & echo \"PID remoto: \$!\""
        log "acompanhe com: $0 logs   |   baixe com: $0 fetch"
        ;;

    logs)
        # Segue o log mais recente em logs/.
        "${SSH[@]}" "cd ${VM_DIR} && f=\$(ls -t logs/*.log 2>/dev/null | head -1); \
            if [[ -z \"\$f\" ]]; then echo 'nenhum log encontrado'; else echo \">> \$f\"; tail -n 200 -f \"\$f\"; fi"
        ;;

    status)
        "${SSH[@]}" "cd ${VM_DIR} && echo '== containers ==' && docker compose ps; \
            echo '== último log ==' && f=\$(ls -t logs/*.log 2>/dev/null | head -1); \
            [[ -n \"\$f\" ]] && { echo \"\$f\"; tail -n 20 \"\$f\"; } || echo 'sem logs'"
        ;;

    fetch)
        # Traz results/ sem os áudios crus (audios_*.npy, ~GBs): eles ficam na VM,
        # pois só servem para recomputar oclusão (que roda lá). Figuras/tabelas/embeddings
        # (o que interessa para escrita/análise) vêm normalmente.
        dest="${1:-${REPO_ROOT}/results}"
        mkdir -p "$dest"
        log "baixando results/ (sem áudios crus) da VM para ${dest} ..."
        rsync -az --info=progress2 -e "$RSH" \
            --exclude 'audios_*.npy' \
            "${VM_SSH}:${VM_DIR}/results/" "${dest}/"
        log "pronto."
        ;;

    shell)
        exec "${SSH[@]}" -t "${COMPOSE} run --rm deepfake bash"
        ;;

    ssh)
        exec "${SSH[@]}"
        ;;

    help|-h|--help)
        sed -n '2,40p' "$0"
        ;;

    *)
        echo "comando desconhecido: ${cmd}" >&2
        echo "use: $0 help" >&2
        exit 2
        ;;
esac
