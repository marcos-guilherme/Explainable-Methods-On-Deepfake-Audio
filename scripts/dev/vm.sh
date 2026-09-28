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
#   scripts/dev/vm.sh rollout RUN_DIR            # gera a figura de attention roll-out (encoder hf_ssl)
#   scripts/dev/vm.sh dft-lrp RUN_DIR [BACKEND]  # DFT-LRP do D_ad (hf_ssl); reporta o portão de conservação
#   scripts/dev/vm.sh rollout-compare [REF_SLUG] # grade comparando encoders HF SSL nos mesmos clipes
#   scripts/dev/vm.sh sonify [-- FLAGS...]       # gera áudios guiados pela relevância DFT-LRP
#   scripts/dev/vm.sh faithfulness [-- FLAGS...] # valida keep/delete/random em segundo plano
#   scripts/dev/vm.sh faithfulness-audio [-- FLAGS...]  # exporta exemplos audíveis das intervenções
#   scripts/dev/vm.sh logs                       # acompanha o log da última run (tail -f)
#   scripts/dev/vm.sh status                      # containers rodando + últimas linhas do log
#   scripts/dev/vm.sh stop                        # para runs em andamento (containers + nohup)
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

    rollout)
        # Gera a figura de attention roll-out (temporal, ilustrativa) para uma run de
        # encoder hf_ssl. Roda em 1º plano (poucos clipes, rápido) dentro do container,
        # onde há GPU e os áudios crus. A figura fica em <run_dir>/figures/.
        do_sync
        run_dir_arg="${1:-}"; shift || true
        if [[ -z "$run_dir_arg" ]]; then
            log "uso: $0 rollout RUN_DIR"
            log "ex.: $0 rollout results/hf_ssl-wavlm-base/wavlm-20260824-173800"
            exit 2
        fi
        case "$run_dir_arg" in
            /*) rdir="$run_dir_arg" ;;
            results/*) rdir="/workspace/${run_dir_arg}" ;;
            *) rdir="/workspace/results/${run_dir_arg}" ;;
        esac
        log "attention roll-out (1º plano): ${rdir}"
        "${SSH[@]}" "${COMPOSE} run --rm -T deepfake \
            python /workspace/scripts/dev/attention_rollout.py --run-dir ${rdir}"
        log "baixe a figura com: $0 fetch"
        ;;

    dft-lrp)
        # Roda o DFT-LRP do D_ad (encoder hf_ssl) em 1º plano no container, onde há GPU e os
        # áudios crus. Reporta o PORTÃO de conservação no log e salva tabela/figura na run.
        # Uso: $0 dft-lrp RUN_DIR [BACKEND] [-- FLAGS...]   (BACKEND: attnlrp [default] | gxi)
        # FLAGS extras vão direto ao script, ex.: -- --stdft-examples 4 (heatmap tempo-frequência).
        do_sync
        run_dir_arg="${1:-}"; shift || true
        backend="attnlrp"
        if [[ "${1:-}" != "" && "${1:-}" != "--" ]]; then backend="$1"; shift || true; fi
        [[ "${1:-}" == "--" ]] && shift || true
        extra="$*"
        if [[ -z "$run_dir_arg" ]]; then
            log "uso: $0 dft-lrp RUN_DIR [attnlrp|gxi] [-- FLAGS...]"
            log "ex.: $0 dft-lrp results/hf_ssl-wav2vec2-base/wav2vec2-20260824-134313 attnlrp"
            log "ex.: $0 dft-lrp <RUN_DIR> attnlrp -- --stdft-examples 4   # heatmap tempo-frequência"
            exit 2
        fi
        case "$run_dir_arg" in
            /*) rdir="$run_dir_arg" ;;
            results/*) rdir="/workspace/${run_dir_arg}" ;;
            *) rdir="/workspace/results/${run_dir_arg}" ;;
        esac
        log "DFT-LRP (1º plano): ${rdir} | backend=${backend} ${extra:+| extra: ${extra}}"
        "${SSH[@]}" "${COMPOSE} run --rm -T deepfake \
            python /workspace/scripts/dev/dft_lrp_ad.py --run-dir ${rdir} --backend ${backend} ${extra}"
        log "baixe a tabela/figura com: $0 fetch"
        ;;

    rollout-compare)
        # Grade comparando o attention roll-out dos encoders HF SSL nos MESMOS clipes.
        # Auto-descobre a última run de cada encoder em results/. REF_SLUG (opcional)
        # escolhe o encoder de referência para a seleção dos clipes (default: wavlm).
        # Saída: results/_aggregate/attention_rollout_compare.{png,pdf}.
        do_sync
        ref_slug="${1:-wavlm}"
        log "rollout-compare (1º plano) | referência: ${ref_slug}"
        "${SSH[@]}" "${COMPOSE} run --rm -T deepfake \
            python /workspace/scripts/dev/attention_rollout_compare.py \
            --results-root /workspace/results --ref-slug ${ref_slug}"
        log "baixe a figura com: $0 fetch"
        ;;

    sonify)
        # Sonifica a explicação DFT-LRP (soft-mask por direção) nos MESMOS clipes, comparando os
        # encoders HF SSL conservativos. Gera .wav (original + rumo a spoof/bonafide por encoder),
        # figura de envoltória e manifest.csv em results/_aggregate/sonification/.
        # Uso: $0 sonify [-- --clip-indices 187 1016 --pctl 99 --floor 0]
        do_sync
        [[ "${1:-}" == "--" ]] && shift
        extra="$*"
        log "sonify (1º plano) | flags: ${extra:-<default>}"
        "${SSH[@]}" "${COMPOSE} run --rm -T deepfake \
            python /workspace/scripts/dev/sonify_lrp.py \
            --results-root /workspace/results ${extra}"
        log "baixe os áudios com: $0 fetch  (ou rsync de results/_aggregate/sonification/)"
        ;;

    faithfulness)
        # Valida a fidelidade da explicação DFT-LRP por bandas (keep/delete/random + re-scoring).
        # Roda em 2º plano (todos os clipes de teste => demorado). Gera curvas + CSVs em
        # results/_aggregate/faithfulness/.
        # Uso: $0 faithfulness [-- --encoder wav2vec2 --ks 1 2 3 4 6 8 12 --n-random 5]
        do_sync
        [[ "${1:-}" == "--" ]] && shift
        extra="$*"
        ts="$(date +%Y%m%d-%H%M%S)"
        remote_log="logs/faithfulness-${ts}.log"
        log "faithfulness em 2º plano | flags: ${extra:-<default>}"
        log "log remoto: ${VM_DIR}/${remote_log}"
        "${SSH[@]}" "cd ${VM_DIR} && nohup docker compose run --rm -T deepfake \
            python /workspace/scripts/dev/faithfulness_bands.py \
            --results-root /workspace/results ${extra} \
            > ${remote_log} 2>&1 & echo \"PID remoto: \$!\""
        log "acompanhe com: $0 logs   |   baixe com: $0 fetch"
        ;;

    faithfulness-audio)
        # Gera só os .wav das intervenções por banda (original, keep/delete top-k e aleatório)
        # de poucos clipes, para ouvir antes da rodada completa. 1º plano (rápido).
        # Uso: $0 faithfulness-audio [-- --encoder wav2vec2 --audio-clips 187 1016 --audio-k 6]
        do_sync
        [[ "${1:-}" == "--" ]] && shift
        extra="$*"
        log "faithfulness-audio (1º plano) | flags: ${extra:-<default>}"
        "${SSH[@]}" "${COMPOSE} run --rm -T deepfake \
            python /workspace/scripts/dev/faithfulness_bands.py \
            --results-root /workspace/results --audio-only ${extra}"
        log "baixe os áudios com: rsync de results/_aggregate/faithfulness/audio/"
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

    stop)
        # Para runs em andamento: os containers one-off do 'docker compose run' têm nome
        # '...-deepfake-run-*'. Para o container (o pipeline roda dentro) e derruba o
        # cliente nohup pendente no host, se houver.
        log "parando runs em andamento na VM ..."
        "${SSH[@]}" "names=\$(docker ps --filter name=deepfake-run --format '{{.Names}}'); \
            if [[ -n \"\$names\" ]]; then echo \"\$names\" | xargs -r docker stop; \
            else echo 'nenhum container de run ativo'; fi; \
            pkill -f 'docker compose run' 2>/dev/null || true"
        log "pronto. Confira com: $0 status"
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
