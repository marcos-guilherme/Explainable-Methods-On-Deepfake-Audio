#!/usr/bin/env bash
# Entrypoint da imagem: despacha entre os modos de uso.
#   jupyter (default) -> Jupyter Lab em 0.0.0.0:8888
#   run               -> executa um notebook de ponta a ponta via papermill
#   bash | shell      -> shell interativo
#   <qualquer coisa>  -> executa o comando arbitrario (ex.: python -c "...")
set -euo pipefail

# Propaga HF_TOKEN (se fornecido) para as duas variaveis que as libs HF entendem.
if [[ -n "${HF_TOKEN:-}" ]]; then
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

MODE="${1:-jupyter}"
if [[ $# -gt 0 ]]; then
    shift
fi

case "${MODE}" in
    jupyter)
        echo ">> Jupyter Lab: http://0.0.0.0:8888  (use tunel SSH: ssh -L 8888:localhost:8888 usuario@vm)"
        if [[ -n "${JUPYTER_TOKEN:-}" ]]; then
            echo ">> Protegido por token (JUPYTER_TOKEN definido)."
        else
            echo ">> Sem token/senha (adequado atras de tunel SSH)."
        fi
        exec jupyter lab \
            --ip=0.0.0.0 \
            --port=8888 \
            --no-browser \
            --allow-root \
            --ServerApp.token="${JUPYTER_TOKEN:-}" \
            --ServerApp.password='' \
            --notebook-dir=/workspace
        ;;

    run)
        # Notebook a executar (sobrescrevivel via env NOTEBOOK).
        NB="${NOTEBOOK:-/workspace/notebooks/deepfake_brspeech_explainability.ipynb}"
        OUT_DIR="/workspace/results"
        mkdir -p "${OUT_DIR}"
        OUT_NB="${OUT_DIR}/$(basename "${NB%.ipynb}")_executed.ipynb"

        echo ">> Executando (papermill): ${NB}"
        echo ">> Saidas (CSVs / figuras / notebook executado) em: ${OUT_DIR}"
        # --cwd faz o kernel rodar dentro de OUT_DIR, entao os caminhos relativos do
        # notebook (master_table.csv, figures/, ...) caem nesse diretorio.
        # --kernel python3: o notebook nao tras kernelspec, entao fixamos o kernel
        # registrado na imagem para o papermill nao falhar.
        exec papermill "${NB}" "${OUT_NB}" \
            --kernel python3 \
            --cwd "${OUT_DIR}" \
            --log-output \
            "$@"
        ;;

    bash|shell)
        exec /bin/bash "$@"
        ;;

    *)
        exec "${MODE}" "$@"
        ;;
esac
