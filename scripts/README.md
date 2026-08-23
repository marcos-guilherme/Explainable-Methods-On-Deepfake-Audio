# brspeech_xai — pipeline de explicabilidade acústica (em estágios, resumível)

Port do notebook `notebooks/deepfake_brspeech_explainability.ipynb` para um pacote Python
executável de forma headless, em estágios resumíveis, configurável por YAML + overrides de CLI.
Compara um detector de deepfake **zero-shot** (`D_zs`) com uma versão **levemente adaptada**
(`D_ad` = XLS-R congelado + regressão logística), usando **associação MFCC↔P(spoof)** (Spearman
intra-classe) e **oclusão espectral causal**, com testes confirmatórios corrigidos por FDR.

## Como rodar (dentro do Docker)

O módulo depende do ambiente da imagem Docker do projeto (torch/fairseq/GPU). A partir da raiz do
repositório:

```bash
# rodada rápida de fumaça (poucas amostras por classe)
make run-scripts CONFIG=scripts/configs/smoke.yaml

# rodada completa (padrão: 1500/classe)
make run-scripts CONFIG=scripts/configs/default.yaml

# equivalente direto (sem Makefile)
docker compose run --rm deepfake run-scripts /workspace/scripts/configs/smoke.yaml
```

Cada execução cria `results/<run_name>-<timestamp>/` com os artefatos, figuras (`figures/*.pdf`),
`config.resolved.yaml`, `run.log` e `run_manifest.json`.

### Overrides de configuração

```bash
docker compose run --rm deepfake run-scripts /workspace/scripts/configs/default.yaml \
  --set data.n_test_per_class=500 --set bands.n_bands=16 --set seed=7
```

## Estágios e artefatos

O pipeline executa 8 estágios, em ordem; cada um grava um marcador `.<estágio>.done.json` (com o
hash da config) e produz artefatos consumidos pelos estágios seguintes:

| # | estágio        | artefatos gerados                                                        |
|---|----------------|--------------------------------------------------------------------------|
| 1 | `collect`      | `audios_<split>.npy`, `srs_<split>.npy`, `samples.parquet`               |
| 2 | `embeddings`   | `emb_train.npy`, `emb_test.npy`                                          |
| 3 | `adapt`        | `d_ad.joblib`, `eer_precheck.json`, `p_spoof_zs.npy`, `p_spoof_ad.npy`   |
| 4 | `features`     | `master_table.parquet` (P(spoof), predições, quadrantes, energia log-mel por banda μ/σ) |
| 5 | `association`  | `spearman_table.csv` (Spearman intra-classe energia-de-banda↔P(spoof) + FDR); figuras: perfil por banda (μ/σ) e scatter de Spearman |
| 6 | `occlusion`    | `occlusion_table.csv` (queda média + IC 95% bootstrap por banda), `spine_agreement.csv` (convergência causal×associativo); figuras divergente, overlay e convergência |
| 7 | `confirmatory` | `confirmatory_tests.csv` (Welch/Levene + FDR nas bandas de maior \|ρ\|); figuras boxplot por quadrante (H3) |
| 8 | `report`       | `performance_table.csv`, curva DET (`det_zs_vs_ad`), `run_manifest.json`  |

## Resumibilidade

- Por padrão, um estágio cujo `.<estágio>.done.json` já existe (com o mesmo hash de config) é
  **pulado**. Assim, apontar para um diretório de run existente retoma de onde parou.
- `--from <estágio>`: começa a partir de um estágio (executa dele em diante).
- `--only <estágio>`: executa apenas um estágio.
- `--force`: ignora os marcadores e re-executa.
- `--no-plots`: não gera figuras (útil em rodadas puramente numéricas).

## Testes

```bash
make test    # docker compose run --rm deepfake python -m pytest scripts/tests -v
```
