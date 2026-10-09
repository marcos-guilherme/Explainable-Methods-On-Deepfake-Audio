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
| 2 | `embeddings`   | `emb_<split>.npy` (train, calibration quando configurado, eval/test)     |
| 3 | `adapt`        | `d_ad.joblib`, `thresholds.json`, `eer_precheck.json`, `p_spoof_zs.npy`, `p_spoof_ad.npy` |
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

## Calibração de limiar (source-only)

O estágio `adapt` separa **fit**, **calibrate** e **score**:

1. **Fit** — o head `D_ad` treina **somente** no split `train` (holdout) ou gera scores OOF no pool (cross-fit).
2. **Calibrate** — o limiar EER é definido **apenas** com scores/labels da fonte de calibração:
   - manifests XAI locais: role `calibration` (`n_calibration_per_class: 602` em `xai-*-local.yaml`);
   - baseline HF/BRSpeech legado (YAML sem campos de calibration): calibração **in-sample no train** após o fit — nunca usa labels do eval/test.
3. **Score** — `p_spoof_ad.npy` / `p_spoof_zs.npy` guardam scores do split de **análise** (eval/test ou pool); predições/quadrantes em `features` usam o limiar fixo de `thresholds.json` (sem recomputar EER no alvo).

`eer_precheck.json` e `performance_table.csv` distinguem métricas **threshold-free** (`eer_diagnostic`, ROC-AUC, AP) de métricas **fixed-threshold** (acurácia/MCC/TPR/FPR no limiar salvo).

## Testes

```bash
make test    # docker compose run --rm deepfake python -m pytest scripts/tests -v
```

## Suíte layer-wise trilíngue na VM

A suíte compara onde a decisão de detecção emerge ao longo das camadas e como
essa trajetória muda entre inglês, português e mandarim. O protocolo principal
usa três encoders de 12 blocos Transformer — HuBERT Base, WavLM Base e
Wav2Vec2 Base — com probes por camada e AttnLRP → DFT-LRP/STDFT-LRP. A auditoria
clássica H1/H2/H3 é opcional, limitada às diagonais da camada 12 e desligada por
padrão.

A comparação ativa usa checkpoints somente pré-treinados no LibriSpeech 960h:
`facebook/hubert-base-ls960`, `microsoft/wavlm-base` e
`facebook/wav2vec2-base`. O checkpoint ajustado para ASR
`facebook/wav2vec2-base-960h` não faz parte deste protocolo.

Para validar a pipeline com menor custo, faça primeiro um `--dry-run` parcial
com inglês e português:

```bash
PYTHONPATH=scripts python -m brspeech_xai.encoder_suite \
  --eng-config scripts/configs/xai-eng-local.yaml \
  --por-config scripts/configs/xai-por-local.yaml \
  --profiles hubert_base \
  --output /mnt/results/hubert_base__eng-por__layerwise_xai__pilot \
  --xai-per-class 2 \
  --dry-run
```

Depois de revisar o `execution_plan.json`, execute o piloto no mesmo output:

```bash
PYTHONPATH=scripts python -m brspeech_xai.encoder_suite \
  --eng-config scripts/configs/xai-eng-local.yaml \
  --por-config scripts/configs/xai-por-local.yaml \
  --profiles hubert_base \
  --output /mnt/results/hubert_base__eng-por__layerwise_xai__pilot \
  --xai-per-class 2
```

Os outputs de um único perfil seguem a convenção
`<model>__<languages>__layerwise_xai__<scope>` descrita em
[Convenção de nomes dos resultados](#convenção-de-nomes-dos-resultados).

Esse piloto `eng+por` valida a pipeline, mas não substitui o experimento final,
que continua trilíngue. A execução final usa um diretório de output próprio por
perfil, com escopo `full` (veja abaixo), sem reutilizar o output parcial.

Execute na VM Linux com Python 3.11, as dependências do projeto, PyTorch e
Transformers compatíveis com a CUDA instalada. Monte no contêiner/VM:

- o repositório, incluindo as três configs;
- os manifests referenciados pelas configs;
- todos os WAVs apontados por `processed_path`;
- um diretório gravável e persistente para `/mnt/results`.

Cada config deve usar `dataset_kind: local_manifest`, conter os papéis
`train`, `calibration` e `test`, e declarar obrigatoriamente
`calibration_split: calibration` e `n_calibration_per_class` positivo. A
calibração in-sample é rejeitada.

Cada perfil é executado separadamente e tem o seu próprio diretório de output,
nomeado pela [convenção](#convenção-de-nomes-dos-resultados), para que cada
resultado de modelo/idioma seja fácil de identificar. O preflight com
`--dry-run` é obrigatório antes de carregar modelos e é feito uma vez por perfil;
o padrão para `hubert_base` é:

```bash
PYTHONPATH=scripts python -m brspeech_xai.encoder_suite \
  --eng-config scripts/configs/xai-eng-local.yaml \
  --por-config scripts/configs/xai-por-local.yaml \
  --zho-config scripts/configs/xai-zho-local.yaml \
  --profiles hubert_base \
  --output /mnt/results/hubert_base__eng-por-zho__layerwise_xai__full \
  --xai-per-class 25 \
  --dry-run
```

Repita com `--profiles wavlm_base` e o output
`/mnt/results/wavlm_base__eng-por-zho__layerwise_xai__full`, e com
`--profiles wav2vec2_base` e o output
`/mnt/results/wav2vec2_base__eng-por-zho__layerwise_xai__full`.

Leia o `execution_plan.json` de cada output (por exemplo,
`/mnt/results/hubert_base__eng-por-zho__layerwise_xai__full/execution_plan.json`)
antes de continuar. Ele registra hashes de configs/manifests, 36 probes e 108
células por perfil (108 probes e 324 células nos três perfis), estimativas
separadas de backprops de explicação/certificação/trace e a fórmula de espaço
dos embeddings. Confirme espaço disponível, quotas, paths montados e orçamento
de backprops.

Depois do preflight, faça primeiro o piloto de um perfil:

```bash
PYTHONPATH=scripts python -m brspeech_xai.encoder_suite \
  --eng-config scripts/configs/xai-eng-local.yaml \
  --por-config scripts/configs/xai-por-local.yaml \
  --zho-config scripts/configs/xai-zho-local.yaml \
  --profiles hubert_base \
  --output /mnt/results/hubert_base__eng-por-zho__layerwise_xai__pilot \
  --xai-per-class 2
```

Antes da execução completa, valide no piloto os certificados de conservação
AttnLRP/DFT/STDFT, hashes e manifests das gerações, identidade das coortes e as
seis tabelas agregadas. Com apenas um perfil,
`encoder_relevance_agreement.csv` deve ser uma tabela vazia válida com status
`not_applicable_less_than_two_profiles`, e a figura correspondente deve ser
pulada; as outras cinco tabelas continuam obrigatórias. Então execute três
execuções separadas, uma por perfil, cada uma com o seu output canônico:

```bash
PYTHONPATH=scripts python -m brspeech_xai.encoder_suite \
  --eng-config scripts/configs/xai-eng-local.yaml \
  --por-config scripts/configs/xai-por-local.yaml \
  --zho-config scripts/configs/xai-zho-local.yaml \
  --profiles hubert_base \
  --output /mnt/results/hubert_base__eng-por-zho__layerwise_xai__full \
  --xai-per-class 25

PYTHONPATH=scripts python -m brspeech_xai.encoder_suite \
  --eng-config scripts/configs/xai-eng-local.yaml \
  --por-config scripts/configs/xai-por-local.yaml \
  --zho-config scripts/configs/xai-zho-local.yaml \
  --profiles wavlm_base \
  --output /mnt/results/wavlm_base__eng-por-zho__layerwise_xai__full \
  --xai-per-class 25

PYTHONPATH=scripts python -m brspeech_xai.encoder_suite \
  --eng-config scripts/configs/xai-eng-local.yaml \
  --por-config scripts/configs/xai-por-local.yaml \
  --zho-config scripts/configs/xai-zho-local.yaml \
  --profiles wav2vec2_base \
  --output /mnt/results/wav2vec2_base__eng-por-zho__layerwise_xai__full \
  --xai-per-class 25
```

Como cada execução contém um único perfil,
`encoder_relevance_agreement.csv` (que exige dois ou mais perfis no mesmo output)
volta a ser uma tabela vazia válida com status
`not_applicable_less_than_two_profiles`.

A retomada é automática: stages com marker e artefatos íntegros são pulados.
`run_status.json` informa `running`, `failed` ou `complete`; `--force`
reexecuta toda a DAG. Embeddings, probes, células, gerações XAI/trace e
agregados ficam sob o diretório de output. Use `--classical-audit` somente se
também quiser executar H1/H2/H3, com o custo adicional correspondente.

Resultados off-diagonal são validação externa sob *corpus shift*, não efeitos
causais do idioma. Os ICs de emergência usam bootstrap estratificado percentil;
agreement entre encoders e reorganização usam aproximação normal de 95% da
média (`normal_95_sem`). A divergência espectral compara, por
profile/source/layer/classe verdadeira, a distribuição média de relevância
absoluta normalizada de cada target off-diagonal com a diagonal `target=source`;
publica Jensen-Shannon e distância/similaridade cosseno, sem IC estimado.

Embeddings, XAI e trace final usam a mesma rotina configurada de
mono/resample/fixed-length (`audio.num_samples`). Cada exemplo STDFT publica
também somas temporal e tempo-frequência, erros absoluto/relativo e tolerância
do certificado de conservação.

O smoke local da Task 11 usa CPU e fakes determinísticos e não comprova
compatibilidade com checkpoint real, CUDA ou áudio real; nenhum sucesso
GPU/modelo HuggingFace real é alegado. O piloto acima continua sendo o próximo
passo de validação na VM.

### Convenção de nomes dos resultados

Todo diretório de resultado que alimenta o relatório layer-wise segue

```text
<model>__<languages>__layerwise_xai__<scope>
```

- `<model>`: perfil do encoder, por exemplo `hubert_base`, `wavlm_base` ou
  `wav2vec2_base`;
- `<languages>`: códigos ISO 639-3 na ordem canônica `eng`, `por`, `zho`,
  unidos por `-`. A suíte aceita qualquer subconjunto não vazio de inglês,
  português e mandarim, e o relatório aceita as sete combinações: `eng`, `por`,
  `zho`, `eng-por`, `eng-zho`, `por-zho` e `eng-por-zho`;
- `layerwise_xai`: protocolo;
- `<scope>`: `pilot` ou `full`.

Exemplos: `hubert_base__eng__layerwise_xai__full`,
`wavlm_base__eng__layerwise_xai__full` e
`hubert_base__eng-por-zho__layerwise_xai__pilot`. O nome é passado em
`--output` ao executar a suíte; um diretório já concluído pode ser renomeado
inteiro, sem alterar seu conteúdo.

O relatório lê **um modelo por diretório**, e é por isso que os comandos acima
executam cada perfil em seu próprio diretório:
`hubert_base__eng-por-zho__layerwise_xai__full`,
`wavlm_base__eng-por-zho__layerwise_xai__full` e
`wav2vec2_base__eng-por-zho__layerwise_xai__full`.

O exemplo HuBERT `hubert_base__eng__layerwise_xai__full` usado abaixo é o estudo
de caso somente em inglês, já concluído, e é separado das futuras execuções
trilíngues acima, cujos resultados ficarão nos diretórios `eng-por-zho`.

## Relatório layer-wise atualizável

`brspeech_xai.layerwise_report` gera, a partir de resultados concluídos, um
bundle LaTeX portátil em português com figuras (PDF e PNG), tabelas (CSV e
LaTeX), `report_manifest.json` e `build_local.ps1`.

O relatório inclui resumo executivo e conclusão; mapas treino
$\times$ avaliação de ROC-AUC e MCC por modelo; e a associação descritiva entre
ROC-AUC e concentração espectral da relevância. Em cada célula dos mapas, a
camada é selecionada pela maior ROC-AUC, e o MCC mostrado vem dessa mesma
camada. As tabelas `transfer_selected_layers.csv` e
`xai_performance_association.csv` persistem essas reduções.

Na VM, informe cada diretório de resultado explicitamente (a opção `--result`
pode ser repetida; nenhum diretório é varrido por padrão e identidades
modelo/idioma/escopo duplicadas são rejeitadas):

```bash
PYTHONPATH=scripts python -m brspeech_xai.layerwise_report \
  --result /mnt/results/hubert_base__eng__layerwise_xai__full \
  --output /mnt/results/layerwise_xai_report
```

Copie o bundle da VM para a máquina Windows (por exemplo, para
`escrita/relatorio-layerwise-xai/`) e compile com o `pdflatex` do MiKTeX:

```powershell
Set-Location escrita\relatorio-layerwise-xai
.\build_local.ps1
```

`build_local.ps1` executa `pdflatex` duas vezes em modo `nonstopmode` e falha se
`report.pdf` não for produzido. Os arquivos temporários do LaTeX não devem ser
versionados.

**Garantia somente leitura e sem GPU.** O gerador só lê artefatos persistidos:
valida o status `complete` da execução, as gerações imutáveis (hashes) e os
markers das células, e grava apenas dentro de `--output`. Ele recusa um
`--output` que se sobreponha a um diretório de resultado ou que seja um
diretório não vazio sem bundle anterior. Nunca carrega encoder, não importa
`torch` nem `transformers` e não exige GPU. A saída é determinística para os
mesmos artefatos: `report_manifest.json` registra, com caminhos relativos e
SHA-256, os artefatos consumidos e os arquivos gerados.

Com um único modelo ou idioma, o texto mantém o escopo de estudo de caso e não
inventa comparações. Quando células 3$\times$3 estão presentes, resultados fora
da diagonal são descritos como transferência sob mudança de corpus/idioma, sem
interpretação causal. A associação XAI$\leftrightarrow$desempenho usa as 12
camadas por célula e é marcada como indisponível quando os dados necessários
faltam.
