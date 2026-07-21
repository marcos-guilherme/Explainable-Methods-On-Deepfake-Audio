# Métodos aplicados — guia de estudo

Documento para você **estudar e confirmar** cada peça do estudo. Para cada método:
**o quê / por quê / como (parâmetros) / pressupostos e ressalvas / onde no código /
onde no paper**. No fim há um glossário e um checklist de verificação.

> Convenção de "registro" (importante para não escorregar na linguagem):
> - **Descritivo**: descreve um padrão nos dados, sem afirmar significância nem causa.
> - **Inferencial**: teste estatístico formal com controle de erro (FDR).
> - **Causal**: baseado em **intervenção** no áudio (só a oclusão permite isso).

---

## 0. Visão geral

- **Pergunta:** quais pistas acústicas o detector de deepfake usa para decidir, e
  como essas pistas mudam depois de uma adaptação leve ao português?
- **Dois detectores** sobre o mesmo front-end XLS-R congelado:
  - `D_zs` (zero-shot): cabeça original do checkpoint;
  - `D_ad` (adaptado): só uma regressão logística treinada sobre os embeddings pt-BR.
- **Três hipóteses:**
  - **H1** (associativa, descritiva): o perfil de associação feature↔score muda entre `D_zs` e `D_ad`.
  - **H2** (causal, com incerteza): as bandas cuja oclusão altera o score mudam entre os detectores.
  - **H3** (inferencial): as features salientes têm distribuição diferente entre os quadrantes de erro.
- **Duas análises complementares** no MESMO eixo de frequência (8 bandas mel):
  - **associativa** (Spearman intra-classe) e **causal** (oclusão espectral);
  - mais uma medida de **convergência** entre elas e uma camada **confirmatória**.

---

## 1. Dados e amostragem

- **O quê:** `AKCIT-Deepfake/BRSpeech-DF` (459.137 clipes: 76.644 bonafide / 382.493 spoof).
- **Como:**
  - **Treino da cabeça `D_ad`:** subconjunto balanceado, **1.500 por classe** (`n_train_per_class`).
  - **Conjunto de análise (test):** **todo** o bonafide de teste (1.320, a classe escassa) + **1.500** spoof → **2.820 clipes**.
  - Test nunca é visto no treino da cabeça (sem vazamento).
- **Por quê subconjunto (e não o corpus todo):** o objetivo é **explicabilidade populacional**, não ranking. Este N já dá poder para H3 (Welch/Levene) e para os ICs de H2; exaurir o bonafide escasso mantém os quadrantes e o limiar de EER significativos.
- **Ressalva:** escala é escolha de projeto; estratificar por gerador de spoof e escalar para o corpus todo é trabalho futuro.
- **Código:** `data.py` (`build_balanced_split`), config em `scripts/configs/default.yaml`.
- **Paper:** §Experimental Setup (tabela de config + parágrafo "Dataset and sampling rationale").

---

## 2. Detectores (`D_zs` vs `D_ad`)

- **O quê:** dois classificadores que mapeiam 1 clipe → 1 score contínuo `P(spoof)`.
- **Como:**
  - **Front-end:** wav2vec 2.0 / XLS-R **congelado** (checkpoint `nii-yamagishilab/mms-300m-anti-deepfake`), embeddings de 1024 dim.
  - **`D_zs`:** usa a cabeça anti-spoofing original do checkpoint.
  - **`D_ad`:** `StandardScaler` + `LogisticRegression` (`C=1.0`, `max_iter=2000`) treinada sobre os embeddings; devolve `P(spoof)` calibrado.
- **Por quê assim:** congelar o encoder e treinar só a cabeça **isola o efeito da adaptação no USO das representações** (contraste "antes/depois" controlado). Logística dá probabilidade calibrada.
- **Pressuposto/ressalva:** a adaptação é **rasa** — as mudanças observadas refletem **re-ponderação** de representações pré-existentes, não features novas aprendidas para pt-BR. Não extrapolar para fine-tuning completo.
- **Pré-condição:** só comparamos pistas se `EER(D_ad) < EER(D_zs)` (senão, "adaptação insuficiente").
- **Código:** `model.py` (`build_detector`), `stages.py` (`stage_embeddings`, `stage_adapt`).
- **Paper:** §Methodology (Detectors), §Results (Performance).

---

## 3. Pré-processamento

- **O quê:** cada clipe → mono → **16 kHz** → janela fixa de **64.600 amostras (~4,04 s)** → normalização (Tak et al., 2022).
- **Por quê:** as features acústicas são extraídas **exatamente do sinal que entra no detector** (não do áudio bruto). A normalização neutraliza nível absoluto; explicar sobre o bruto atribuiria pistas erradas.
- **Código:** `preprocessing.py` (`preprocess`, `to_mono`, `resample_to_16k`).
- **Paper:** §Methodology (Pre-processing).

---

## 4. Feature acústica — energia log-mel por banda

- **O quê:** **energia log-mel em 8 bandas** (as MESMAS bordas da oclusão), resumida por **média (μ)** e **desvio (σ)** ao longo dos frames → **16 features** por clipe.
- **Como:**
  - STFT de potência (janelas de 25 ms, hop 10 ms: `n_fft=400`, `hop=160`).
  - Integra a potência dos bins de FFT dentro de cada banda; aplica `log`; tira μ e σ sobre os frames.
  - Bordas: 8 bandas espaçadas em **escala mel** de **20 a 7900 Hz** (`bands.mel_band_edges`).
- **Por quê (substituiu a MFCC):** MFCC é **cepstral** — cada coeficiente é uma base global, **não localizada em frequência**. Energia de banda é localizada, então a análise associativa (§6) vive no **mesmo eixo** da oclusão causal (§7) e as duas podem ser comparadas banda a banda. Também some a antiga dependência de "surrogate + SHAP" (que tinha fidelidade baixa).
- **Colunas:** `band{k}_mean`, `band{k}_std` (`k=1..8`). Rótulos por faixa em kHz.
- **Bandas (Hz):** 1: 20–282 · 2: 282–639 · 3: 639–1125 · 4: 1125–1788 · 5: 1788–2693 · 6: 2693–3926 · 7: 3926–5607 · 8: 5607–7900.
- **Pressuposto/ressalva:** energia de banda **não é fonema**; é ponteiro acústico de **onde** está a potência, não prova fonética.
- **Código:** `bands.py`, `features.py` (`mel_band_features`), `stages.py` (`stage_features`).
- **Paper:** §Methodology (Acoustic Features).

---

## 5. Métricas de desempenho e quadrantes

- **EER (Equal Error Rate):** ponto em que FPR = FNR; devolve também o **limiar** de decisão. Métrica padrão de anti-spoofing (menor é melhor).
- **Limiar → predições → quadrantes:** `pred = (P(spoof) ≥ limiar)`. Cruzando `pred` com o rótulo real formam-se os quadrantes:
  - **TP** (spoof pego), **TN** (bonafide aceito), **FP** (bonafide marcado como spoof), **FN** (spoof que passou).
- **MCC** (Matthews) e **accuracy:** métricas resumo complementares.
- **DET curve:** FNR × FPR em eixos de desvio-normal, com o ponto de EER marcado.
- **Código:** `metrics.py` (`compute_eer`, `quadrant`), `stages.py` (`stage_report`), `plotting.py` (`plot_det`).
- **Paper:** §Results (tabela de performance, Tab. de quadrantes, Fig. DET).

---

## 6. Análise associativa — Spearman intra-classe (H1)

- **O quê:** para cada detector, correlação de **Spearman ρ** (posto, monotônica) entre cada feature de banda e o score `P(spoof)`.
- **Como:**
  - **Dentro de cada classe** (bonafide e spoof **separados**).
  - **FDR (Benjamini–Hochberg)** sobre todos os p-valores do detector.
  - Perfil por banda = **max|ρ| sobre classes**; seleção das top-`N` features (`top_n=10`) para H3.
- **Por quê intra-classe:** como o score já separa as classes, juntar tudo criaria **correlação espúria** para quase qualquer feature que apenas difere entre classes. Dentro da classe, ρ reflete covariação monotônica **genuína**.
- **Por quê Spearman (e não surrogate/SHAP):** é **model-agnostic** e transparente — sem modelo substituto, sem pré-condição de fidelidade. **Custo:** é **univariada** (não modela interações entre bandas).
- **Registro:** **descritivo** no contraste entre detectores (cada ρ é FDR-controlado, mas não afirmamos significância da *diferença* `D_zs` vs `D_ad`).
- **Código:** `stats.py` (`spearman_intraclass`, `top_features_by_rho`), `stages.py` (`stage_association`).
- **Paper:** §Explainability (Acoustic Association via Spearman + Fig. de perfil e scatter).

---

## 7. Análise causal — oclusão espectral (H2)

- **O quê:** intervir no áudio removendo uma banda por vez e medir a **queda de `P(spoof)`**.
- **Como:**
  - Filtro **band-stop Butterworth ordem 4, zero-phase** (`sosfiltfilt`) sobre a forma de onda (evita a inversão mel→áudio, que é lossy).
  - **Queda = baseline (áudio íntegro) − ocluído**, por clipe e por banda.
    - queda **positiva** → ocluir derruba `P(spoof)` → a banda **contribui para a decisão spoof**;
    - queda **negativa** → ocluir sobe `P(spoof)` → a banda **contribui para a decisão bonafide**.
  - **Subconjunto estratificado por quadrante:** até **150 clipes por quadrante** (`per_quadrant`) → ~600 por detector.
  - **Incerteza:** **IC 95% por bootstrap** sobre os clipes (`n_boot=1000`, percentis 2,5/97,5). Banda cujo **IC exclui zero** tem efeito causal confiável.
  - Áudio reamostrado para 16 kHz antes de filtrar (bordas < Nyquist 8 kHz).
- **Por quê é causal:** é **intervenção** no input do próprio detector (não observação passiva).
- **Ressalva (o confound → Fase 2):** o band-stop **remove o conteúdo E injeta artefato** de filtragem. Uma queda pode vir da ausência da banda *ou* da distorção. Hoje isso está declarado como limitação; a **Fase 2** adiciona um controle (ruído com energia casada) para isolar o efeito do conteúdo.
- **Registro:** **causal com incerteza quantificada** (IC por banda); sem teste formal de significância *entre* detectores.
- **Código:** `occlusion.py` (`bandstop`, `occlusion_drop`, `stratified_idx`, `bootstrap_ci`), `stages.py` (`stage_occlusion`).
- **Paper:** §Explainability (Spectral Occlusion Profiles + Figs. divergente e overlay).

---

## 8. Convergência das duas análises

- **O quê:** um número por detector = **Spearman banda-a-banda** entre o perfil **causal** (|queda de oclusão| por banda) e o perfil **associativo** (max|ρ| por banda).
- **Por quê:** testa **validade convergente** — as bandas que o detector *causalmente* usa são as mesmas *associadas* ao score?
- **Como:** correlaciona os dois vetores de 8 valores (um por banda).
- **Ressalva CRÍTICA:** são só **8 pontos** → é **resumo descritivo, não teste com poder**. No `D_zs`, ρ=0,69 é **sugestivo mas NÃO significativo** (p=0,06 > 0,05). No `D_ad`, ρ=0,10 (p=0,82): as análises **divergem** → para o `D_ad` ancoramos afirmações por banda **na causal**, não na associativa. (Não especulamos o porquê da divergência.)
- **Código:** `stats.py` (`cross_spine_agreement`), `plotting.py` (`plot_spine_convergence`).
- **Paper:** §Methodology (Convergence of the Two Analyses) e §Explainability (mesma subseção nos resultados).

---

## 9. Testes confirmatórios (H3)

- **O quê:** testes clássicos **só nas features de maior |ρ|** (evita "pescaria" sobre as 16), ligados à estrutura de erro:
  - **Welch t** (diferença de **média**) entre **TN vs FP** (rejeições certas vs erradas);
  - **Levene** (diferença de **variância**) entre **TP vs FN** (spoofs pegos vs perdidos).
- **Como:** **FDR (Benjamini–Hochberg)** sobre todos os testes do detector; significância reportada como **q < 0,05**.
- **Por quê esses contrastes:** TN vs FP isola *por que confunde bonafide com spoof* (diferença de nível médio da feature); TP vs FN isola *por que perde spoofs* (diferença de variabilidade).
- **Registro:** **único inferencial** do estudo (cabe "significativo após FDR").
- **Código:** `stats.py` (`confirmatory_tests`), `stages.py` (`stage_confirmatory`), `plotting.py` (`plot_confirmatory_box`).
- **Paper:** §Explainability (Confirmatory Tests + boxplots por quadrante).

---

## 10. Reprodutibilidade e engenharia

- **Pipeline em estágios** (resumível): `collect → embeddings → adapt → features → association → occlusion → confirmatory → report`. Cada estágio grava artefatos e um "done marker" (`.<estágio>.done.json`) com **hash da config**; re-rodar pula o que já está feito (mesmo hash).
- **Seeds fixas** (`seed=42`) em `random`/`numpy`/`torch`.
- **Docker + A100**: ambiente reprodutível; execução headless via `run-scripts <config>.yaml`.
- **Artefatos-chave** (em `results/<run>/`): `master_table.parquet`, `spearman_table.csv`, `occlusion_table.csv`, `spine_agreement.csv`, `confirmatory_tests.csv`, `performance_table.csv`, `figures/*.pdf`.
- **Config central:** `scripts/configs/default.yaml` (parâmetros abaixo).
- **Código:** `pipeline.py`, `artifacts.py`, `cli.py`, `seed.py`, `logging_utils.py`.

### Parâmetros do run atual (`default.yaml`)
| Parâmetro | Valor |
|---|---|
| `n_train_per_class` | 1.500 |
| `n_test_per_class` | 1.500 (bonafide limitado a 1.320) |
| bandas (`n_bands`, `f_min`, `f_max`) | 8, 20 Hz, 7900 Hz |
| `per_quadrant` (oclusão) | 150 (~600/detector) |
| `n_boot` (IC) | 1.000 |
| `association.top_n` | 10 |
| `seed` | 42 |

---

## 11. Números do run atual (para conferência)

- **Desempenho:** EER 25,8% (`D_zs`) → 20,9% (`D_ad`); MCC 0,483 → 0,581; acc 0,742 → 0,791.
- **Quadrantes:** `D_zs` TP 1113 / TN 979 / FP 341 / FN 387; `D_ad` TP 1186 / TN 1044 / FP 276 / FN 314.
- **Associação (top):** `D_zs` energia grave (20–282 Hz μ: ρ≈+0,45 spoof; 282–639 Hz μ: ρ≈+0,41). `D_ad` migra p/ médios (1,79–2,69 kHz μ: ρ=−0,34 em bonafide).
- **Oclusão:** `D_zs` broadband (7/8 bandas com IC≠0); `D_ad` seletiva (graves) + 1,13–1,79 kHz **inverte sinal** (contribui p/ bonafide).
- **Convergência:** `D_zs` ρ=0,69 (p=0,06, **não** significativo); `D_ad` ρ=0,10 (p=0,82).
- **Confirmatória:** `D_zs` 9/20 e `D_ad` 13/20 testes significativos (q<0,05).

---

## 12. Glossário rápido

- **`P(spoof)`**: probabilidade contínua de o clipe ser spoof (score do detector).
- **EER**: taxa em que FPR = FNR; define o limiar de decisão.
- **FDR / Benjamini–Hochberg**: controla a proporção esperada de falsos positivos entre os testes; reporta `q-value`.
- **Spearman ρ**: correlação de **posto** (monotônica, robusta a não-linearidade/outliers).
- **Welch t**: teste t para médias sem assumir variâncias iguais.
- **Levene**: teste de igualdade de **variâncias**.
- **Bootstrap (percentil)**: reamostra clipes com reposição para estimar IC da média.
- **Band-stop zero-phase**: filtro que remove uma faixa de frequência sem defasar o sinal.
- **Registro descritivo/inferencial/causal**: ver topo do documento.

---

## 13. Checklist para você confirmar

- [ ] A distinção **descritivo vs inferencial vs causal** está clara e você concorda com onde cada hipótese cai (H1 descritiva, H2 causal-com-incerteza, H3 inferencial)?
- [ ] Faz sentido **intra-classe** no Spearman (evitar correlação espúria)?
- [ ] Concorda que a **convergência (8 bandas)** é descritiva e o p=0,06 do `D_zs` **não** é significativo?
- [ ] Entende a **ressalva do confound** da oclusão e por que a Fase 2 a endereça?
- [ ] Os **contrastes de H3** (TN vs FP para média; TP vs FN para variância) fazem sentido para você?
- [ ] Os **parâmetros** (§10) e os **números** (§11) batem com o que está no paper?
