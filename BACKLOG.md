# Backlog — preparação do JMDS

## Objetivo

Construir um conjunto rastreável para detecção e explicabilidade de deepfakes
de áudio. A primeira etapa usa somente o subconjunto inglês cujo vínculo entre
áudio e metadados pode ser comprovado.

## Decisões

- Priorizar inglês antes de português e mandarim.
- Usar somente associações áudio–metadado demonstráveis.
- Não tratar o JMDS como um corpus pareado real–fake: os conteúdos e falantes
  não são necessariamente correspondentes.
- Preservar os áudios brutos; qualquer padronização deve gerar uma cópia.
- Registrar origem, checksum e transformação de cada amostra.
- Não usar o `eval` pristine inglês até resolver a divergência dos IDs.

## Situação dos dados

### Inglês — prioridade atual

Fonte dos dados: ASVspoof 5, identificado como `ASVspoof2024` nos protocolos
do JMDS.

| Split | Pristine | Generated | Situação |
|---|---:|---:|---|
| train | 18.797 | 65.424 | IDs pristine correspondem ao ASVspoof 5 |
| dev | 15.667 | 54.808 | IDs pristine correspondem ao ASVspoof 5 |
| eval | 15.513 | 54.252 | IDs pristine não correspondem ao protocolo público verificado |

Os áudios generated já estão no pacote JMDS. A aquisição dos pristine de train
e dev a partir dos arquivos oficiais do ASVspoof 5 foi concluída e o manifesto
raw (`english_raw.csv`) foi gerado. O processamento concluiu 154.696 amostras
e publicou `english_processed.csv`. A auditoria real das 154.696 amostras foi
concluída sem paths ausentes, hashes duplicados ou interseção de falantes entre
train e dev. A baseline técnica obteve balanced accuracy 0,4953 e ROC AUC
0,4544, sem evidência de um atalho técnico simples nas variáveis avaliadas.

### Estrutura do código

- Modularização em camadas (`core`, `storage`, `sources`, `profiles`,
  `pipelines`, `cli`) criada, com fachadas de compatibilidade temporárias.
  A suíte automatizada, a validação real dos protocolos, o processamento e a
  auditoria real foram concluídos.
- Arquitetura e regras de extensão: [docs/data-framework.md](docs/data-framework.md).
- `config`, `manifest`, `audio` e `audit` mantêm contratos legados, enquanto as
  regras inglesas e a orquestração ficam isoladas em `profiles` e `pipelines`.

### Português — metadados inventariados, áudio adiado

- Os 10 artefatos CORAA v1.1 foram baixados com recibos. Os sete arquivos de
  áudio têm SHA-256 oficial do Hugging Face LFS verificado; os três CSVs têm
  tamanho exato e SHA-256 observado, sem alegação de checksum oficial.
- Metadados nativos extraídos e verificados sobre dados reais (sem extrair
  áudio CORAA): `python -m jmds_prepare.portuguese_metadata --config
  configs/portuguese-metadata.yaml` publica seis artefatos locais em
  `E:\JMDS_open_v2_public\portuguese_preparation` (402.456 linhas CORAA,
  3.011 linhas MLAAD generated com resolução 1:1 de WAV, `file_path`/`utt_id`
  únicos, hashes de entrada conferidos, `paired_samples: false`,
  `excluded_jmds_pristine_count=1000`, política CORAA CC BY-NC-ND conservadora,
  `field_origins` separados, seção `jmds_mlaad`, summary com
  dataset/quality/comparable sem IDs/paths/text brutos). Regeneração final
  pós-revisão READY: backup reversível em
  `E:\JMDS_open_v2_public\portuguese_preparation.pre-global-review-fix-20261002`
  (seis artefatos preservados); segunda execução idempotente com hashes
  idênticos nos seis outputs finais, zero `.tmp`. Hashes finais: CSV CORAA
  `1b4fccd1…705e`, CSV JMDS `c8daeb58…5db`, profiles/summary/provenance
  JSON atualizados (`682383fb…`, `987b426e…`, `311c5e20…`, `bdf9c4d6…`).
- Generated: JMDS/MLAAD, 3.011 amostras.
- Pristine pretendido: CORAA.
- Não existe mapa público por amostra entre o protocolo pristine do JMDS e os
  arquivos CORAA.
- Os metadados pristine vêm do CORAA nativo, não das linhas pristine do JMDS.
- O desenho `CORAA real × MLAAD sintético` possui confundimento classe–corpus.
- O split `eval` português é inventariado, mas não tem papel experimental
  automático.
- Artefatos derivados do CORAA permanecem locais; não commitar nem redistribuir
  (CC BY-NC-ND 4.0 conservador).
- Investigar associação dos WAVs generated do JMDS com o `meta.csv` original
  do MLAAD por hash ou fingerprint para recuperar modelo, arquitetura,
  transcrição e referência de voz.
- Preparação de áudio português (extração CORAA, padronização, manifesto
  processado) ainda não existe.

### Mandarim — metadados inventariados, áudio adiado

- `data_aishell3.tgz` foi baixado com tamanho exato e SHA-256 observado
  `be2507d431ad59419ec871e60674caedb2b585f84ffa01fe359784686db0e0`.
  O OpenSLR não publicou checksum oficial no catálogo consultado.
- Metadados nativos extraídos e verificados sobre dados reais (streaming do
  archive, sem extrair WAVs): `python -m jmds_prepare.mandarin_metadata
  --config configs/mandarin-metadata.yaml` publica seis artefatos locais em
  `E:\JMDS_open_v2_public\mandarin_preparation` (88.035 linhas AISHELL-3 após
  reconciliação train 63.262 / test 24.773 linhas observadas vs 23.262
  oficiais documentadas no profile; 24.642 linhas ADD generated com resolução
  1:1 de WAV; `utt_id` únicos; hashes de entrada conferidos;
  `paired_samples: false`; `excluded_jmds_pristine_count=4410`; licença
  AISHELL-3 Apache-2.0; campos esparsos ADD não inferidos). Segunda execução
  idempotente com hashes idênticos nos seis outputs finais, zero `.tmp`. Hashes
  finais: CSV AISHELL `ffd2d67e…b970`, CSV ADD `ab869ee9…7e7c`, profiles/summary/provenance
  JSON (`b8d930e6…c578`, `93f0546b…c476`, `fa3821c3…c67e`, `c1b7daa4…b8f0`).
- Generated: JMDS/ADD, 24.642 amostras (`zho`), sendo 7.146 train, 7.497 dev
  e 9.999 eval. Os `utt_id` são únicos no subconjunto e os WAVs resolvem 1:1
  em `dataset/Chinese_ADD_Generated/{split}/wav/`. Os campos `attack_id`,
  `gender` e `spk_id` são `unk` em todas as linhas, portanto oferecem pouca
  informação explicativa.
- Pristine pretendido: AISHELL-3. Metadados vêm do archive nativo via adapter
  de streaming (`spk-info.txt`, `content.txt`, prosody labels); WAVs do archive
  não são extraídos. A discrepância test (24.773 observadas vs 23.262 oficiais)
  está documentada em `aishell3_metadata_profile.json` e provenance.
- As 4.410 linhas pristine `zho/AISHELL3` do protocolo JMDS permanecem apenas
  como referência anonimizada e não entram no manifesto gerado.
- Não existe mapa público comprovado por amostra entre os IDs pristine do JMDS
  e os arquivos AISHELL-3.
- Preparação de áudio mandarim (extração AISHELL, padronização, manifesto
  processado) ainda não existe.

## P0 — aquisição íntegra do inglês

Aquisição inglesa train/dev concluída. Os itens marcados abaixo seguem esse
estado; os demais precisam de evidência própria.

- [x] Registrar licença, DOI, URLs, tamanhos e checksums oficiais do ASVspoof 5
      em `reports/provenance.json`.
- [x] Confirmar os IDs ingleses de train/dev novamente por comparação
      automatizada entre os protocolos JMDS e ASVspoof 5.
- [x] Baixar sequencialmente `flac_T_aa.tar`–`flac_T_ae.tar`.
- [x] Verificar o MD5 de cada arquivo de train antes da extração.
- [x] Extrair somente os 18.797 FLAC pristine solicitados pelo protocolo JMDS.
- [x] Baixar sequencialmente `flac_D_aa.tar`–`flac_D_ac.tar`.
- [x] Verificar o MD5 de cada arquivo de dev antes da extração.
- [x] Extrair somente os 15.667 FLAC pristine solicitados pelo protocolo JMDS.
- [x] Remover cada TAR somente após checksum, extração e validação
      bem-sucedidos.
- [ ] Remover, após confirmação manual, dois resíduos de tentativas antigas:
      `flac_T_aa.tar.partial` e `flac_D_aa.tar.partial.invalid` (cerca de
      18,2 GB no total). Eles não são usados pelo ledger nem pela aquisição
      concluída.
- [x] Confirmar que todos os IDs esperados foram encontrados uma única vez.
- [x] Comparar `spk_id`, gênero, split e classe entre os dois protocolos.

## P0 — manifesto e rastreabilidade

Os manifestos raw e processado foram gerados. O manifesto processado contém
154.696 linhas com paths e checksums calculados.

- [x] Criar um manifesto canônico com, no mínimo:
  `utt_id`, `spk_id`, `gender`, `language`, `dataset`, `split`, `label`,
  `attack_id`, `source_path`, `processed_path`, `sha256_source` e
  `sha256_processed`.
- [x] Marcar a procedência de cada campo: JMDS, ASVspoof 5 ou calculado.
- [x] Falhar explicitamente quando um arquivo ou campo obrigatório estiver
      ausente, duplicado ou divergente.
- [x] Produzir relatório de contagem por split, classe, falante e ataque.

## P0 — auditoria contra vazamento

- [x] Procurar arquivos idênticos por SHA-256.
- [x] Procurar duplicatas após decodificação/normalização do áudio.
- [x] Verificar interseção de falantes entre os splits usados pelo experimento.
- [x] Manter variantes do mesmo utterance-fonte no mesmo grupo quando essa
      relação estiver disponível.
- [x] Auditar duração, sample rate, canais, loudness, silêncio, SNR e codec por
      classe.
- [x] Criar baseline com características técnicas simples; desempenho alto
      nessa baseline indica atalho de corpus/pipeline.

## P1 — aquisição das fontes futuras (concluída)

Escopo atual: somente download verificado dos originais, com recibos. Extração,
adapters e integração com os dados ingleses não fazem parte deste item.

- [x] AISHELL-3 (`data_aishell3.tgz`): download, tamanho exato e recibo.
- [x] CORAA v1.1: sete arquivos de áudio e três CSVs de metadados, com recibos.

Cada item só deve ser marcado quando todos os seus recibos existirem. Sem
checksum oficial, o SHA-256 observado é um registro, não uma verificação
oficial. Adapters futuros usam metadados nativos de cada corpus e nunca copiam
metadados pristine anonimizados do JMDS sem mapa por amostra comprovado.

## P1 — padronização

- [x] Definir um pipeline único para pristine e generated.
- [x] Gerar WAV PCM mono a 16 kHz sem alterar os arquivos brutos.
- [ ] Aplicar a mesma política de duração às duas classes.
- [ ] Registrar versão do pipeline e parâmetros no manifesto.
- [ ] Não normalizar de forma que destrua os artefatos de síntese que serão
      investigados; validar cada transformação por ablação.

## P1 — particionamento experimental

- [ ] Manter o train oficial como conjunto de treinamento inicial.
- [ ] Não usar o mesmo conjunto para ajuste e resultado final.
- [ ] Definir validação e teste bloqueados por falante e, quando possível, por
      utterance-fonte.
- [ ] Decidir se o dev oficial será dividido em validação/teste ou se o teste
      aguardará o mapeamento do eval.
- [ ] Reportar métricas por ataque e não somente uma média agregada.
- [ ] Tratar o desbalanceamento com ponderação, amostragem ou subconjuntos
      controlados; não usar acurácia simples isoladamente.

## P2 — explicabilidade

- [ ] Validar explicações com testes de remoção e inserção.
- [ ] Executar randomização de pesos/modelo como sanity check.
- [ ] Medir estabilidade entre seeds e amostras equivalentes.
- [ ] Verificar se as explicações se concentram em silêncio, padding, ruído,
      duração ou bordas do espectrograma.
- [ ] Comparar explicações antes e depois de controles de codec, volume e ruído.
- [ ] Interpretar uma explicação fiel de um modelo enviesado como evidência do
      viés, não como evidência causal de deepfake.

## Questões em aberto

- Qual versão exata do ASVspoof foi usada para construir o `eval` inglês do
  JMDS?
- Os autores podem fornecer o mapa original–anonimizado para CORAA, AISHELL-3
  e os demais pristine?
- Os códigos `MLAAD001`, `MLAAD002`, `MLAAD003` e `MLAAD020` podem ser ligados
  aos modelos/arquiteturas originais?
- Qual será o conjunto de teste final intocado do inglês?

## Critério de conclusão da primeira etapa

A etapa inglesa estará pronta para modelagem quando:

1. todos os pristine train/dev estiverem presentes e verificados;
2. o manifesto reproduzível estiver gerado sem divergências;
3. duplicatas e vazamentos tiverem sido auditados;
4. as duas classes tiverem passado pelo mesmo pipeline documentado;
5. validação e teste tiverem papéis definidos sem reutilização indevida.
