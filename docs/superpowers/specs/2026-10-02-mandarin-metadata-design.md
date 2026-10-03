# Extração de metadados mandarim

## Objetivo

Extrair e validar separadamente os metadados nativos das duas fontes mandarim,
sem modificar os originais nem afirmar pareamento entre corpora:

- AISHELL-3 para áudio pristine;
- JMDS v2 / ADD para áudio generated.

O resultado servirá para caracterizar o mandarim e selecionar futuramente
subconjuntos comparáveis aos de português e inglês. Extração e processamento
de áudio ficam fora deste escopo.

## Evidência de entrada

O inventário somente leitura encontrou:

- JMDS/ADD: 24.642 linhas `language=zho`, `label=generated`,
  `dataset=ADD`, sendo 7.146 train, 7.497 dev e 9.999 eval; IDs únicos e WAVs
  1:1 em `dataset/Chinese_ADD_Generated/{split}/wav/{utt_id}.wav`;
- os campos `attack_id`, `gender` e `spk_id` do ADD são `unk`, `codec` é `-`
  e `native` é `yes` em todas as linhas;
- AISHELL-3: archive `data_aishell3.tgz` com cerca de 19 GB, aproximadamente
  88.035 utterances e 218 falantes; metadados em `spk-info.txt`,
  `{train,test}/content.txt` e `train/prosody_label_train-set.txt`;
- o `test/content.txt` observado possui 24.773 linhas, enquanto a publicação
  oficial reporta 23.262 amostras test. Essa diferença deve ser explicada pela
  reconciliação entre metadados e membros WAV antes da publicação.

As 4.410 linhas pristine `zho/AISHELL3` do protocolo JMDS não entram no
manifesto AISHELL-3: seus IDs são anonimizados e não existe mapa público para
os nomes `SSB...` do archive.

## Arquitetura

O perfil mandarim imutável concentra códigos, splits, paths e contagens JMDS
comprovadas. Dois adapters independentes ficam em `sources`:

- o adapter AISHELL-3 percorre o `.tgz` sequencialmente, lê apenas arquivos
  textuais e inventaria nomes/tamanhos dos WAVs sem extrair seu conteúdo;
- o adapter JMDS/ADD valida o protocolo heterogêneo, seleciona somente
  `zho/generated/ADD` e exige resolução 1:1 dos WAVs.

O pipeline de metadados deve ser parametrizado por perfil/adapters e reutilizar
layout, serialização e publicação idempotente já usados no português. Não deve
conter ramificações por idioma.

## Reconciliação AISHELL-3

Antes de construir o manifesto, o adapter deve:

1. validar paths TAR seguros, membros regulares e nomes únicos;
2. analisar `spk-info.txt`, train/test `content.txt` e prosody com schemas
   explícitos e encoding comprovado;
3. obter `speaker_id` do identificador nativo e associar gênero, idade e
   sotaque somente pelo `spk-info.txt`;
4. comparar IDs dos metadados com os membros WAV de cada split;
5. publicar contagens de correspondentes, ausentes e extras por fonte de
   metadados, explicando a divergência test;
6. falhar para duplicatas, path traversal, speaker inexistente ou perda de
   áudio/metadado que torne o manifesto ambíguo.

Nenhuma linha será descartada silenciosamente para forçar uma contagem
publicada. O relatório deve distinguir utterances com transcrição, prosódia e
WAV disponível.

## Artefatos

O destino local terá:

- `manifests/aishell3_metadata.csv`;
- `manifests/jmds_add_generated_metadata.csv`;
- `reports/aishell3_metadata_profile.json`;
- `reports/jmds_add_generated_metadata_profile.json`;
- `reports/mandarin_metadata_summary.json`;
- `reports/mandarin_metadata_provenance.json`.

O manifesto AISHELL-3 preserva identificador, split, path interno do archive,
speaker, gênero, faixa etária, sotaque, transcrição, pinyin e disponibilidade
de prosódia, além da rastreabilidade. O manifesto JMDS/ADD preserva os nove
campos JMDS mais split, arquivo/linha de origem e `audio_path`.

Os profiles JSON descrevem separadamente campos, tipos, nulabilidade,
procedência e distribuições, sem listar transcrições, IDs ou paths. O summary
mantém as fontes separadas, registra `paired_samples: false` e compara apenas
características realmente disponíveis. A provenance registra SHA-256/tamanho
do archive e protocolos, licença Apache-2.0 do AISHELL-3, unknowns justificados
do JMDS/ADD, schemas, origens por artefato e as 4.410 linhas pristine JMDS
excluídas.

## Publicação e recursos

Os seis payloads são publicados em uma única operação content-idempotente. Uma
segunda execução idêntica deve manter todos os hashes. Conteúdo divergente não
é sobrescrito.

O scan do `.tgz` é sequencial e pode ser demorado, mas não grava nem decodifica
WAVs. O adapter deve processar membros em streaming e manter apenas índices e
textos necessários em memória; nunca carregar o archive ou áudio inteiro.

## Testes

O desenvolvimento será test-first com um pequeno `.tgz` fixture contendo
metadados e WAVs fictícios. Cobertura mínima:

- schemas, encodings, paths TAR e membros duplicados/inseguros;
- parser de speaker, content e prosody;
- reconciliação metadata–WAV e relatório de diferenças;
- filtro `zho/generated/ADD`, contagens, IDs e paths 1:1;
- profiles, summary, provenance e ausência de pairing;
- publicação idempotente e conflitos;
- pipeline sem dependência de CLI ou idioma literal;
- execução real duas vezes e validação independente dos seis outputs.

## Limites científicos

AISHELL-3 pristine e ADD generated são corpora distintos, não pares real–fake.
Os metadados esparsos do ADD não devem ser preenchidos por inferência. O
desequilíbrio entre as fontes e o papel experimental de `eval` serão tratados
somente na etapa posterior de seleção, não nesta extração.
