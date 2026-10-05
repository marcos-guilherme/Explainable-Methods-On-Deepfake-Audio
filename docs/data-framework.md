# Framework de dados: arquitetura e extensão

Este documento descreve como o pacote `jmds_prepare` está organizado e como
estendê-lo para outro corpus. Os nomes de comandos e de módulos ficam em inglês,
como no código. A aquisição das fontes externas está em
[sources.md](sources.md); o uso diário dos comandos está no
[README](../README.md).

Estado desta descrição: a estrutura de camadas existe e é protegida por testes
de contrato e de dependência. O fluxo inglês de train/dev também foi executado
sobre os dados reais: aquisição e processamento produziram 154.696 amostras, e
a auditoria foi concluída. Os limites de reutilização para novos corpora
continuam explícitos em "Limites atuais" (veja também o
[BACKLOG](../BACKLOG.md)).

## Camadas, responsabilidades e dependências permitidas

Cada camada só pode importar das camadas listadas como permitidas. A direção
geral é: `cli` → `pipelines` → `sources`/`storage` → `profiles`/`core`.

- `core`
  - Responsabilidade: primitivas sem conhecimento de corpus, como escrita
    atômica de JSON, hashes (SHA-256, MD5), publicação idempotente por
    conteúdo (`core/publication.py`), utilitários de caminho e tipos
    compartilhados de extração de metadados (`core/two_source.py`).
  - Pode importar: apenas `core`.
- `profiles`
  - Responsabilidade: dados imutáveis que descrevem um subconjunto (idioma,
    nome do dataset, splits, contagens esperadas, arquivos de protocolo,
    TARs oficiais exigidos e política de exclusão do eval). É o único lugar
    onde regras específicas do inglês devem estar escritas.
  - Pode importar: `core` (somente primitivas compartilhadas, como
    `TwoSourceArtifactNames`) e `profiles`. Hoje contém `profiles/english.py`,
    `profiles/portuguese.py` e `profiles/mandarin.py`.
- `storage`
  - Responsabilidade: tudo que toca disco ou rede de forma genérica: layout de
    `data_root`, download verificado e retomável, ledger de extração,
    validação de áudio, estimativas de espaço e catálogos de aquisição com
    recibos.
  - Pode importar: `core`, `profiles` e `storage`. Um teste de camada verifica
    isso.
- `sources`
  - Responsabilidade: leitura e validação estrita do que cada origem publica:
    protocolos do JMDS, protocolos do ASVspoof 5 e o registro do Zenodo.
    Cada módulo conhece o formato de uma origem e nada além disso.
  - Pode importar: `core`, `profiles` e `sources`.
- `pipelines`
  - Responsabilidade: orquestração dos comandos (validação, aquisição,
    processamento, auditoria, proveniência). Recebem seus colaboradores
    explicitamente, em vez de importá-los de `cli`.
  - Pode importar: `core`, `profiles`, `sources`, `storage`, `config` e, por
    enquanto, tipos de `extraction`. Nunca importa `cli`.
- `cli`
  - Responsabilidade: interpretar argumentos, carregar a configuração e ligar
    os colaboradores concretos aos pipelines. É a única camada que conhece
    todas as outras.
  - Pode importar: qualquer camada.

Os testes verificam, por análise dos imports, apenas as regras de `core` e
`storage`. As demais são a convenção a seguir; a de `pipelines` nunca importar
`cli` também está declarada no código.

Ainda existem módulos de nível superior com implementação própria, não migrados
para as camadas: `config`, `manifest`, `audio`, `audit` e `extraction`. Eles
importam `core`, `profiles` e `storage`, mas também uns aos outros e a fachada
`protocols`, e continuam presos ao inglês, como descrito em "Limites atuais".

## Fachadas de compatibilidade temporárias

Alguns módulos existem apenas para que importações antigas continuem
funcionando e podem ser removidos quando nenhum código ou teste depender deles:

- `jmds_prepare.protocols` reexporta o que hoje vive em `sources.jmds` e
  `sources.asvspoof`.
- `jmds_prepare.zenodo` reexporta `sources.zenodo` e as funções de
  `storage.download`.
- Aliases privados em `jmds_prepare.cli` (por exemplo `_write_json_atomic`,
  `_is_valid_flac`). Há dois tipos:
  - aliases injetados: um wrapper do `cli` lê o valor atual a cada chamada,
    então substituí-los em `cli` altera o comportamento;
  - aliases só de importação: existem para não quebrar imports; para mudar o
    comportamento é preciso substituir o objeto no módulo que o chama.

Código novo deve importar da implementação canônica (`core`, `storage`,
`sources`, `profiles`), não das fachadas.

## Fluxo do inglês

O inglês usa o áudio gerado do JMDS v2 e o áudio pristine oficial do
ASVspoof 5 (Zenodo). Só `train` e `dev` são preparados; `eval` é excluído de
forma explícita pelo perfil, porque os IDs do eval inglês do JMDS não
correspondem ao protocolo oficial.

1. `validate-protocols`: compara os protocolos JMDS e ASVspoof 5 de train/dev
   e grava `reports/correspondence.json`. O eval é lido de forma estrita
   apenas para registrar a divergência conhecida.
2. `acquire-english --dry-run`: mostra tamanhos e espaço necessários sem criar
   nem apagar nada.
3. `acquire-english`: baixa cada TAR oficial, confere tamanho e MD5, extrai
   somente os FLAC pedidos pelo protocolo, registra cada um no ledger e gera o
   manifesto raw (`manifests/english_raw.csv`).
4. `process-english`: gera cópias WAV PCM-16 mono a 16 kHz e o manifesto
   processado. Os arquivos brutos não são alterados.
5. `audit-english`: decodifica fontes e saídas e publica os relatórios de
   auditoria e a baseline técnica.

Os passos 4 e 5 foram executados sobre as 154.696 amostras reais de train/dev.
A auditoria não encontrou paths ausentes, hashes de fonte/saída duplicados,
falantes compartilhados entre splits ou grupos-fonte compartilhados entre
splits. A baseline técnica simples obteve balanced accuracy 0,4953 e ROC AUC
0,4544; isso não demonstrou um atalho técnico simples nas variáveis avaliadas,
mas também não substitui as validações experimentais futuras.

## Fontes externas futuras: aquisição separada

`python -m jmds_prepare.source_acquisition` baixa originais descritos por
catálogos em `configs/sources/` (hoje AISHELL-3 e CORAA v1.1). É independente
dos quatro comandos ingleses: não extrai, não converte e não escreve em
`data_root`. Veja [sources.md](sources.md).

## Metadados portugueses (CORAA + JMDS/MLAAD)

Existe um CLI autônomo, separado de `jmds_prepare.cli`:

```powershell
python -m jmds_prepare.portuguese_metadata --config configs/portuguese-metadata.yaml
```

Ele usa `profiles/portuguese.py`, os adapters estritos `sources/coraa.py` e
`sources/jmds_mlaad.py`, os builders em `pipelines/metadata_profiling.py` e
`pipelines/metadata_provenance.py`, e a orquestração em
`pipelines/metadata_extraction.py`. A publicação dos seis artefatos passa por
`core/publication.py` (idempotente por conteúdo). O layout está em
`storage/metadata_layout.py`; a configuração YAML em `metadata_config.py`.

O pipeline lê os três CSVs nativos do CORAA e o protocolo JMDS por split,
filtra somente linhas `por/MLAAD/generated`, resolve cada WAV em
`Portugese_MLAAD_Generated/{split}/wav/` e publica dois manifestos CSV e quatro
relatórios JSON sob `output_root`. Não extrai áudio CORAA, não altera arquivos
fonte e não mapeia IDs pristine do JMDS para caminhos CORAA. O resumo mantém
as fontes separadas (`paired_samples: false`). O split `eval` português é
inventariado, mas não recebe papel experimental automático.

Os protocolos JMDS completos são multi-corpus e multi-idioma. Neles podem
existir IDs com prefixo `U_` e IDs repetidos entre corpora sem que isso invalide
o arquivo de origem. Por isso, o adapter valida no protocolo completo apenas o
esquema e as combinações semânticas relevantes para o português; a regra de
formato por split e a unicidade de `utt_id` são aplicadas depois do filtro,
exclusivamente ao subconjunto `por/generated/MLAAD`. Ampliar essas duas regras
ao protocolo inteiro rejeitaria dados JMDS v2 reais e não deve ser usado como
"correção" de validação.

## Metadados mandarins (AISHELL-3 + JMDS/ADD)

Existe um CLI autônomo, separado de `jmds_prepare.cli`:

```powershell
python -m jmds_prepare.mandarin_metadata --config configs/mandarin-metadata.yaml
```

Ele usa `profiles/mandarin.py`, o adapter de streaming `sources/aishell3.py`, o
adapter estrito `sources/jmds_add.py`, os builders em
`pipelines/mandarin_metadata_profiling.py` e
`pipelines/mandarin_metadata_provenance.py`, e a orquestração genérica em
`pipelines/metadata_extraction.py` (`extract_two_source_metadata`). A
publicação dos seis artefatos passa por `core/publication.py` (idempotente por
conteúdo). O layout está em `storage/mandarin_metadata_layout.py`; a
configuração YAML em `mandarin_metadata_config.py`.

O pipeline faz uma passagem sequencial sobre `data_aishell3.tgz` sem extrair ou
decodificar membros `.wav` (somente caminho e tamanho em bytes), reconcilia
utterances de `content.txt` com membros WAV do archive e publica o manifesto
pristine. Para JMDS, lê os três protocolos por split, filtra somente linhas
`zho/ADD/generated`, resolve cada WAV em
`Chinese_ADD_Generated/{split}/wav/` e exclui as 4.410 linhas pristine
`zho/AISHELL3` do manifesto gerado. Não mapeia IDs pristine do JMDS para nomes
de utterance do archive AISHELL-3.

Contagens esperadas sobre dados reais: AISHELL-3 88.035 linhas após
reconciliação (train 63.262; test com 24.773 linhas observadas em
`test/content.txt` versus 23.262 amostras test reportadas oficialmente — a
divergência aparece no profile e na proveniência); JMDS/ADD generated 24.642
(train 7.146, dev 7.497, eval 9.999) com resolução 1:1 de WAV. Campos esparsos
do ADD (`attack_id`, `gender`, `spk_id` = `unk`) não são inferidos. O resumo
mantém as fontes separadas (`paired_samples: false`).

A execução sobre dados locais é opt-in: copie
`configs/mandarin-metadata.example.yaml`, aponte `jmds_root`, `aishell_archive` e
`output_root` fora do `data_root` inglês e execute o comando acima. Config e
saídas derivadas ficam no `.gitignore`.

## Como adicionar um corpus sem espalhar `if language`

A regra de projeto é que a diferença entre corpora vive em dados e em módulos
dedicados, nunca em ramificações por idioma dentro de código compartilhado.
Passos esperados:

1. Criar um perfil em `profiles/<corpus>.py`, uma dataclass imutável como a de
   `profiles/english.py`, com idioma, nome do dataset, splits, contagens
   esperadas, `metadata_source` e política de exclusão.
2. Criar o leitor da origem em `sources/<origem>.py`, com validação estrita
   (colunas, IDs, duplicatas, rótulos). O leitor devolve dados no esquema
   comum; quem consome não precisa saber de onde vieram.
3. Reutilizar `storage` e `core` sem alterá-los. O layout já é parametrizado
   por `language_slug` e `source_slug` (`DataLayout`); basta construir um
   layout para o novo par.
4. Criar um pipeline (ou parametrizar o existente) que receba o perfil e o
   leitor como argumentos. O perfil entra por parâmetro; o pipeline não decide
   nada olhando o idioma.
5. Registrar um comando novo no `cli` que apenas ligue perfil, leitor e
   pipeline. Se o novo corpus precisar de uma decisão de formato, ela é
   resolvida no perfil ou no leitor, não com `if` no pipeline.
6. Adicionar testes de contrato do esquema e um teste de camada se uma
   dependência nova for introduzida.

Se aparecer a tentação de escrever `if language == ...` fora de `profiles`,
isso indica que um dado está faltando no perfil ou que o leitor da origem está
incompleto.

## Metadados nativos dos adapters futuros

Esta seção vale para os adapters de CORAA e MLAAD (português) e de AISHELL-3 e
ADD (mandarim).

- O adapter deve obter falante, gênero, transcrição, split e demais campos dos
  metadados nativos do próprio corpus.
- O adapter nunca deve copiar os metadados pristine anonimizados do JMDS para
  as amostras do corpus original, a menos que exista um mapa
  original–anonimizado comprovado, por amostra, entre os IDs do JMDS e os
  arquivos do corpus. Hoje esse mapa não existe publicamente para CORAA nem
  para AISHELL-3 (veja o [BACKLOG](../BACKLOG.md)).
- Sem esse mapa, os IDs pristine do JMDS não identificam amostras do corpus
  original, e qualquer vínculo por ID seria suposição.
- Cada campo do manifesto deve registrar sua procedência (`metadata_source`),
  como já ocorre no inglês.

## Limites atuais

- `config`, `manifest`, `audio` e `audit` ainda usam o perfil inglês
  diretamente (por exemplo, `PreparationConfig.validate` exige os splits e
  arquivos do inglês, e a validação do manifesto aceita só `eng`).
- Os quatro comandos de áudio expostos em `cli.py` são apenas ingleses.
  Metadados portugueses e mandarins têm CLIs autônomos; preparação de áudio
  português ou mandarim ainda não existe.
- Adapters estritos existem para CORAA, JMDS/MLAAD generated (português),
  AISHELL-3 streaming e JMDS/ADD generated (mandarim), mas só para inventário
  de metadados.
- Artefatos derivados do CORAA permanecem locais e não devem ser commitados nem
  redistribuídos (política conservadora CC BY-NC-ND 4.0).
