# Metodologia dos conjuntos experimentais trilíngues para explicabilidade em áudio deepfake

**Estado empírico documentado em:** 5 de outubro de 2026.

Este estudo não reúne três conjuntos de dados prontos. Ele constrói três
**conjuntos experimentais**, um para cada idioma. Cada conjunto combina uma
fonte de áudio humano ou autêntico, identificada pelo rótulo técnico
`bonafide`, e uma fonte de áudio sintético ou manipulado, identificada pelo
rótulo técnico `spoof`.

Os três conjuntos experimentais têm o mesmo tamanho e as mesmas funções no
experimento, mas suas fontes não são equivalentes. Por isso, comparações entre
idiomas também envolvem mudanças de conjunto de origem, gravação, locutores,
conteúdo e geração. O desenho permite estudar desempenho, robustez e
explicações de modelos nesses domínios;
não permite isolar um efeito causal do idioma.

## Mapa dos nomes

- **Inglês:** áudio humano de ASVspoof 5 e áudio sintético inglês distribuído no
  JMDS v2.
- **Português:** áudio humano de CORAA v1.1 e áudio sintético do subconjunto
  MLAAD distribuído no JMDS v2.
- **Mandarim:** áudio humano de AISHELL-3 e áudio sintético do subconjunto ADD
  distribuído no JMDS v2.

JMDS v2 é a liberação local e o protocolo que fornecem os áudios sintéticos
usados neste estudo. Dentro dessa liberação, os áudios em português são
identificados como MLAAD e os áudios em mandarim como ADD. No material de
protocolo e nos catálogos de amostras, o rótulo legado `ASVspoof2024` é
preservado para a fonte inglesa; neste capítulo, seu nome de exibição é sempre
ASVspoof 5.

## Glossário

- **Conjunto experimental:** subconjunto selecionado para um idioma.
- **Conjunto de origem:** conjunto do qual os áudios e metadados são obtidos.
  O termo técnico *corpus* também pode designar esse tipo de coleção.
- **Divisão original (split):** divisão publicada pelo conjunto de origem,
  como `train`, `dev`, `test` ou `eval`.
- **Função no experimento:** finalidade atribuída depois do mapeamento das
  divisões originais: treino, calibração ou teste.
- **Áudio humano:** classe 0, identificada formalmente pelo rótulo `bonafide`.
- **Áudio sintético:** classe 1, formada por fala sintética ou manipulada e
  identificada formalmente pelo rótulo `spoof`.
- **Catálogo de amostras:** registro linha a linha das amostras e de seus
  metadados; não é o conjunto de arquivos de áudio. O arquivo técnico que
  armazena esse catálogo recebe o nome de *manifesto*.
- **Pareamento:** restrição que aproxima as classes segundo uma variável
  observada. Pareamento por falante não significa pareamento por conteúdo.

## Visão rápida

### Conjunto experimental inglês

**Áudio humano:** ASVspoof 5

**Áudio sintético:** áudio sintético inglês distribuído no JMDS v2

**Total:** 6.022 amostras

### Conjunto experimental português

**Áudio humano:** CORAA v1.1

**Áudio sintético:** subconjunto MLAAD distribuído no JMDS v2

**Total:** 6.022 amostras

### Conjunto experimental mandarim

**Áudio humano:** AISHELL-3

**Áudio sintético:** subconjunto ADD distribuído no JMDS v2

**Total:** 6.022 amostras

## Objetivo e unidade de análise

A unidade de análise é um arquivo de áudio com identificador padronizado e único,
uma classe e uma única função no experimento: treino, calibração ou teste. Cada
arquivo é contado uma única vez em seu conjunto experimental.

O estudo foi desenhado para examinar:

- o desempenho de detectores de áudio deepfake dentro de cada conjunto
  experimental;
- a robustez externa quando idioma e conjunto de origem mudam simultaneamente;
- propriedades acústicas que podem funcionar como atalhos para o detector;
- explicações produzidas por métodos de explicação de modelos (XAI) sobre
  conjuntos fixos e reprodutíveis.

O balanceamento ocorre por classe dentro de cada função no experimento.
Contagens iguais não tornam as classes equivalentes em locutor, conteúdo,
duração ou condição de aquisição.

# Como os conjuntos experimentais são construídos

## Número de amostras e sua origem

O menor conjunto elegível é o subconjunto MLAAD distribuído no JMDS v2. Sua
disponibilidade determina quantas amostras são selecionadas por classe:

- treino: 1.806 amostras;
- calibração: 602 amostras;
- teste: 603 amostras.

São 3.011 amostras por classe e 6.022 por idioma. Os três conjuntos
experimentais somam 18.066 amostras.

Essas quantidades tornam os idiomas comparáveis em número de amostras e mantêm
separados o ajuste do modelo, a calibração de limiares e a avaliação final.
Elas decorrem da disponibilidade dos dados, não de uma análise de poder
estatístico. Portanto, não garantem
poder, precisão de intervalos ou detecção de um tamanho de efeito específico.
As análises estatísticas precisam declarar as medidas estimadas, como as
amostras são agrupadas nos cálculos e os intervalos de incerteza.

## Conjunto experimental inglês

### Fontes

A classe de áudio humano vem dos arquivos oficiais de ASVspoof 5. A classe de
áudio sintético vem do material inglês distribuído no JMDS v2. A
correspondência entre os protocolos das duas fontes foi validada para as
divisões originais `train` e `dev`.

### Mapeamento das divisões

A divisão original `train` de ambas as fontes alimenta a função de treino. A
divisão original `dev` de ambas é separada entre as funções de calibração e
teste.

Assim, o teste inglês vem de `dev`, e não de `eval`. O `eval` não é usado
porque seus identificadores no JMDS v2 não correspondem exatamente aos do
protocolo oficial de ASVspoof 5. Calibração e teste não têm sobreposição de
falantes nem, quando essa informação está disponível, de grupos de origem.

### Seleção e pareamento

A seleção inglesa é pareada por falante entre áudio humano e áudio sintético
dentro de cada função no experimento. Ela não pareia gravação, sentença nem
conteúdo. Portanto, não constitui uma comparação da mesma fala em versões
humana e sintética.

### Metadados disponíveis

O catálogo principal de amostras conserva, quando sustentados pelas fontes, o
identificador de falante, o grupo de origem e o identificador de ataque.
Informações técnicas e de protocolo mais detalhadas permanecem nos registros
que documentam a origem e o processamento dos dados.

### Limitação principal

Mesmo com pareamento por falante, as classes podem diferir em conteúdo,
duração e processamento. Além disso, o teste construído a partir de `dev` não
sustenta generalização para o `eval` oficial.

## Conjunto experimental português

### Fontes

A classe de áudio humano vem de CORAA v1.1. A classe de áudio sintético vem do
subconjunto MLAAD distribuído no JMDS v2. Não existe um mapa público amostra a
amostra entre os identificadores humanos anonimizados no JMDS v2 e os arquivos
de CORAA v1.1; esse vínculo não é presumido.

### Mapeamento das divisões

Em CORAA v1.1, `train`, `dev` e `test` alimentam, respectivamente, treino,
calibração e teste. No subconjunto MLAAD distribuído no JMDS v2, `train`, `dev`
e `eval` alimentam, respectivamente, treino, calibração e teste.

### Seleção e pareamento

Não há pareamento entre CORAA v1.1 e o subconjunto MLAAD distribuído no JMDS
v2. O número previsto de amostras é preenchido separadamente por classe e
função no experimento.

### Metadados disponíveis

CORAA v1.1 oferece texto, variedade, sotaque, gênero discursivo, estilo de
fala, tarefa e votos de qualidade. O catálogo de amostras usado neste estudo
não oferece `speaker_id` nem gênero demográfico. Esses campos não são inferidos
de diretórios, nomes de arquivo ou texto.

O protocolo do subconjunto MLAAD distribuído no JMDS v2 inclui campos como
ataque, gênero declarado, formato de codificação (*codec*), condição original e
conjunto de origem. Esses campos nem sempre estão completos ou são diretamente
comparáveis aos de CORAA v1.1.

### Limitação principal

A classe e o conjunto de origem mudam juntos: todo áudio humano vem de CORAA
v1.1, e todo áudio sintético vem do subconjunto MLAAD distribuído no JMDS v2.
Assim, não é possível separar o efeito da classe do efeito do conjunto de
origem (confusão classe–corpus). Uma diferença atribuída à classe pode refletir
gravação, segmentação, conteúdo ou processamento. A ausência de `speaker_id`
também impede afirmar separação sem sobreposição de locutores.

## Conjunto experimental mandarim

### Fontes

A classe de áudio humano vem de AISHELL-3. A classe de áudio sintético vem do
subconjunto ADD distribuído no JMDS v2. Identificadores humanos anonimizados no
JMDS v2 não são tratados como nomes de gravações de AISHELL-3.

### Mapeamento das divisões

A divisão original `train` de AISHELL-3 é separada por falante entre treino e
calibração. Sua divisão `test` alimenta a função de teste. No subconjunto ADD
distribuído no JMDS v2, `train`, `dev` e `eval` alimentam, respectivamente,
treino, calibração e teste.

### Seleção e pareamento

Não há pareamento entre AISHELL-3 e o subconjunto ADD distribuído no JMDS v2.
Na divisão interna de AISHELL-3, os falantes usados em treino não são
reutilizados em calibração.

### Metadados disponíveis

AISHELL-3 oferece identificador de falante, gênero, idade, sotaque,
transcrição, informação fonética ou prosódica e divisão original. O protocolo do
subconjunto ADD distribuído no JMDS v2 contém campos equivalentes aos demais
protocolos da liberação, mas vários são esparsos: ataque, gênero e falante
aparecem como desconhecidos, e codec não apresenta categoria informativa.
Valores ausentes não são inferidos.

### Limitação principal

Também aqui a classe e o conjunto de origem mudam juntos: todo áudio humano
vem de AISHELL-3, e todo áudio sintético vem do subconjunto ADD distribuído no
JMDS v2. Assim, este desenho não permite separar os efeitos da classe dos
efeitos do conjunto de origem.

## Seleção determinística e separação experimental

Todas as seleções usam a semente de inicialização (*seed*) 42. A prioridade de
cada candidato é determinada de forma reprodutível pela seed e por seu
identificador; a ordem alfabética do identificador resolve empates. Para as
mesmas entradas e regras de inclusão, a seleção é repetível.

O protocolo exige:

- identificadores candidatos únicos na população elegível;
- referências de origem únicas;
- identificadores padronizados e caminhos processados únicos;
- número exato de amostras por idioma, função no experimento e classe;
- nenhuma reutilização da mesma amostra entre funções no experimento;
- separação por `speaker_id` e `group_id` quando esses campos existem e uma
  divisão original alimenta mais de uma função;
- separação por falante entre treino e calibração em AISHELL-3;
- correspondência dos conjuntos de falantes entre classes no inglês, sem
  exigir correspondência de conteúdo.

A ordem e a posição de seleção são registradas para auditoria. Como mudanças
nos catálogos de amostras de origem podem alterar a população elegível, a
reprodução também depende dos hashes dessas entradas.

Calibração é reservada a decisões de limiar e a hiperparâmetros previstos no
protocolo. O teste não participa dessas decisões.

## Representação do áudio

Cada amostra selecionada é representada como WAV mono, PCM16 — áudio sem
compressão com amostras de 16 bits — a 16 kHz. Os arquivos originais são
preservados. O WAV padronizado pode ser uma cópia convertida ou um
arquivo já compatível e validado.

Na construção dos conjuntos experimentais, não há normalização de duração,
recorte ou preenchimento para tamanho fixo, ajuste do volume percebido,
remoção de silêncio, redução de ruído ou aumento de dados. Qualquer adaptação
posterior à entrada de um modelo pertence ao protocolo desse modelo e precisa
ser relatada separadamente.

## Esquema conceitual dos metadados

O catálogo principal de amostras contém o conjunto mínimo de informações
usado por todos os idiomas. Cada linha
descreve uma unidade de análise por meio dos seguintes campos:

- `sample_id`: identificador estável e único;
- `language`: idioma do conjunto experimental (`eng`, `por` ou `zho`);
- `role`: função no experimento (`train`, `calibration` ou `test`);
- `label`: 0 para bonafide e 1 para spoof;
- `corpus`: conjunto de origem responsável pelo áudio;
- `native_split`: divisão original antes do mapeamento;
- `original_ref`: referência ao áudio de origem;
- `processed_path`: localização do WAV padronizado;
- `speaker_id`: falante, apenas quando sustentado pela fonte;
- `group_id`: grupo usado para impedir vazamento de variantes relacionadas;
- `attack_id`: categoria de ataque para spoof, quando informada;
- `sha256_source`: hash SHA-256 do conteúdo de origem lido;
- `sha256_processed`: hash SHA-256 do WAV padronizado;
- `selection_seed`: seed da seleção;
- `selection_rank`: posição da amostra em seu grupo de seleção;
- `selection_reason`: regra científica que justificou a inclusão;
- `selection_source`: catálogo de amostras ou registro que comprovou que a
  amostra atendia aos critérios de inclusão.

`original_ref` pode revelar caminhos locais ou a estrutura da infraestrutura.
Antes de qualquer compartilhamento, esses dados sensíveis devem ser removidos.
`attack_id` permanece vazio para o rótulo
formal `bonafide`. Em qualquer campo, vazio significa ausência documentada,
não autorização para atribuir um valor.

Neste documento, *hash* é um código usado para verificar que os bytes não
mudaram. O algoritmo adotado é o SHA-256.

Os metadados nativos mais ricos de CORAA v1.1, AISHELL-3 e dos subconjuntos
distribuídos no JMDS v2 não são todos copiados para esse catálogo mínimo. Eles
permanecem nos registros do histórico de origem para rastreabilidade e análises
secundárias.

# Como a qualidade é auditada

## Auditoria acústica

Todos os WAV padronizados são decodificados e verificados quanto a formato, taxa
de amostragem, número de canais e subtipo PCM. Para cada combinação de
conjunto de origem, classe e função no experimento, a auditoria resume:

- duração em segundos;
- RMS, uma medida de energia média, em escala linear de amplitude;
- pico absoluto;
- fração de silêncio, definida como a proporção de amostras com
  `|x| <= 1e-4`.

Essas medidas ajudam a detectar potenciais atalhos. Sozinhas, não medem
qualidade perceptual, inteligibilidade ou naturalidade.

## Integridade e histórico de origem

A rastreabilidade inclui:

- hash SHA-256 do áudio de origem e do WAV padronizado para cada amostra;
- hash SHA-256 e tamanho dos catálogos de amostras de entrada e dos arquivos de
  origem relevantes;
- verificação de referências existentes e legíveis;
- unicidade de identificadores, referências e caminhos;
- contagens exatas por idioma, função no experimento e classe;
- registro da seed, das regras de separação e do ambiente de preparação;
- preservação das reconciliações, licenças, revisões e justificativas para
  metadados ausentes.

Um hash calculado localmente permite verificar se os arquivos mudaram depois
da auditoria. Ele não deve ser apresentado como um valor oficial da fonte
quando ela não publicou um hash para comparação.

## Bundle de dados e licenças

O bundle de dados é um conjunto autocontido, organizado para ser transferido à
máquina virtual (VM). Ele deve conter somente os conjuntos experimentais
selecionados, os catálogos de amostras sem caminhos locais ou outros dados
sensíveis e um inventário de
integridade. Não deve conter os conjuntos de origem completos. Isso fixa as
unidades experimentais e evita uma nova seleção no ambiente de execução.

O inventário de integridade, armazenado no arquivo técnico
`bundle_receipt.json`, enumera arquivo, idioma, função no experimento, classe,
tamanho e hash SHA-256. Para a transferência, o bundle é compactado em
`tar.zst`. Tanto o arquivo compactado quanto o conteúdo extraído precisam ser
verificados na VM.

O bundle não é automaticamente redistribuível:

- **CORAA v1.1:** CC BY-NC-ND 4.0. A política conservadora deste estudo mantém
  originais e derivados no ambiente autorizado, restringe o uso ao contexto
  não comercial e não redistribui áudio derivado, características extraídas ou
  subconjuntos.
- **AISHELL-3:** Apache 2.0 segundo a página oficial OpenSLR. Eventual
  redistribuição precisa conservar licença e atribuição.
- **ASVspoof 5:** a licença e a versão declaradas no registro oficial do
  Zenodo devem acompanhar o histórico de origem.
- **JMDS v2:** os metadados locais usados aqui não declaram licença separada
  para MLAAD ou ADD. Essa ausência deve permanecer explícita e não autoriza
  inferir permissão de redistribuição.

Quando contém áudio de CORAA v1.1, o bundle permanece local ao ambiente
autorizado e não pode ser redistribuído. Mesmo um catálogo de amostras sem
áudio pode exigir a remoção de dados sensíveis de `original_ref` e uma nova
análise de licença.

Esta é uma política operacional de pesquisa, não um parecer jurídico.

# O que os resultados permitem afirmar

## Interpretações sustentadas

Os resultados podem ser descritos como:

- desempenho dentro de cada conjunto experimental e do protocolo definido;
- robustez externa sob mudança conjunta de idioma e conjunto de origem;
- explicações do comportamento do modelo auditado;
- diferenças acústicas que indicam potenciais atalhos a controlar.

Quando o idioma de treino difere do idioma de teste, o resultado mede o
comportamento do detector diante da mudança simultânea de idioma e conjunto de
origem. Ele não demonstra mecanismos linguísticos universais.

## Interpretações não sustentadas

Este desenho, isoladamente, não permite concluir:

- causalidade linguística;
- superioridade geral de um idioma, conjunto de origem ou gerador;
- que uma explicação fiel ao modelo corresponde a um mecanismo humano ou
  causal;
- que diferenças entre áudio humano e áudio sintético decorrem exclusivamente da
  autenticidade;
- generalização para o `eval` oficial inglês ou para conjuntos de origem não
  observados.

As ameaças centrais são a mudança conjunta de classe e conjunto de origem em
português e mandarim, cujos efeitos não podem ser separados. Também são
ameaças os potenciais atalhos acústicos, a ausência de pareamento por conteúdo,
a falta de `speaker_id` no português, o uso de `dev` para o teste inglês, a
seleção com uma única seed, as quantidades definidas sem análise de poder, os metadados
esparsos e eventuais adaptações posteriores exigidas pelos modelos.

## Checklist para o artigo

O artigo deve informar:

- data e versão do protocolo;
- conjuntos de origem e associação entre conjunto e classe em cada idioma;
- links oficiais, revisões, licenças e restrições de uso;
- contagens por idioma, função no experimento, classe e conjunto de origem;
- origem do número de amostras e ausência de análise de poder;
- seed 42 e regra determinística de seleção;
- critérios de unicidade, separação e tratamento de falantes e grupos;
- mapeamento das divisões originais, com destaque para o teste inglês vindo de
  `dev`;
- pareamento inglês por falante, nunca por gravação ou conteúdo;
- ausência de pareamento em português e mandarim;
- ausência de `speaker_id` no catálogo de amostras português;
- formato WAV mono PCM16 a 16 kHz e ausência de normalização de duração;
- recorte, preenchimento ou normalização eventualmente aplicados pelo modelo;
- definição de silêncio como `|x| <= 1e-4`;
- duração, RMS, pico e silêncio por conjunto de origem, classe e função no
  experimento;
- hashes, referências ausentes, duplicatas e histórico de origem
  das entradas;
- metadados disponíveis, ausentes e não copiados para o catálogo principal de
  amostras;
- mudança conjunta de classe e conjunto de origem em português e mandarim;
- diferença de duração entre CORAA v1.1 e o subconjunto MLAAD distribuído no
  JMDS v2, além da diferença observada no treino inglês;
- estado efetivo do conjunto experimental mandarim e do bundle na submissão;
- unidade estatística, método de incerteza e correções por multiplicidade;
- resultados cruzados como validação sob mudança conjunta de conjunto de origem
  e idioma,
  sem alegação de causalidade linguística;
- política de acesso aos artefatos e proibição de redistribuir o bundle quando
  ele contiver CORAA v1.1.

## Referências oficiais

- **ASVspoof 5 — registro humano usado neste estudo:**
  <https://zenodo.org/records/14498691>
- **CORAA — repositório oficial:**
  <https://github.com/nilc-nlp/CORAA>
- **CORAA v1.1 — distribuição fixada neste estudo:**
  <https://huggingface.co/datasets/gabrielrstan/CORAA-v1.1>
- **Licença de CORAA:**
  <https://github.com/nilc-nlp/CORAA/blob/main/LICENSE>
- **AISHELL-3 — OpenSLR SLR93:**
  <https://openslr.org/93/>
- **Distribuição de AISHELL-3:**
  <https://www.openslr.org/resources/93/>
- **Apache License 2.0:**
  <https://www.apache.org/licenses/LICENSE-2.0>
- **Creative Commons BY-NC-ND 4.0:**
  <https://creativecommons.org/licenses/by-nc-nd/4.0/>

JMDS v2 foi usado a partir da liberação pública local e de seus protocolos. O
repositório deste estudo não registra links oficiais externos para JMDS v2,
MLAAD ou ADD; por isso, nenhuma URL ou referência bibliográfica é inventada.
A versão da liberação, seus protocolos e respectivos hashes devem
acompanhar os artefatos do histórico de origem.
