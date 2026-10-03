# Extração de metadados portugueses

## Objetivo

Extrair e validar os metadados nativos das duas fontes portuguesas sem alterar
os originais nem afirmar uma correspondência inexistente entre elas:

- CORAA v1.1 para áudio pristine;
- JMDS v2 / MLAAD para áudio generated.

O resultado deve permitir caracterizar o português e definir, posteriormente,
um subconjunto comparável entre idiomas. A seleção experimental e a extração
dos arquivos de áudio ficam fora deste escopo.

## Evidência de entrada

O inventário somente leitura encontrou:

- CORAA: 402.456 linhas, sendo 382.258 train, 7.522 dev e 12.676 test; todos os
  `file_path` são preenchidos e únicos;
- JMDS português generated: 3.011 linhas com `language=por`, sendo 1.806 train,
  602 dev e 603 eval; `dataset=MLAAD` em todas e nenhum `utt_id` duplicado.

Os 1.000 registros pristine portugueses do protocolo JMDS não entram no
manifesto CORAA, pois não existe mapa público por amostra entre seus IDs
anonimizados e os arquivos CORAA.

## Arquitetura

As regras portuguesas ficam em um perfil imutável. Cada origem tem um adapter
estrito em `sources`:

- o adapter CORAA valida os três CSVs, preserva os 15 campos nativos e adiciona
  apenas campos calculados de rastreabilidade;
- o adapter JMDS/MLAAD valida o protocolo completo antes de filtrar
  `language=por`, `label=generated` e `dataset=MLAAD`.

Um pipeline recebe perfil e adapters por parâmetro, escreve os artefatos de
forma atômica e não contém ramificações por idioma. Uma interface de linha de
comando apenas liga os componentes e recebe os caminhos.

## Artefatos

O destino local terá:

- `manifests/coraa_metadata.csv`: campos nativos CORAA mais `split`,
  `metadata_source_file`, `metadata_source_row` e identidade da versão;
- `manifests/jmds_mlaad_generated_metadata.csv`: os nove campos nativos JMDS
  mais `split`, `metadata_source_file`, `metadata_source_row` e, quando puder
  ser resolvido sem suposição, `audio_path`;
- `reports/coraa_metadata_profile.json`: relação dos campos CORAA com
  descrição, tipo observado, nulabilidade, procedência e valores ou
  distribuições resumidas, além das contagens por split;
- `reports/jmds_mlaad_generated_metadata_profile.json`: relação equivalente
  para o JMDS/MLAAD, incluindo distribuições de ataque, gênero, codec, native,
  split e disponibilidade do áudio;
- `reports/portuguese_metadata_summary.json`: contagens e distribuições por
  split, fonte, subcorpus, sotaque, estilo, tarefa, ataque, gênero, codec,
  native e indicadores de qualidade, além das características comuns usadas
  para comparar as duas fontes sem associar suas amostras;
- `reports/portuguese_metadata_provenance.json`: versão/revisão, licença,
  política de uso, SHA-256 e tamanho de cada CSV/protocolo de entrada, esquema
  dos artefatos e procedência de cada campo.

Os CSVs de saída preservam uma linha por registro de origem. Não haverá um
manifesto combinado com união esparsa de colunas.

## Validação e erros

O processo deve falhar antes de publicar qualquer artefato quando encontrar:

- arquivo ou coluna obrigatória ausente, extra ou fora da ordem esperada;
- split incompatível com o arquivo;
- path ou ID vazio, inseguro ou duplicado dentro da origem;
- rótulo, idioma ou dataset inesperado no subconjunto generated;
- mudança nas contagens portuguesas esperadas do JMDS;
- destino parcialmente existente com conteúdo divergente.

Como o áudio CORAA ainda está arquivado, seu `file_path` será validado
sintaticamente e contra o split, não por existência no disco. A existência dos
WAVs JMDS será registrada separadamente, sem transformar ausência em vínculo
com CORAA.

Publicação usa temporários no mesmo diretório, flush/fsync e promoção atômica.
Uma repetição idêntica é idempotente; conteúdo divergente não é sobrescrito.

## Testes

O desenvolvimento será test-first. Fixtures pequenas cobrirão:

- leitura válida de cada fonte e preservação dos campos nativos;
- esquema, ordem, tipos textuais, paths, IDs, duplicatas e splits inválidos;
- filtro português generated sem aceitar os pristine anonimizados;
- proveniência e hashes das entradas;
- dicionários JSON separados, contagens e distribuições do resumo;
- publicação atômica, repetição idêntica e conflito de conteúdo;
- regra arquitetural: pipeline não importa CLI e não decide pelo idioma.

Depois da suíte unitária, a extração será executada sobre os metadados reais.
As contagens, unicidade, hashes e artefatos publicados serão conferidos sem
extrair ou modificar áudio.

## Limites científicos e de licença

Os dois manifestos descrevem corpora distintos. Eles não formam pares
real–fake, e igualdade de texto, ordem ou split não autoriza associação entre
amostras. O desequilíbrio entre 402.456 pristine e 3.011 generated será
resolvido somente na etapa posterior de seleção experimental.

Os artefatos derivados do CORAA permanecem locais. A política conservadora
registrada para CC BY-NC-ND 4.0 impede sua publicação ou redistribuição neste
projeto.
