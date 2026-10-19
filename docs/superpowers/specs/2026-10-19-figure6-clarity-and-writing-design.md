# Redesenho da Figura 6 e revisão da escrita científica

## Objetivo

Substituir a atual Figura 6, sobre mudança de relevância entre
corpora/idiomas, por uma visualização que priorize padrões ao longo das camadas.
Em paralelo, remover do corpo principal frases de implementação, proveniência
operacional e metadiscurso que não contribuem para a interpretação científica.
Os detalhes necessários à reprodutibilidade permanecerão nos apêndices e
artefatos suplementares.

## Problemas atuais

A Figura 6 combina nove painéis, três pares de alvos e 324 intervalos
bootstrap. As hastes se sobrepõem e dificultam reconhecer padrões por camada,
encoder e domínio. O título e os eixos também exibem expressões internas como
“intervalos persistidos”, nomes de perfis e “JS (bits)”.

O rótulo em bits é impreciso: a implementação usa
`scipy.spatial.distance.jensenshannon`, que retorna a raiz quadrada da
divergência Jensen--Shannon. A base 2 controla a escala, mas a distância não
deve ser apresentada como uma quantidade diretamente medida em bits.

## Nova Figura 6

A figura principal terá três heatmaps empilhadas, uma por encoder.

- Colunas: camadas C1--C12.
- Linhas: nove combinações, agrupadas pelo idioma em que o probe foi treinado.
- Dentro de cada grupo: inglês--português, inglês--mandarim e
  português--mandarim.
- Cor: distância Jensen--Shannon entre os perfis médios de relevância dos dois
  conjuntos-alvo.
- Escala: única e compartilhada pelos três encoders.
- Separadores horizontais: limites entre probes treinados em inglês, português
  e mandarim.
- Barra de cor: “Distância JS (0 = perfis idênticos; log base 2)”.

Os painéis usarão nomes científicos visíveis, não IDs internos. A figura não
mostrará intervalos bootstrap. O gráfico atual com estimativas pontuais e
intervalos será mantido no apêndice como diagnóstico de incerteza.

## Semântica apresentada ao leitor

Para uma camada, cada conjunto-alvo é representado pelo perfil médio de
relevância absoluta normalizada nas 24 bandas. A distância compara esses dois
perfis:

- 0 indica distribuições idênticas;
- valores maiores indicam maior redistribuição da relevância entre as bandas;
- o valor não é porcentagem;
- a distância não informa quais bandas ganharam ou perderam relevância;
- não há limiar universal para mudança pequena ou grande.

A interpretação será relativa ao estudo. O texto poderá comparar medianas entre
encoders, mas não converter `0,113` em “11,3% de diferença”. A Figura 5 será
indicada como complemento para identificar a direção espectral da mudança.

## Correção terminológica da métrica

Textos, eixos e legendas substituirão “bits” por “distância JS, log base 2” ou
“adimensional, log base 2”. O dado continuará sendo a distância retornada por
SciPy; nenhum valor será recalculado.

Os artefatos tabulares novos usarão uma unidade semanticamente explícita, como
`distance_base2`, em vez de `bits`. O schema de colunas permanecerá estável.
Testes deverão impedir que distância JS volte a ser apresentada como
divergência JS ou porcentagem.

## Revisão do corpo principal

A revisão cobrirá somente o conteúdo anterior ao início dos apêndices, incluindo
títulos e legendas das figuras principais. Devem ser removidos ou reescritos:

- nomes de arquivos, diretórios, manifests, hashes e IDs internos;
- expressões de armazenamento, como “persistido” ou “bundle”;
- descrições de geração automática e detalhes de implementação;
- contagens operacionais que não alteram a interpretação no corpo principal;
- metadiscurso como instruções sobre a estrutura do relatório;
- repetições entre resultado, interpretação, cautela e conclusão.

Devem permanecer:

- decisões metodológicas que afetam a validade, como probe fixo, camada,
  calibração e definição das métricas;
- tamanho amostral quando necessário para interpretar uma estimativa;
- intervalos e incerteza quando sustentam a conclusão;
- limitações sobre confundimento corpus/idioma e ausência de causalidade;
- referências aos apêndices para detalhes técnicos.

A revisão não reduzirá precisão científica para obter concisão.

## Texto da seção da Figura 6

A seção explicará que o encoder, o idioma de treino e o probe permanecem fixos,
enquanto o conjunto-alvo muda. O resultado principal apresentará a faixa
observada e as medianas relevantes por encoder, em linguagem direta.

A legenda deverá incluir:

> Distância JS entre perfis médios de relevância. Zero indica distribuições
> idênticas entre as 24 bandas; valores maiores indicam maior redistribuição da
> relevância ao trocar o corpus/idioma avaliado. A métrica não informa a direção
> da mudança.

O texto ligará explicitamente a Figura 6 à Figura 5 para localizar as bandas
responsáveis pela mudança.

## Material suplementar

O gráfico atual de pontos e intervalos será movido para o apêndice de
comparações. Sua legenda indicará que as hastes são intervalos bootstrap de 95%.
Os grupos agregado, real e sintético continuarão disponíveis no detalhamento
técnico existente.

## Validação

1. Testes verificarão a matriz de 9 linhas por 12 camadas para cada encoder.
2. Todos os painéis usarão a mesma normalização de cor.
3. A ordem será encoder, idioma do probe, par de alvos e camada conforme a
   convenção do relatório.
4. Testes verificarão os separadores de grupos e os rótulos científicos.
5. A Figura 6 principal não conterá hastes bootstrap nem IDs internos.
6. O gráfico de intervalos aparecerá exatamente uma vez no apêndice.
7. O corpo principal não conterá termos operacionais proibidos nem nomes de
   arquivos.
8. O texto não chamará distância JS de porcentagem ou de divergência em bits.
9. O PDF será compilado em duas passagens e inspecionado quanto a referências,
   caixas excedentes e legibilidade.

## Fora de escopo

- Alterar os valores calculados da distância JS.
- Remover os intervalos bootstrap dos dados suplementares.
- Usar a Figura 6 para afirmar efeito causal do idioma.
- Identificar direção espectral somente pela distância JS.
- Simplificar os apêndices técnicos além do necessário para acomodar a figura
  de incerteza.
