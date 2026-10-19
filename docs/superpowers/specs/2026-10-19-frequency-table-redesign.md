# Redesenho das tabelas de relevância espectral

## Problema

A tabela atual usa uma linha por encoder e idioma e chama a última coluna de
“Camadas”. Valores como “6 de 12” representam recorrência da faixa dominante,
mas podem ser interpretados como uma camada específica. A ordenação por encoder
também aproxima linhas de idiomas diferentes e favorece comparações em que mais
de uma variável muda simultaneamente.

## Tabela do corpo principal

A síntese principal será reorganizada para manter o idioma fixo em cada linha:

| Idioma | HuBERT Base | Wav2Vec2 Base | WavLM Base |
|---|---|---|---|
| inglês | faixa (n/12) | faixa (n/12) | faixa (n/12) |

Cada célula exibirá a faixa ou as faixas co-dominantes e sua recorrência entre as
12 camadas. A legenda definirá explicitamente que `n/12` é o número de camadas
em que a faixa atingiu a maior massa média de relevância, não o índice de uma
camada selecionada.

Essa organização permite comparar os três encoders mantendo o idioma constante.
Comparações verticais entre idiomas continuarão sendo descritas como
cross-corpus/cross-lingual e não como efeito isolado do idioma.

## Detalhe camada a camada no suplemento

O apêndice de relevância receberá três tabelas, uma por idioma. Cada tabela terá
12 linhas, correspondentes às camadas C1--C12, e três colunas de resultados,
uma por encoder:

| Camada | HuBERT Base | Wav2Vec2 Base | WavLM Base |
|---|---|---|---|
| C1 | faixa(s) dominante(s) | faixa(s) dominante(s) | faixa(s) dominante(s) |

Cada célula será calculada diretamente a partir da massa média por banda daquela
camada. Empates serão preservados como faixas co-dominantes segundo a mesma
regra da síntese. As tabelas não substituirão a Figura 5: a figura continuará
mostrando a distribuição completa da relevância nas 24 bandas, enquanto as
tabelas mostrarão apenas a faixa de maior massa.

## Dados e ordenação

- Idiomas seguirão a ordem inglês, português e mandarim.
- Encoders seguirão a ordem HuBERT Base, Wav2Vec2 Base e WavLM Base.
- Camadas seguirão C1--C12.
- Faixas usarão os limites mel persistidos de cada experimento.
- Nenhuma camada será escolhida por desempenho ou fidelidade para essas tabelas.

## Validação

1. Testes provarão que a tabela principal contém uma linha por idioma e mantém
   os três encoders na mesma linha.
2. Testes provarão que `n/12` corresponde à contagem das camadas dominadas pela
   faixa apresentada.
3. Testes provarão que cada combinação idioma--encoder--camada aparece uma única
   vez nas tabelas suplementares.
4. Empates artificiais em fixtures deverão produzir todas as faixas
   co-dominantes.
5. O PDF será compilado em duas passagens e verificado quanto a referências,
   largura das tabelas e legibilidade.

## Fora de escopo

- Reduzir a Figura 5 a uma única faixa por camada.
- Interpretar a faixa dominante como pista causal.
- Comparar idiomas como se corpus e condições acústicas fossem controlados.
- Alterar as bandas mel ou recalcular explicações.
