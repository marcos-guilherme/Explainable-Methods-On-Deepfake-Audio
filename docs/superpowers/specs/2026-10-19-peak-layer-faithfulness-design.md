# Fidelidade por intervenção nas camadas de pico

## Objetivo

Comparar a fidelidade DFT-LRP da camada 12 com uma camada de pico
pré-especificada para cada encoder: HuBERT Base na camada 2, Wav2Vec2 Base na
camada 5 e WavLM Base na camada 7. O experimento deve distinguir baixa
fidelidade arquitetural de degradação específica da última camada sem substituir
nem reinterpretar retrospectivamente o baseline já publicado.

## Escopo experimental

- Manter a camada 12 como baseline para todos os encoders.
- Executar uma camada de pico por encoder: `hubert_base=2`,
  `wav2vec2_base=5`, `wavlm_base=7`.
- Preservar o protocolo atual: células diagonais, mesma coorte XAI persistida,
  oito exemplos por classe, mesmos valores de `k`, controles aleatórios e
  inferiores, energia RMS igualada e intervalos bootstrap.
- Não executar células fora da diagonal nem selecionar camadas a partir dos
  resultados de fidelidade.
- Tratar 2/5/7 como escolhas exploratórias pré-especificadas pela análise de
  desempenho anterior, não como novas camadas confirmatórias selecionadas por
  este experimento.

## Interface e execução

`layerwise_faithfulness` aceitará `--layer-map` com pares
`perfil=camada`. Quando ausente, o comportamento continuará restrito à camada
12 para compatibilidade.

Com `--layer-map`, cada raiz de suíte executará somente a camada associada ao
perfil nela contido. Perfis ausentes, duplicados, camadas fora de 1--12 ou mapas
que não correspondam às raízes fornecidas causarão erro antes de carregar os
encoders.

Os resultados usarão o diretório já parametrizado por camada. Assim, a execução
de pico não sobrescreverá a geração ativa da camada 12 e continuará resumível e
reprodutível pelo fingerprint existente, atualizado para registrar o escopo
multicamada.

## Generalização do protocolo

As validações deixarão de exigir especificamente a camada 12 e passarão a
exigir:

1. camada válida e explicitamente solicitada;
2. origem igual ao alvo;
3. coorte autoritativa persistida para a mesma camada e célula;
4. probe e limiar correspondentes à mesma camada;
5. equivalência numérica do detector antes das intervenções.

Avisos e manifests declararão a camada efetivamente executada. Permanecem as
limitações sobre filtragem STFT, controle de energia, ausência de causalidade no
mundo real e ausência de transferência fora da diagonal.

## Integração ao relatório

O carregador reunirá os resultados das camadas 12 e de pico sem exigir uma
camada global única. A camada 12 continuará alimentando a síntese principal
existente para manter comparabilidade histórica.

Uma nova subseção e uma nova tabela/figura compararão, por encoder:

- camada 12 versus camada de pico;
- contagem e proporção de intervalos inteiramente acima de zero;
- diferença média das intervenções principais menos aleatórias e principais
  menos inferiores;
- classes e idiomas separadamente, sem agregar observações incompatíveis.

A narrativa só atribuirá a baixa fidelidade à última camada quando o pico mostrar
melhora consistente nos dois controles e em múltiplas células. Resultados mistos
serão descritos como evidência inconclusiva.

## Falhas e retomada

- Entradas inválidas falham antes do uso de GPU.
- Uma célula incompleta não será publicada como geração ativa.
- Gerações válidas existentes serão retomadas pelo fingerprint; `--force`
  continuará disponível.
- O relatório falhará se encontrar schemas incompatíveis, camadas inesperadas ou
  cobertura parcial que possa produzir uma comparação enganosa. Ausência total
  das camadas de pico manterá o relatório antigo válido, sem a nova subseção.

## Testes e aceitação

1. Testes unitários cobrirão parsing e validação de `--layer-map`.
2. Os testes atuais provarão que a execução padrão ainda usa somente a camada
   12.
3. Testes do runner provarão que cada perfil usa sua camada mapeada e que probe,
   coorte e destino pertencem à mesma camada.
4. Testes do relatório cobrirão carregamento multicamada, preservação do
   baseline, cálculo da comparação e ausência da subseção quando não houver pico.
5. Um smoke test na VM validará uma célula antes da rodada completa.
6. A rodada final deverá concluir nove células de pico, regenerar o relatório e
   compilar o PDF em duas passagens sem referências indefinidas nem caixas
   horizontais excedentes.

## Fora de escopo

- Selecionar automaticamente a melhor camada pela própria fidelidade.
- Reexecutar todas as 12 camadas.
- Alterar o tamanho da coorte, os controles ou o protocolo bootstrap.
- Inferir causalidade sobre objetivos de pré-treino ou pistas acústicas.
