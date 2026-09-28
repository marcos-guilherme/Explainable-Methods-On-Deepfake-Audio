# Backlog de pesquisa

Ideias registradas para depois. Cada item traz a ideia, por que pode valer a pena
e o principal cuidado. Nada aqui está decidido.

## Extensão para análise entre idiomas

**Ideia.** Levar o arcabouço atual (bandas mel, associação H1, oclusão causal H2,
saliência DFT-LRP e validação de fidelidade) para além do português brasileiro,
comparando as pistas espectrais que o detector usa em idiomas diferentes.

**Por que pode valer.** A parte espectral não depende do idioma (a grade de 24 bandas
mel é a mesma), então dá para sobrepor idiomas num eixo comum de frequência e
perguntar, por exemplo, se a concentração de efeito causal nos graves é universal ou
específica do pt-BR, e se a adaptação a cada idioma move as pistas na mesma direção.

**Cuidado principal (confound).** Datasets de deepfake em idiomas diferentes costumam
vir com sistemas de síntese (TTS/vocoders), gravação e codecs diferentes. Uma
diferença observada pode ser do idioma ou do método de geração do falso (ou do canal).
Mitigações: usar um corpus multilíngue com o mesmo pipeline de síntese (ex.: MLAAD),
ou fixar o sistema de geração e variar só o idioma. Reportar como observacional, com o
confound explícito.

### Caminho A (espectral, baixo risco)
Rodar o mesmo pipeline num corpus multilíngue e sobrepor os perfis por idioma nas
lentes que já temos. Reaproveita praticamente todo o código.

### Caminho B (temporal/prosódico, novo e mais arriscado)
Motivação: idiomas diferem em ritmo, velocidade de fala e prosódia, uma dimensão
temporal que pode importar.

Ponto de partida da discussão: a representação populacional com eixo de tempo (STDFT-LRP
ao longo do tempo) **não** serve para isso, porque o tempo de relógio não está alinhado
entre enunciados (e menos ainda entre idiomas). Média no instante `t` mistura conteúdos
fonéticos não relacionados.

Para a lente temporal virar comparável entre idiomas, trocar o tempo de relógio por um
eixo linguisticamente comum:
- **Alinhamento fonético** (forced alignment): agregar relevância por tipo de segmento
  (vogal, plosiva, fricativa, silêncio, onset/offset).
- **Eixo condicionado por acústica** (sem alinhamento): relevância como função de uma
  propriedade local (energia, f0, sonoridade voiced/unvoiced).
- Time-warping para duração canônica: mais fraco, ainda mistura conteúdo, não recomendado.

**A definir.** Quais dados multilíngues usar (candidato: MLAAD). Ordem sugerida na
discussão: A como base, B como passo seguinte.
