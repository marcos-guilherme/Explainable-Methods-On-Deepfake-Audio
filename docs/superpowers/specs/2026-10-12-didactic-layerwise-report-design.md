# Relatório layer-wise didático — desenho editorial

## Objetivo

Reorganizar o relatório trilíngue de XAI para que um leitor técnico não
especialista em XAI compreenda primeiro os resultados principais, sem perder a
rastreabilidade das 27 células, 12 camadas, três encoders e três idiomas.

O corpo principal deve ocupar aproximadamente 16–20 páginas. Todos os resultados
completos permanecem no mesmo PDF em apêndices e nos CSVs do bundle.

## Problema observado

O relatório atual converte matrizes em prosa repetitiva. Desempenho, emergência,
relevância, reorganização e exemplos STDFT descrevem separadamente até 27 células.
As legendas repetem modelos e pares de idiomas, e tabelas extensas interrompem a
narrativa. O resultado preserva informação, mas dificulta identificar o efeito
principal e distinguir evidência, interpretação e limitação.

## Princípios

1. Organizar Resultados do padrão global para os mecanismos de XAI e a fidelidade.
2. Cada figura responde uma pergunta principal.
3. Mostrar estimativas, dispersão observável e incerteza; não usar barplots.
4. Não representar como distribuição o que existe apenas como estimativa e IC.
5. Separar resultado, interpretação e principal limite.
6. Manter no apêndice toda matriz ou tabela retirada do corpo.
7. Não atribuir efeito causal ao idioma: idioma, corpus e áudio mudam em conjunto.
8. Não recalcular encoder ou explicação e não exigir GPU para gerar o relatório.

## Evidência quantitativa que orienta a narrativa

- Existem 27 células modelo × idioma de treino × idioma de avaliação.
- Para `delta_roc_auc`, 201 de 216 estimativas são negativas e 196 têm IC
  inteiramente abaixo de zero; a mediana é -0,373.
- Para `delta_mcc`, 216 de 216 estimativas são negativas e 215 têm IC inteiramente
  abaixo de zero; a mediana é -0,593.
- A fração de decisões iguais entre encoders tem 324 estimativas, mediana 0,861 e
  amplitude 0,571–0,989. O texto deve dizer “acordo alto na mediana, mas
  heterogêneo”, sem impor categoria normativa.
- Na fidelidade da camada 12 diagonal, os ICs ficam acima de zero em 50 de 84
  comparações do Wav2Vec2 Base, 10 de 84 do HuBERT Base e 11 de 84 do WavLM Base.
  Isso significa mais sinais positivos neste protocolo, não superioridade causal.

## Estrutura do corpo principal

### 1. Orientação

- Pergunta e contribuição do estudo.
- Desenho 3 encoders × 3 idiomas × 3 idiomas de avaliação.
- Convenções visuais e glossário curto: diagonal, transferência, ROC-AUC, MCC,
  intervalo bootstrap e relevância.
- Inventário reduzido a uma tabela compacta.

### 2. Resultados

A ordem é fixa:

1. padrão global de desempenho;
2. magnitude da queda em transferência;
3. acordo entre encoders;
4. mudança de relevância entre corpora/idiomas;
5. divergência espectral;
6. fidelidade por intervenção.

Cada bloco contém:

- pergunta em uma frase;
- figura;
- até três achados quantitativos;
- interpretação;
- principal cautela;
- referência ao apêndice e CSV correspondentes.

### 3. Síntese

Uma tabela curta fecha Resultados com quatro colunas:

| Pergunta | Evidência | Resultado | Cautela |
|---|---|---|---|

### 4. Limitações e conclusão

As limitações deixam de ser uma lista genérica no fim e passam também a aparecer
junto de cada resultado. A seção final mantém apenas as limitações transversais e
uma conclusão curta.

## Figuras principais

### Figura 1 — visão geral matricial

- Heatmaps 3×3 por modelo.
- Treino no eixo vertical; avaliação no horizontal.
- ROC-AUC e MCC em painéis ou linhas distintas, com escalas próprias.
- Mesma ordem `eng`, `por`, `zho` em todos os painéis.
- A célula mostra valor e camada selecionada exclusivamente pela maior ROC-AUC;
  o MCC vem da mesma camada.

### Figura 2 — queda de ROC-AUC

- Dot/forest estimation plot de `delta_roc_auc`.
- Ponto, IC bootstrap de 95% e linha de referência zero.
- Facetas por modelo; codificação consistente para origem e alvo.
- A legenda define o delta como fora da diagonal menos diagonal no mesmo alvo.

Não usar Gardner–Altman com densidade enquanto os 2.000 draws não forem
persistidos. Estimativa e limites não permitem reconstruir honestamente a
distribuição bootstrap.

### Figura 3 — queda de MCC

Mesmo contrato visual da Figura 2 para `delta_mcc`, em figura separada para não
misturar escalas nem interpretações.

### Figura 4 — acordo entre encoders

- Heatmaps com ordem fixa dos três pares de modelos e nove células.
- Um painel por métrica principal, sempre com escala própria.
- Corpo principal: resumo por mediana entre camadas.
- Apêndice: perfil completo das 12 camadas e métricas adicionais.
- Não combinar fração de acordo, kappa, Spearman e distâncias XAI na mesma escala.

### Figura 5 — mudança de relevância

- Dot-interval plot por modelo e par de idiomas-alvo.
- Ponto e IC bootstrap por camada/célula.
- Separar grupos `all`, `real` e `synthetic` por faceta ou codificação discreta.
- Não usar violin: as camadas não são réplicas intercambiáveis e os draws
  bootstrap não estão persistidos.

### Figura 6 — divergência espectral

- Distância de Wasserstein por camada com IC bootstrap.
- Small multiples por modelo e par de idiomas.
- Distância maior significa maior deslocamento da massa de relevância no eixo de
  frequência; não significa melhor ou pior modelo.
- Perfis médios de relevância por frequência, se exibidos, ficam em painel
  complementar ou apêndice e não recebem o nome de Wasserstein.

### Figura 7 — fidelidade por intervenção

- Dot-interval plot de `top_minus_random` e `top_minus_bottom`.
- Facetas por modelo e classe; valores de `k` explicitados.
- Linha zero e IC bootstrap de 95%.
- Escopo visível no título ou subtítulo: camada 12, células diagonais, oito
  amostras por classe.

### Exemplos STDFT

Exemplos locais aparecem no fim ou no apêndice, próximos do resultado que
ilustram. Não conduzem a narrativa nem sustentam generalização populacional.

## Apêndices do mesmo PDF

1. protocolo, inventário, hashes e proveniência;
2. desempenho completo por modelo, célula e camada;
3. emergência, consolidação e reorganização completas;
4. relevância espectral e associação XAI↔desempenho;
5. tabelas completas das quatro comparações bootstrap;
6. fidelidade detalhada por classe, `k` e controle;
7. exemplos STDFT e diagnósticos de conservação.

Os CSVs continuam sendo a fonte linha a linha. O PDF fornece uma representação
completa, porém compacta, das mesmas informações.

## Mudanças no gerador

- Separar construção de narrativa principal e apêndices.
- Substituir loops de parágrafo por célula por agregações e tabelas compactas.
- Criar builders independentes para as sete figuras principais.
- Centralizar ordem, cores, rótulos de modelos/idiomas e escalas.
- Encurtar legendas: pergunta, codificação visual e limite; inventários completos
  ficam fora das legendas.
- Registrar todas as figuras, tabelas e CSVs no manifesto como hoje.
- Preservar geração determinística, publicação atômica e validação dos artefatos.

## Referências

As referências sugeridas por ferramentas de busca servem apenas como candidatas.
Antes de entrarem na bibliografia, DOI, ano, volume, páginas, título e pertinência
devem ser verificados na fonte primária. O texto e o aviso de uso do Consensus não
serão redistribuídos.

## Critérios de aceitação

1. Corpo principal entre 16 e 20 páginas no conjunto trilíngue completo
   (decisão do usuário: corpo conciso, sem conteúdo adicionado só para ganhar páginas).
2. As sete figuras principais existem e obedecem aos contratos acima.
3. Nenhuma célula, camada ou métrica atualmente publicada desaparece do PDF,
   apêndices ou CSVs.
4. Não há parágrafos repetidos para cada uma das 27 células no corpo.
5. Legendas não repetem o inventário completo de modelos e idiomas.
6. Todas as afirmações quantitativas são derivadas programaticamente das tabelas.
7. O texto distingue descrição, incerteza bootstrap e inferência causal.
8. Bundle determinístico, hashes válidos e compilação `pdflatex` em duas passagens.
9. Testes cobrem completude do apêndice, ordem das figuras, escalas separadas,
   cautelas obrigatórias e ausência de regressões no manifesto.

## Fora de escopo

- adicionar novos métodos XAI;
- reexecutar encoders, probes ou explicações;
- persistir todos os draws bootstrap;
- alegar efeito causal do idioma;
- produzir uma figura de “model improvement” sem experimento adicional que a
  sustente.
