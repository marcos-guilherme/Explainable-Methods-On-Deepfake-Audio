# Explicação ouvível da decisão do detector — guia completo

Documento para entender, do começo ao fim, como transformamos a explicação do detector
de deepfake em **áudio que dá para ouvir** e, depois, em **evidência que dá para medir**.
A ideia é ser acessível para quem nunca mexeu com processamento de sinais, mas sem abrir mão
do rigor técnico. Cada seção começa com a intuição e termina com o detalhe formal.

> Convenção de "registro" (a mesma do resto do projeto):
> - **Ilustrativo**: mostra/faz ouvir um padrão, sem afirmar significância nem causa.
> - **Fidelidade por intervenção**: modifica o áudio e mede a resposta do detector, mas não
>   demonstra causalidade no mundo real porque a própria filtragem pode introduzir artefatos.
>
> A parte ouvível é ilustrativa. O re-scoring fornece evidência de fidelidade do modelo sob
> uma intervenção controlada, não uma prova causal forte.

---

## 0. A pergunta e a ideia central

O detector recebe um áudio e responde "spoof" (sintético) ou "bonafide" (real). Queremos
saber **em que parte do som** ele se apoiou para decidir. Duas perguntas guiam tudo:

1. **Onde, no som, está a evidência?** Em quais frequências e em quais instantes.
2. **Essa evidência é de fato usada?** Se apagarmos justamente ela, a decisão desmorona?
   E se apagarmos uma parte qualquer, do mesmo tamanho, o estrago é menor?

A ideia central é simples de enunciar: **pegar o áudio original, deixar tocando só as regiões
que o modelo usou e silenciar o resto**. Isso produz duas coisas. Uma versão ouvível (você
literalmente escuta o que empurra para "spoof" e o que empurra para "bonafide") e, quando
devolvemos esse áudio filtrado ao próprio detector, um número que mede se aquela região
sustentava mesmo a decisão.

Analogia: é como um realçador de texto. Em vez de pintar as palavras que importam num
documento, "pintamos" as frequências e instantes que importam no áudio. E depois testamos:
se apagar só o que foi pintado, a pessoa ainda entende o texto? Se não entende, o realce
estava certo.

---

## 1. De onde vem a "evidência": a relevância tempo-frequência

Antes de filtrar, precisamos saber **quanto cada instante e cada frequência pesaram** na
decisão. Isso vem de um método de explicação chamado **DFT-LRP** (mais precisamente, a sua
versão janelada, a **STDFT-LRP**).

**Intuição.** O detector é uma pilha de operações matemáticas. A LRP ("Layer-wise Relevance
Propagation") caminha da resposta final de volta para a entrada, distribuindo a "culpa" pela
decisão entre as amostras do sinal. O resultado é um número por instante do áudio: quanto
aquele pedacinho de som empurrou a decisão. A DFT-LRP dá o passo final e leva essa culpa do
**tempo** para a **frequência**, aproveitando que a transformada de Fourier é linear e
invertível.

**Formal.** Para cada janela do sinal obtemos um mapa \(R(t, f)\): a relevância assinada em
cada instante \(t\) e cada frequência \(f\). O **sinal** dessa relevância carrega a direção:

- \(R(t,f) > 0\): aquela região empurra a decisão para **spoof**.
- \(R(t,f) < 0\): aquela região empurra para **bonafide**.

Propriedade importante (conservação): a soma de toda a relevância no plano tempo-frequência
é igual à relevância total no tempo. Nada é criado nem destruído na travessia para a
frequência. Isso é o que chamamos de LRP "conservativo" e é verificado por um certificado
numérico (o erro de conservação fica na casa de \(10^{-12}\)).

- **Código:** `scripts/brspeech_xai/dft_lrp.py` (`stdft_lrp`, `time_to_freq_relevance`).

---

## 2. Fatiar o som: a STFT (transformada de Fourier de curto prazo)

Para "pintar" regiões de tempo **e** frequência ao mesmo tempo, precisamos de uma
representação que tenha os dois eixos. É a STFT.

**Intuição.** Um som inteiro é longo demais para analisar de uma vez, e ele muda ao longo do
tempo (uma vogal, depois um silêncio, depois um "s"). Então cortamos o áudio em pedacinhos
curtos que se sobrepõem (janelas), e para cada pedacinho perguntamos "quais frequências estão
presentes aqui?". O resultado é o espectrograma: um mapa tempo × frequência.

**Formal.** Os ingredientes, com os valores que usamos (áudio a 16 kHz):

- **Janela (`win = 512` amostras ≈ 32 ms).** O tamanho do pedacinho. Define a resolução em
  frequência: \(16000 / 512 \approx 31{,}25\) Hz por "faixa" (bin).
- **Salto (`hop = 128` amostras).** Quanto a janela anda a cada passo. Com 512/128, as janelas
  se sobrepõem em 75%, o que dá suavidade e permite reconstruir o sinal depois.
- **Janela de Hann.** Cada pedacinho é multiplicado por um "sino" que zera nas bordas. Sem
  isso, os cortes abruptos criariam frequências falsas (vazamento espectral).
- **Acolchoamento (`pad = win/2`).** Zeros no começo e no fim, para que toda amostra original
  caia no centro de alguma janela (a Hann apagaria as bordas do áudio, senão).
- **WOLA (soma dos pesos das janelas, `wsum`).** Como as janelas se sobrepõem, cada amostra é
  coberta por várias delas. `wsum` guarda quanto peso total cada amostra recebeu. Vamos usar
  isso para desfazer o janelamento na volta.

- **Código:** análise idêntica em `dft_lrp.stdft_lrp` e em `sonify_lrp._softmask_reconstruct`.

---

## 3. A máscara suave (soft-mask): da relevância para um "controle de volume"

Agora transformamos a relevância \(R(t,f)\) num **controle de volume por região**, um número
entre 0 e 1 para cada instante e frequência. É o "soft-mask".

**Intuição.** Pense num equalizador que, em vez de ter alguns botões, tem um botão para cada
frequência e cada instante. A máscara ajusta cada botão: 1 = deixa passar inteiro (região
muito relevante), 0 = silencia (região irrelevante), e valores intermediários atenuam de
forma proporcional. "Suave" porque não é liga/desliga; é um degradê.

**Formal.** Três passos:

1. **Separar por direção.** Para o áudio "rumo a spoof", ficamos só com a parte positiva da
   relevância, \(R^+ = \max(R, 0)\). Para o áudio "rumo a bonafide", só com a parte negativa,
   \(R^- = \max(-R, 0)\). Assim geramos dois áudios: um que isola o que puxa para sintético e
   outro que isola o que puxa para real.
2. **Normalizar por percentil (`pctl = 99`).** Dividimos pela relevância do percentil 99
   daquele clipe. Isso coloca as regiões mais relevantes perto de 1 sem deixar um único pico
   dominar a escala. Percentil mais alto deixa a máscara mais seletiva (pinta menos, só o topo).
3. **Cortar em [0, 1]** e aplicar um **piso opcional (`floor`)**. O corte satura o que passa
   de 1. O `floor` (padrão 0) é quanta energia de fundo manter fora das regiões relevantes:
   com 0, o irrelevante é silenciado por completo; com 0,2, sobra 20% de fundo.

O resultado é uma matriz \(M(t,f) \in [0,1]\), do mesmo tamanho do mapa de relevância.

- **Código:** `sonify_lrp._softmask_reconstruct` (bloco da máscara).

---

## 4. Aplicar a máscara: mexer no volume, preservar o timbre

Com o controle de volume pronto, aplicamos janela a janela. Este é o coração da sonificação.

**Intuição.** Em cada pedacinho, olhamos quais frequências estão presentes, abaixamos o
volume das que a máscara mandou abaixar, e reconstruímos o pedacinho. Um detalhe crucial:
mexemos só no **volume** de cada frequência, nunca no seu "timbre" (a fase). É isso que faz o
resultado soar como um áudio natural filtrado, e não como um chiado robótico.

**Formal.**

- Levamos a janela para a frequência com a FFT, obtendo um espectro **complexo** (que carrega
  magnitude **e** fase).
- Multiplicamos esse espectro pela máscara, que é **real** e está em [0, 1]. Multiplicar um
  número complexo por um real positivo escala a magnitude e **mantém a fase intacta**.
- Voltamos ao tempo com a FFT inversa e somamos com as janelas vizinhas (overlap-add).

\[
\hat{x}_{\text{janela}} = \text{IFFT}\big(\, \text{FFT}(x_{\text{janela}} \cdot w)\ \cdot\ M \,\big)
\]

- **Código:** `sonify_lrp._softmask_reconstruct` (laço de `rfft`/`irfft`).

---

## 5. Desfazer o janelamento: por que a reconstrução é exata

Depois de somar todas as janelas, cada amostra ficou multiplicada pela soma dos pesos da Hann
que a cobriram (`wsum`). Dividir por `wsum` desfaz exatamente esse efeito. Removemos o
acolchoamento e temos o áudio final.

**Por que isso importa (a garantia de honestidade do método).** Se a máscara for **toda 1**
(não apagamos nada), a ida-e-volta pela STFT devolve o **sinal original**, amostra por amostra,
a menos de erro de ponto flutuante (casa de \(10^{-12}\)). Consequência: **qualquer diferença
que você ouça vem só do que a máscara realmente atenuou**, não de artefato da transformada.
Essa é a base do teste de sanidade que vamos automatizar.

\[
M \equiv 1 \;\Rightarrow\; \hat{x} = x
\]

- **Código:** `sonify_lrp._softmask_reconstruct` (`y = y / wsum`), teste de sanidade planejado
  em `scripts/tests/`.

---

## 6. Da ilustração à medição: a validação de fidelidade

Ouvir é convincente, mas não basta como evidência. Um som filtrado pode parecer "certo" e ainda
assim
não corresponder ao que o modelo usa. Por isso o passo decisivo: **devolver o áudio filtrado
ao próprio detector e medir a decisão**. Se a região apontada é mesmo a evidência, mexer nela
tem que mexer na resposta, mais do que mexer numa região qualquer.

**Intuição.** Dois experimentos complementares, como cobrir partes de uma foto para ver o que
é essencial para reconhecê-la:

- **Manter só o essencial (keep / suficiência):** deixe tocando só as frequências mais
  relevantes. Se a decisão se mantém, essa parte **basta**.
- **Remover o essencial (delete / comprehensiveness):** apague justamente as frequências mais
  relevantes. Se a decisão desmorona, essa parte era **necessária**.
- **Controle inferior (bottom):** repita com as faixas de menor relevância DFT-LRP. Se o
  ranking é informativo, as top-k devem ser mais suficientes/necessárias que as bottom-k.
- **Controle aleatório pareado (random):** faça o mesmo com a **mesma quantidade** de faixas,
  escolhidas ao acaso no mesmo clipe e no mesmo \(k\). Isso estima o efeito genérico de filtrar.
- **Controle de energia RMS:** depois da filtragem, cada versão top, bottom e random é
  reescalada para ter o mesmo RMS do áudio original. Assim, uma queda de confiança não pode ser
  atribuída apenas à redução global de energia. O reescalonamento não recorta amplitudes
  (`clipping`); `--no-rms-match` desliga o controle para análise de sensibilidade. Na exportação
  ouvível, a conversão separada para PCM normaliza o pico, mas o detector recebe a onda sem esse
  recorte.

**Formal.**

- **Unidade:** as 24 faixas mel (as mesmas de H1/H2/DFT-LRP), para tudo falar a mesma língua.
- **Ranqueamento:** a relevância DFT-LRP do próprio clipe, orientada à classe prevista (se
  previu spoof, contam as faixas positivas; se bonafide, as negativas).
- **Varredura de \(k\):** repetimos com \(k \in \{1,2,3,4,6,8,12\}\) faixas mantidas/removidas
  e resumimos a curva com uma média estilo AOPC, mais um ponto principal (top-6).
- **O que reportamos:** curvas e AOPC para top, bottom e random. Para comprehensiveness,
  reportamos também diferenças pareadas por clipe e agregadas: top−random, top−bottom e
  random−bottom, sempre usando os mesmos clipes e valores de \(k\).
- **Incerteza:** intervalo de confiança de 95% por bootstrap **sobre os clipes** (a unidade de
  variação é o clipe, não a banda), além do teste de Wilcoxon quando há pares suficientes e
  diferenças não degeneradas. Casos insuficientes ou todos-zero são marcados explicitamente.

A leitura esperada da figura: a linha **keep** fica alta (a evidência basta), a linha
**delete** cai forte (a evidência é necessária), enquanto **bottom** e **random** fornecem duas
referências sob a mesma intervenção. Essa comparação reduz interpretações triviais por energia
ou escolha de bandas, mas não elimina o confound introduzido pela filtragem STFT.

- **Código implementado:** `scripts/dev/faithfulness_bands.py`, acionado pelo subcomando
  `vm.sh faithfulness`.

---

## 7. Relação com o paper L2I (o que pegamos emprestado e o que não)

Nossa parte ouvível se inspira no **"Listen to Interpret" (L2I, Parekh et al., 2024)**, mas
não é uma cópia. Vale deixar claro para não prometer o que não fazemos.

**O que o L2I faz.** Aprende um **dicionário NMF** de padrões espectrais e treina uma **rede
interpretadora extra** sobre os mapas convolucionais de um classificador CNN. A explicação
ouvível é a remontagem dos componentes NMF mais relevantes.

**O que transferimos (o espírito).**

- A parte ouvível: **soft-mask + STFT inversa**, guiada pela nossa relevância DFT-LRP no lugar
  do dicionário NMF.
- A parte de rigor: **fidelidade por re-scoring** (remover a evidência e medir a queda), que é
  exatamente a forma como o L2I valida.

**O que não replicamos (e por quê).**

- Não há dicionário aprendido: nossa base é **frequência** (24 faixas), então **filtramos o
  áudio**, não remontamos objetos aprendidos.
- Não treinamos rede interpretadora extra nem as suas perdas.
- Arquitetura diferente: o L2I usa CNN com eixos tempo-frequência nas camadas; nossos encoders
  são **transformers** (wav2vec2/hubert/wavlm), sem essa grade 2D.
- Natureza da tarefa: o L2I brilha com **objetos sonoros** separáveis (um latido, um
  instrumento); a pista de deepfake é um **artefato de síntese** espalhado pela fala, sem
  objeto para isolar.

Conclusão honesta: fazemos uma **adaptação fiel do espírito** do L2I, apropriada à nossa
tarefa e arquitetura, não uma reimplementação.

---

## 8. Ressalvas (para não escorregar na linguagem)

- **Filtrar introduz artefato.** É o mesmo confound da oclusão de H2. Bottom, random e
  RMS-match tornam a comparação mais justa, mas não removem esse confound.
- **Sem causalidade forte.** A sonificação ilustra e a métrica quantifica fidelidade sob uma
  intervenção STFT específica. Ela não estabelece que as bandas tenham efeito causal em áudio
  real não filtrado, nem que manipular o processo gerador nessas bandas produziria o mesmo efeito.
- **Escopo inicial.** Começamos por um encoder (wav2vec2) em todo o split de teste, para
  validar o método; a extensão a hubert e wavlm reusa o mesmo código.

---

## 9. Glossário rápido

- **STFT / STDFT:** transformada de Fourier de curto prazo. Fatia o som em janelas e mostra
  as frequências de cada fatia (mapa tempo × frequência).
- **Bin de frequência:** cada "faixa fina" do espectro de uma janela. Aqui, ~31 Hz de largura.
- **Faixa mel (banda):** agrupamento das frequências em 24 faixas na escala mel (mais fina no
  grave, mais larga no agudo), a unidade que usamos em H1/H2/DFT-LRP.
- **Relevância (LRP):** quanto cada parte da entrada pesou na decisão; tem sinal (para spoof ou
  para bonafide).
- **DFT-LRP / STDFT-LRP:** levam essa relevância do tempo para a frequência (e para o plano
  tempo-frequência), de forma conservativa.
- **Soft-mask:** controle de volume por região (0 a 1) derivado da relevância.
- **Fase:** o "timbre" de cada frequência; preservá-la é o que mantém o áudio natural.
- **WOLA:** técnica de soma ponderada das janelas que torna a reconstrução exata.
- **Suficiência (keep):** manter só o relevante basta para a decisão?
- **Comprehensiveness (delete):** remover o relevante derruba a decisão?
- **Bottom-k:** as \(k\) faixas de menor relevância, usadas como controle do ranking.
- **RMS-match:** reescala cada onda perturbada para a energia RMS da original, sem clipping.
- **AOPC:** média da curva ao longo de vários \(k\); resume o experimento num número.
- **Bootstrap:** reamostrar os clipes para estimar a incerteza (intervalo de confiança).
