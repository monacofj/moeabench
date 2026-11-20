<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Ground Truth no MoeaBench: fundamentos matemáticos, geração e qualidade das referências

Este documento explica como o **Ground Truth (GT)** deve ser compreendido, construído, utilizado e validado no MoeaBench. Destina-se a leitores que não conhecem o histórico do projeto. Apresenta conceitos desde a definição de otimização multiobjetivo, desenvolve os resultados matemáticos e numéricos dos casos DPF3 e DPF5 e esclarece suas consequências para avaliação experimental, amostragem, calibração e referências históricas.

Os resultados científicos dos DPF3 e DPF5 foram estabelecidos na investigação [#119](https://github.com/monacofj/moeabench/issues/119) e documentados por provas independentes, testes e experimentos na [PR #139](https://github.com/monacofj/moeabench/pull/139). As demonstrações dizem respeito às **equações matemáticas publicadas**; uma demonstração sobre o modelo não significa que toda versão do avaliador de software implemente aquelas equações corretamente. Sempre que este documento descreve defeitos de uma implementação concreta, indica o contexto histórico da auditoria, sem convertê-los em características matemáticas permanentes dos problemas.

## 1. Por que Ground Truth é importante?

O MoeaBench organiza experimentos com **problemas de otimização multiobjetivo (MOPs)** e **algoritmos de otimização multiobjetivo (MOEAs)**. Enquanto um problema com objetivo único procura uma solução que minimize determinada função, um problema multiobjetivo procura compromissos entre várias funções, que em geral não podem ser minimizadas simultaneamente.

Suponhamos um projeto em que se deseja minimizar custo, consumo e tempo. Uma solução barata pode ser demorada; uma solução rápida pode consumir muita energia. Há muitas soluções possíveis, mas somente algumas são **Pareto-ótimas**: não existe outra solução viável melhor em ao menos um objetivo sem piorar algum dos demais.

Ao avaliar um MOEA, perguntar apenas quantas soluções ele produziu não é suficiente. Também precisamos saber se as soluções se aproximam das ótimas, se cobrem todos os compromissos relevantes, se representam os extremos e se a comparação entre dois algoritmos usa a mesma referência. Um algoritmo pode gerar uma nuvem visualmente regular e, ainda assim, representar somente uma pequena região da verdadeira frente.

Chamamos **Ground Truth (GT)** à referência científica utilizada como padrão para essa comparação. Em problemas com solução matemática conhecida, o GT deve ser fundamentado nessa solução, e não simplesmente considerado verdadeiro porque um algoritmo executou por muitas gerações.

É importante distinguir:

- **Verdade matemática:** os conjuntos PS e PF, definidos pela função do problema e por todas as decisões viáveis.
- **Representação computacional:** uma amostra finita de pontos que procura cobrir a PF com qualidade documentada.
- **Artefato de referência:** dados persistidos e versionados, usados em uma experiência, indicador ou certificação, cuja origem deve ser preservada.

Uma amostra pequena pode conter somente pontos rigorosamente ótimos e ser ruim para medir cobertura. Uma amostra densa pode estar sobre a geometria errada. Uma referência historicamente reproduzível pode conter valores posteriormente reconhecidos como matematicamente incorretos. Cada situação exige interpretação diferente.

## 2. Definições matemáticas e terminologia

### 2.1. Função multiobjetivo e dominância

Sejam \(N\) variáveis de decisão e \(M\) objetivos. Uma função de avaliação é

\[
F:X\subseteq\mathbb R^N\longrightarrow\mathbb R^M,\qquad
F(x)=(f_1(x),f_2(x),\ldots,f_M(x)),
\]

onde \(X\) representa as decisões viáveis. Nos DPFs abordados neste documento todos os objetivos são minimizados.

Uma decisão \(a\) **domina** \(b\) quando

\[
f_i(a)\le f_i(b)\quad\text{para todo }i,
\qquad
f_j(a)<f_j(b)\quad\text{para ao menos um }j.
\]

A expressão “não dominado” sempre deve indicar o **universo da comparação**. Se um conjunto de cem pontos não contém pares de dominância, esses pontos são não dominados **entre si**. Ainda pode existir outro ponto viável, fora da amostra, que domine algum deles. Logo, filtrar uma amostra por não dominância não constitui prova de otimalidade global.

### 2.2. Pareto set e Pareto front

O **Pareto set (PS)** é o conjunto das decisões viáveis não dominadas por nenhuma outra decisão viável. A **Pareto front (PF)** é sua imagem no espaço dos objetivos:

\[
PS=\{x\in X:\nexists y\in X\text{ que domine }x\},
\qquad PF=F(PS).
\]

A PS pertence ao espaço de decisão, com \(N\) coordenadas; a PF pertence ao espaço dos objetivos, com \(M\) coordenadas. Duas decisões diferentes podem ter a **mesma** imagem em \(F\), especialmente quando há redundância ou degeneração.

Na família DPF, o parâmetro \(D\) indica a dimensão **essencial** da formulação; \(M-D\) objetivos adicionais podem ser redundantes ou obtidos por projeções. O parâmetro \(K\) aparece na configuração do número de variáveis. \(D\) não significa número total de variáveis e não deve ser confundido com \(N\).

### 2.3. Dimensão geométrica, degeneração e componentes

Uma PF com \(M\) objetivos não precisa ter dimensão geométrica \(M-1\). Uma frente **degenerada** pode estar inteiramente contida em uma variedade de dimensão inferior. Uma frente **mista** pode conter componentes de dimensões diferentes, como uma superfície e uma curva.

Isso tem consequência prática. Distribuir uma mesma quantidade de pontos em uma curva e em uma superfície não produz automaticamente a mesma cobertura. Tampouco uma distribuição uniforme de valores no espaço de decisão implica uma distribuição uniforme de **objetivos**: a transformação pode distorcer distâncias, colapsar coordenadas e produzir duplicatas.

### 2.4. Amostragem, cardinalidade e cobertura

Uma referência finita pode ser examinada segundo diferentes quantidades:

1. **Pontos solicitados:** o argumento nominal de amostragem, frequentemente chamado \(n_{\mathrm{points}}\).
2. **Decisões produzidas:** número de linhas efetivamente devolvidas por uma construção de PS.
3. **Objetivos produzidos:** número de linhas da PF avaliada, eventualmente reduzido por filtragem.
4. **Objetivos distintos:** número de vetores efetivamente distintos em espaço objetivo, considerando a precisão e o critério de igualdade usados.
5. **Pontos matematicamente distintos:** conceito mais forte, que não pode ser inferido apenas de uma comparação exata entre números de ponto flutuante.

Uma forma de quantificar a cobertura de uma amostra \(R\) sobre a PF verdadeira \(P\) é a distância dirigida

\[
h(P,R)=\sup_{p\in P}\min_{r\in R}\|p-r\|_2.
\]

A **distância de Hausdorff** é \(d_H(P,R)=\max\{h(P,R),h(R,P)\}\). Se todos os pontos de \(R\) pertencem exatamente à PF, então \(h(R,P)=0\). Uma prova de limite superior de \(h(P,R)\) é muito mais forte que medir distâncias sobre alguns pontos de sondagem. O máximo observado numa sondagem finita **não** é um limite certificado sobre a frente contínua.

É igualmente necessário verificar **extremos**, interfaces, regiões desconectadas, densidade e espaçamento, e não apenas medianas ou valores médios de distância.

### 2.5. Fidelidade matemática, numérica e estatística

Um GT adequado requer verificações independentes:

- **Equações:** as funções implementadas coincidem com a definição publicada em todos os ramos e dimensões suportados.
- **Pertinência global:** os pontos da referência são realmente Pareto-ótimos, não apenas não dominados dentro da amostra.
- **Completude matemática:** a parametrização alcança todas as partes da PF.
- **Cobertura finita:** a quantidade e distribuição dos pontos representam suficientemente essa frente contínua.
- **Representabilidade:** os valores podem ser calculados com a precisão numérica escolhida, incluindo transformações e arredondamentos.
- **Custo:** o método tem complexidade e uso de memória compatíveis com o objetivo da calibração.
- **Reprodutibilidade e proveniência:** sementes, parâmetros, versão matemática, código, ambiente e referência utilizada são identificáveis.

Essas verificações não são intercambiáveis. Uma identidade esférica é condição matemática necessária de determinado problema, mas não prova global de Pareto. Por outro lado, uma demonstração da PF contínua não comprova a qualidade de sua discretização finita.

## 3. Responsabilidade por MOP e arquitetura da API

### 3.1. Cada problema determina sua estratégia de GT

**O gerador cientificamente adequado deve ser específico do MOP e de sua configuração.** O problema conhece sua função de avaliação, restrições, dimensão intrínseca, parametrização, extremos e eventuais componentes. Não se deve impor um amostrador geral a todos os MOPs apenas para evitar esforço de implementação.

Isso **não implica proibir reutilização de infraestrutura**. Avaliação vetorizada, álgebra numérica, geração determinística, verificação de erro, armazenamento, controle de sementes e funções geométricas matematicamente equivalentes podem ser compartilhados. O critério de reutilização é **equivalência científica comprovada**, não redução artificial de código. Um amostrador hiperesférico pode ser compartilhado por problemas cuja PF realmente tenha aquela geometria; não pode, sem validação, substituir uma construção deformada por potência 100 ou uma frente formada por duas componentes de dimensões distintas.

Essa responsabilidade é coerente com a [ADR 0006](adr/0006-benchmarks-reimplementation.md), que atribui PS/PF analíticos ao MOP, e com a direção aceita em princípio na [ADR 0057](adr/0057-proposed-purpose-specific-gt-budgets.md).

### 3.2. Funções públicas e suas diferenças

**evaluation(X)** avalia a função objetivo para uma matriz de decisões. Um erro na avaliação muda o **problema otimizado** e potencialmente os resultados dos próprios MOEAs.

**ps(n_points)** gera decisões que deveriam pertencer à PS. O tamanho nominal do pedido não significa necessariamente que o retorno tenha a mesma quantidade de vetores de objetivos distintos.

**pf(n_points)** fornece uma referência no espaço objetivo. Em **BaseMop**, a implementação padrão avalia o resultado de **ps()**. Consequentemente, herda erros de PS ou do avaliador. Um MOP pode especializar **pf()**; o DPF3 analisado empregava filtragem por não dominância, reduzindo a quantidade efetiva de pontos e impedindo assumir correspondência posicional entre a matriz original de PS e a matriz retornada de PF.

**calibrate()** é a operação **explícita** de obtenção e persistência do GT. É independente das antigas métricas FAIR; calibração de GT existia como conceito anterior àquela pesquisa e deve permanecer parte da API do MoeaBench.

**calibrated_gt** disponibiliza a referência calibrada conservada para a instância, com retorno de cópia defensiva.

Na implementação examinada, **moeabench/mops/calibration.py** aceita carregar sidecar existente, receber **source_gt** externo, usar **source_search** empírico ou gerar a referência com **pf(n_points)**. Os dados persistidos incluem **gt_reference** e metadados básicos no esquema JSON **mop_gt_v1**. O uso explícito de **force** controla substituição de artefatos.

### 3.3. Finalidade: visualização, comparação e calibração

Uma PF leve para visualização não precisa ter a mesma densidade de uma referência científica para indicadores. A exibição pode utilizar redução de pontos desde que preserve as características essenciais, sem transformar essa redução no GT de análise. Chamadas comuns de **ps()** ou **pf()** não devem provocar criação implícita de arquivos ou caches persistentes.

Os padrões históricos de calibração usam **2.000 pontos para DTLZ** e **10.000 para DPF** quando não há orçamento explícito. Esses valores são decisões operacionais históricas (ADR 0018), não teoremas de suficiência. A [ADR 0057](adr/0057-proposed-purpose-specific-gt-budgets.md) aceita **em princípio** a seleção automática de orçamento por geometria, precisão e custo, mas não define uma regra numérica universal. A qualidade do GT e o custo da sua obtenção devem ser tratados pelo MOP concreto.

A possibilidade de conservar o **GT efetivamente usado** em uma exportação de experimento pertence ao contrato de persistência do experimento: exportar dados não autoriza gerar uma nova referência silenciosamente.

## 4. DPF3: definição matemática e papel dos limiares caóticos

### 4.1. Objetivos essenciais

Na formulação publicada do DPF3, os primeiros \(D-1\) parâmetros de posição definem ângulos

\[
\theta_j=\frac{\pi}{2}x_j^{100},\qquad j=1,\ldots,D-1,
\]

e a função de distância é

\[
g(x)=\sum_{i=D}^{N}(x_i-0{,}5)^2\geq0.
\]

Se \(H_D(\theta)\) representa o vetor não negativo da hiperesfera unitária com \(D\) coordenadas, \(\|H_D(\theta)\|_2=1\), então os \(D\) objetivos essenciais são

\[
\gamma(x)=(1+g(x))(\mathbf1-H_D(\theta(x))).
\]

Os primeiros \(D-1\) objetivos publicados são as primeiras coordenadas de \(\gamma\). A última coordenada essencial é representada por uma série de projeções que criam objetivos redundantes.

A potência **100** é fundamental. Uma grade uniforme em \(x_j\) não é uniforme no ângulo \(\theta_j\), e pode deixar regiões da PF muito pouco representadas.

### 4.2. O que significa “parâmetros caóticos” \(\eta\)?

O DPF3 utiliza \(U=M-D\) valores \(\eta_1,\ldots,\eta_U\) que funcionam como **limiares de projeção**. A publicação prescreve que os valores sejam gerados a partir de uma sequência determinística de um **mapa logístico**, da forma

\[
z_{n+1}=3{,}8z_n(1-z_n).
\]

O termo “caótico” descreve o comportamento não linear e sensível desse sistema dinâmico. Não significa que os parâmetros devam ser sorteados a cada avaliação do MOP ou a cada execução do MOEA. Uma vez fixados, eles são parte da definição do problema.

A definição publicada exige que sejam utilizados **em ordem crescente**. Por exemplo, se a sequência gerar \(0{,}34;0{,}85;0{,}47\), a lista de limiares deverá ser \(0{,}34;0{,}47;0{,}85\).

### 4.3. Como os limiares atuam na projeção?

Se \(t=\gamma_D\) é a última coordenada essencial, os \(U+1\) objetivos projetados são

\[
\begin{aligned}
p_0(t)&=\min(t,\eta_1),\\
p_i(t)&=\min\{\max(t,\eta_i),\eta_{i+1}\},
\quad i=1,\ldots,U-1,\\
p_U(t)&=\max(t,\eta_U).
\end{aligned}
\]

Didaticamente, os limiares dividem a variação da coordenada \(t\) em intervalos consecutivos. Em um intervalo, varia uma projeção; as outras permanecem constantes. Com \(\eta\) **ordenados crescentemente**, vale a identidade

\[
\sum_{i=0}^{U}p_i(t)=t+\sum_{j=1}^{U}\eta_j.
\]

Essa identidade permite recuperar a **ordem** da coordenada essencial ao olhar apenas os objetivos projetados. Se todos os objetivos projetados em \(t_a\) são menores ou iguais aos correspondentes em \(t_b\), necessariamente \(t_a\le t_b\).

Sem ordenação, os intervalos podem ser invertidos ou sobrepostos, algumas projeções podem ficar artificialmente constantes, e a identidade deixa de ser garantida. Isso muda as funções de objetivos e pode modificar a geometria da PF e os indicadores. A falta de ordenação, isoladamente, não autoriza concluir que todos os pontos ficaram dominados.

### 4.4. Por que isso importa na implementação?

A auditoria da versão de DPF3 examinada identificou que a geração de parâmetros no **BaseDPF** fornecia a sequência na ordem de geração e que **DPF3.evaluation()** a usava sem impor a ordenação crescente exigida. Quando \(M-D\ge2\), essa diferença pode alterar os valores objetivos. Em \(M=3,D=2\), há apenas um limiar, de modo que os problemas de cardinalidade desse caso não decorrem de ordenação.

Nos ensaios independentes com \(g=0\), comparando as **mesmas decisões** no avaliador examinado e na reconstrução publicada, observaram-se os seguintes **máximos de desvio entre componentes**:

| DPF3 | Desvio máximo observado |
|---|---:|
| \(M=3,D=2\) | 0 |
| \(M=5,D=2\) | 0,3844009575 |
| \(M=6,D=3\) | 0,3844009575 |
| \(M=7,D=4\) | 0,3844009575 |

São máximos **nos pontos ensaiados**, não cotas universais de erro. Não se deve ordenar indiscriminadamente um gerador compartilhado sem primeiro examinar outros consumidores: uma eventual implementação de correção precisa preservar as definições próprias dos demais DPFs.

### 4.5. Demonstração global da PS e da PF do DPF3 publicado

Considere um ponto \(y\) com \(g(y)=0\). Se outro ponto viável \(x\) o dominasse estritamente, a identidade das projeções ordenadas obrigaria os objetivos essenciais \(\gamma(x)\) a dominar os de \(\gamma(y)\). Como \(g(x)\ge0\), isso exigiria que o vetor esférico não negativo \(H_D(\theta(x))\), de norma 1, fosse componente a componente maior ou igual ao correspondente de \(y\), com uma melhoria estrita compatível com a dominância.

Mas **dois vetores não negativos de norma unitária não podem ter essa relação estrita**: aumentar uma coordenada sem diminuir as demais aumenta a norma. Portanto, nenhum ponto com \(g=0\) pode ser dominado.

Reciprocamente, para qualquer decisão com \(g>0\), manter os parâmetros angulares e fixar todas as coordenadas de distância em \(0{,}5\) produz uma decisão viável com \(g=0\). Sua avaliação melhora ou preserva todos os objetivos essenciais, com pelo menos uma melhora estrita, que a projeção ordenada preserva.

Logo, para o **modelo matemático publicado**,

\[
\boxed{PS_{DPF3}=\{x:g(x)=0\}},\qquad PF_{DPF3}=F(PS_{DPF3}).
\]

A prova vale para todas as dimensões intrínsecas elegíveis, sob as hipóteses da formulação. Ela certifica a **parametrização matemática contínua**, e não a correção automática de qualquer implementação computacional. Veja a [demonstração detalhada](gt-dpf3-global-pareto-proof-issue119.md).

## 5. DPF3: cancelamento numérico, PF finita e cobertura geométrica

### 5.1. O cancelamento em \(1-\cos(\theta)\)

Para \(D=2\), a primeira coordenada essencial utiliza

\[
1-\cos(\theta),\qquad \theta=\frac{\pi}{2}x_1^{100}.
\]

Quando \(\theta\) é pequeno, \(\cos(\theta)\) está muito próximo de 1. Em **float64**, \(\cos(\theta)\) pode arredondar para exatamente 1, de modo que a subtração devolve **zero** mesmo quando o resultado matemático é positivo. Esse fenômeno é chamado **cancelamento catastrófico**.

A identidade trigonométrica

\[
1-\cos(\theta)=2\sin^2(\theta/2)
\]

evita a subtração de quase iguais e é mais estável. Entretanto, para valores extremamente pequenos, o resultado ainda pode sofrer **underflow**, ficando abaixo da menor magnitude representável. Portanto, mudar só a expressão trigonométrica não resolve toda a discretização nem a precisão das demais coordenadas.

Na avaliação de 512 pontos de decisão com **160 dígitos decimais de precisão**, 511 valores da primeira coordenada eram matematicamente positivos (o ponto \(x_1=0\) é exatamente zero). Desses positivos:

| Cálculo em float64 | Valores positivos representados como zero |
|---|---:|
| Expressão direta \(1-\cos(\theta)\) | 423 |
| Expressão mais estável \(2\sin^2(\theta/2)\) | 12 |

Incluindo o zero matemático em \(x_1=0\), há **424 zeros** no cálculo direto e **13** no estável. Para 128 amostras, o cálculo direto perdeu 105 dos 127 positivos; o estável perdeu três. Os totais de zeros e os de **positivos perdidos** diferem exatamente pela presença do zero real — distinção necessária para interpretar as tabelas de ensaio.

A troca experimental somente da primeira componente estável elevou a quantidade de pontos não dominados dentro da amostra, mas **não** corrigiu a ordenação dos limiares, todas as demais perdas de precisão nem a distribuição geométrica.

### 5.2. A filtragem por não dominância não recupera GT

A implementação de DPF3 auditada devolvia aproximadamente:

| Pedido de pontos | Decisões geradas por ps() | Objetivos retornados por pf() |
|---|---:|---:|
| 128 | 128 | 23 |
| 512 | 512 | 88 |

Esses números ocorreram nos casos analisados com \(D=2\) e \(M=3\) ou \(M=5\). A função pf() filtrava objetivos com base em não dominância **interna à amostra**, reduzindo sua cardinalidade. Além de ser insuficiente para provar a otimalidade **global**, a filtragem impede associar automaticamente a \(i\)-ésima decisão ao \(i\)-ésimo ponto retornado da PF.

### 5.3. Referência angular com limite matemático demonstrável

Em vez de construir uma grade uniforme no parâmetro \(x_1\), a geometria do DPF3 sugere amostrar **diretamente o ângulo verdadeiro**:

\[
\theta_i=\frac{i}{n-1}\frac{\pi}{2},
\qquad
x_{1,i}=\left(\frac{2\theta_i}{\pi}\right)^{1/100},
\quad i=0,\ldots,n-1.
\]

Para \(D=2\), a curva publicada no espaço objetivo é **1-Lipschitz** em \(\theta\). A projeção ordenada também não aumenta distâncias além do correspondente deslocamento escalar. Como cada ângulo está no máximo a **meio passo** de um valor da grade, obtemos, em aritmética exata,

\[
\boxed{d_H(PF,GT_n)\le\frac{\pi}{4(n-1)}}.
\]

A distância usa objetivos **não normalizados**. O limite vale para a imagem exata da grade no **modelo publicado**, cujos pontos já foram demonstrados Pareto-ótimos. Para \(n=128\), o limite é aproximadamente **0,00618424**; para \(n=512\), é **0,00153698**.

A construção experimental produziu **128 e 512 objetivos distintos**, respectivamente, para \(M=3,5,10\), em contraste com os 23 e 88 pontos retornados pelo pf() anteriormente examinado. Trata-se de um **modelo de referência demonstrado e reproduzido**, não de uma simples melhoria visual.

### 5.4. Extensão a qualquer dimensão intrínseca

A parametrização hiperesférica \(H_D\) possui derivadas parciais angulares mutuamente ortogonais, cada uma de norma no máximo 1. A aplicação angular é, portanto, **1-Lipschitz** na norma euclidiana; a composição com as projeções ordenadas preserva esse limite.

Uma **grade tensorial** com \(k\) valores em cada um dos \(D-1\) ângulos livres possui \(k^{D-1}\) linhas de decisões e garante, em aritmética real exata,

\[
\boxed{d_H(PF,GT_k)\le
\frac{\sqrt{D-1}\,\pi}{4(k-1)}}.
\]

O limite é independente do número de objetivos redundantes \(M\), desde que a formulação e os limiares respeitem a definição publicada. É um limite **conservador**, mas verdadeiro para toda a PF contínua, ao contrário de uma distância medida apenas em pontos de sondagem.

Há duas ressalvas importantes. Primeiro, o custo nominal cresce **exponencialmente** com \(D\), tornando grades tensoriais inadequadas para muitas configurações de alta dimensão. Segundo, singularidades da parametrização esférica podem mapear decisões diferentes para o mesmo vetor de objetivos, reduzindo a cardinalidade efetiva.

| DPF3 | Valores por ângulo | Linhas da grade | Objetivos exatamente distintos em float64 | Limite matemático |
|---|---:|---:|---:|---:|
| \(M=5,D=3\) | 5 | 25 | 23 | 0,277680 |
| \(M=5,D=3\) | 9 | 81 | 75 | 0,138840 |
| \(M=6,D=4\) | 4 | 64 | 45 | 0,453450 |
| \(M=6,D=4\) | 7 | 343 | 266 | 0,226725 |

Essas contagens de objetivos distintos são **numéricas**; não demonstram quantos objetivos são matematicamente únicos. Há singularidades esféricas que fazem diferentes combinações de ângulos corresponderem a um mesmo objetivo, e o arredondamento também pode mascarar ou criar diferenças.

### 5.5. Precisão da transformação inversa

Para colocar um ângulo desejado numa decisão, a construção aplica \(x_j=(2\theta_j/\pi)^{1/100}\). Armazenar esse \(x_j\) em float64 e reconstruir \(\theta_j\) por \(x_j^{100}\) introduz arredondamento. A análise com **mpmath a 100 dígitos** encontrou, nos casos examinados, desvios angulares de ordem de **\(10^{-15}\) a \(10^{-14}\)** e desvios semelhantes nos objetivos reconstruídos.

Uma **medição de alta precisão não é um limite formal de arredondamento IEEE**. O limite de Hausdorff demonstrado é para a construção **ideal em aritmética exata**; uma garantia rigorosa para a implementação em float64 precisa incluir uma envoltória conservadora dos erros de transformação e avaliação. Para detalhamento da prova e dos valores observados, consulte [gt-dpf3-d2-angular-cover-proof-issue119.md](gt-dpf3-d2-angular-cover-proof-issue119.md) e [gt-dpf3-nd-cover-and-acceptance-issue119.md](gt-dpf3-nd-cover-and-acceptance-issue119.md).

## 6. DPF5: definição geométrica de dois componentes

### 6.1. Dimensões e distância

O DPF5 possui dois ramos da PF, geralmente chamados **P** e **D**. O ramo P tem dimensão geométrica \(M-1\), enquanto o ramo D tem dimensão geométrica \(D-1\), inferior à dimensão do primeiro. O problema exige \(N\ge M\) na definição considerada.

Sua função de distância é

\[
g(x)=(x_M-x_1)^2+\sum_{i=M+1}^{N}(x_i-0{,}5)^2.
\]

Logo \(g\ge0\). Para \(g=0\), deve-se ter **\(x_M=x_1\)** e todas as coordenadas \(x_i\) com \(i>M\) iguais a \(0{,}5\). As primeiras \(M-1\) variáveis são parâmetros de posição.

### 6.2. Equações corretas de P e D

Denote por \(H_q\) o mapa padrão de coordenadas hiperesféricas unitárias **não negativas**, com \(q\) coordenadas e \(q-1\) ângulos.

No ramo **P**, os objetivos são gerados por uma hiperesfera de \(M\) coordenadas, utilizando todos os \(M-1\) ângulos:

\[
u_P=H_M(\theta_1,\ldots,\theta_{M-1}),
\quad\theta_j=(\pi/2)x_j.
\]

No ramo **D**, calcula-se primeiro

\[
h=H_D(\theta_1,\ldots,\theta_{D-1}),
\]

e a primeira coordenada \(h_1\) é replicada \(M-D+1\) vezes, cada vez dividida por \(\sqrt{M-D+1}\), seguidas pelas demais \(D-1\) coordenadas de \(h\). Assim,

\[
u_D=(\underbrace{h_1/\sqrt{M-D+1},\ldots,
h_1/\sqrt{M-D+1}}_{M-D+1\ \mathrm{termos}},h_2,\ldots,h_D).
\]

Em ambos os ramos, \(u\ge0\) e \(\|u\|_2=1\). O avaliador publicado usa:

\[
F(x)=(1+g(x))u(x),
\]

com a **condição estrita**:

\[
u(x)=
\begin{cases}
u_P(x),&x_1<1/3,\\
u_D(x),&x_1\ge1/3.
\end{cases}
\]

Portanto, o ponto **\(x_1=1/3\) pertence ao ramo D**. A igualdade pode ter efeito material em grades estruturadas, que frequentemente incluem exatamente esse valor, embora uma amostra aleatória contínua quase nunca o produza.

### 6.3. Invariante esférico como verificação independente

Das equações segue que

\[
\boxed{\sum_{i=1}^{M} f_i(x)^2=(1+g(x))^2}.
\]

Isso vale tanto para \(g=0\) quanto para \(g>0\), em ambos os ramos, para as equações publicadas. O invariante é um controle **necessário** e muito útil para detectar índices repetidos, ângulos omitidos ou coordenadas escaladas incorretamente. Sozinho, não prova Pareto-otimalidade nem a correção de uma amostra finita.

Na versão de DPF5 auditada, o ramo P divergia da formulação publicada para diversas combinações de \(M,D\), e o ramo D também divergia para alguns casos, em particular \(D=4\). A condição de igualdade da implementação atribui \(x_1=1/3\) a P, em oposição à publicação.

### 6.4. Prova da PS/PF global do DPF5 publicado

Para qualquer decisão \(y\) com \(g(y)=0\), \(\|F(y)\|_2=1\). Se algum ponto viável \(x\) o dominasse estritamente, como todos os objetivos são não negativos, teríamos \(\|F(x)\|_2<1\). Entretanto, toda decisão viável satisfaz \(\|F(x)\|_2=1+g(x)\ge1\): contradição.

Isso vale também entre os ramos **P e D**. Portanto **nenhum ponto publicado com \(g=0\) é globalmente dominado por outro**, independentemente do ramo.

Reciprocamente, para qualquer \(x\) com \(g(x)>0\), podemos preservar os primeiros \(M-1\) parâmetros e construir uma decisão viável \(y\) com \(y_M=y_1\) e \(y_i=0{,}5\) para \(i>M\). Ela possui \(g(y)=0\), preserva o ramo e a direção \(u\), e satisfaz

\[
F(x)=(1+g(x))F(y).
\]

Portanto, \(y\) domina estritamente \(x\). Conclui-se, para o **modelo publicado**,

\[
\boxed{PS_{DPF5}=\{x:g(x)=0\}},\qquad PF_{DPF5}=F(PS_{DPF5}).
\]

A prova cobre os dois componentes e sua fronteira de definição. Ela **não implica** que um gerador finito cubra adequadamente ambos os componentes. Veja [gt-dpf5-global-pareto-proof-issue119.md](gt-dpf5-global-pareto-proof-issue119.md).

## 7. DPF5: divergências de implementação e cardinalidade

### 7.1. Diferenças nas equações por dimensão

A reconstrução independente identificou problemas nos índices e no reaproveitamento das expressões trigonométricas do ramo P em dimensões maiores. No caso \(D=2\), os ramos estritos coincidiam com a publicação para \(M=3\) e \(M=4\), mas o ramo P divergia para \(M\ge5\). Para \(D=3\) ou \(D=4\), outras configurações também apresentaram diferenças, incluindo o ramo D em \(D=4\).

Uma verificação do **resíduo máximo da soma dos quadrados** da norma do ramo P, com \(g=0\), forneceu:

| \(M,D\) | Resíduo máximo do avaliador examinado | Resíduo máximo da reconstrução publicada |
|---|---:|---:|
| \(3,2\) | \(2{,}22\times10^{-16}\) | \(2{,}22\times10^{-16}\) |
| \(5,2\) | 0,9943366 | \(4{,}44\times10^{-16}\) |
| \(6,3\) | 0,171273 | aproximadamente zero numérico |
| \(6,4\) | 0,314780 | aproximadamente zero numérico |

No ramo **D** para \(M=6,D=4\), o resíduo máximo auditado foi aproximadamente **0,321684**, enquanto a reconstrução independente respeitava a identidade. São discrepâncias de equação — não podem ser atribuídas apenas à escolha de pontos da PS ou à filtragem.

Em ensaios com **\(g=0{,}04\)** e \(D=2\), a falha do ramo P persistiu. O resíduo máximo da identidade esférica escalada foi aproximadamente **0,07764** em \(M=5\), **0,17928** em \(M=6\) e **0,49334** em \(M=10\). Logo, não se trata de um erro exclusivo da geração de pontos sobre a PS.

### 7.2. A amostra de DPF5 pode ter muito menos objetivos distintos

Na implementação auditada, quando \(M=3\), a função ps() usa uma grade com **floor(sqrt(n_points))** valores por eixo. Consequentemente:

| \(M=3,D=2\) | Decisões pedidas | Decisões retornadas | Objetivos distintos no avaliador da época | Objetivos distintos na reconstrução publicada |
|---|---:|---:|---:|---:|
| Primeiro exemplo | 128 | 121 | 51 | 51 |
| Segundo exemplo | 512 | 484 | 190 | 169 |

Não existe contradição entre **190 e 169**: são os resultados da **mesma grade de decisões**, mas avaliados com **duas definições diferentes da fronteira entre P e D**. A grade de 22 valores por eixo usada no pedido 512 inclui \(x_1=1/3\). No avaliador antigo esses pontos pertenciam a P; na equação publicada pertencem a D.

Com as equações publicadas e pedido 512, há **154 decisões no ramo P**, todas com objetivos distintos, e **330 decisões no ramo D**, mas somente **15 objetivos distintos**. Portanto, o componente degenerado concentra múltiplas decisões com a mesma imagem em espaço objetivo.

### 7.3. Dominância interna no avaliador incorreto

No DPF5 \(M=5,D=2\), a amostra de 512 decisões apresentou **462 objetivos dominados dentro da amostra** sob a função de avaliação examinada. O componente P tinha 183 decisões, das quais 151 eram dominadas dentro do próprio ramo; o conjunto completo aumentava a contagem para 462, em razão de relações entre os ramos.

Ao reconstruir o **modelo matemático publicado** sobre as decisões consideradas, todos os 462 casos de dominância observada desapareceram, coerentemente com a prova de que todos os pontos com \(g=0\) são globalmente Pareto-ótimos.

Isso mostra por que devemos separar **defeito no avaliador** de **defeito no amostrador**. Uma amostra pode estar mal distribuída mesmo depois que os valores matemáticos passam a ser Pareto-ótimos. Fazer apenas uma filtragem não substitui a correção das equações.

## 8. Cobertura das duas componentes do DPF5

### 8.1. O que foi medido

A investigação avaliou as **decisões produzidas pelo gerador existente** segundo as **equações independentes publicadas**. Foram gerados pontos de sondagem independentes em P e D. Para cada ramo, mediu-se a distância em espaço objetivo até o ponto amostrado mais próximo **do mesmo ramo**, após contagem dos objetivos distintos. Essa separação evita mascarar uma deficiência de cobertura em P por meio de muitos pontos em D.

Os máximos e medianas obtidos com sondagem finita descrevem a qualidade das **configurações estudadas**. Não constituem um limite superior certificado para a distância de Hausdorff da PF contínua.

| \(M,D\) | PS efetiva, pedido 512 | Objetivos distintos publicados | Mediana P | Máximo observado P | Mediana D | Máximo observado D |
|---|---:|---:|---:|---:|---:|---:|
| \(3,2\) | 484 | 169 | 0,02977 | 0,07854 | 0,01596 | 0,03738 |
| \(5,2\) | 512 | 512 | 0,12904 | 0,32301 | 0,00128 | 0,01118 |
| \(6,3\) | 512 | 512 | 0,16216 | 0,49741 | 0,02042 | 0,09446 |
| \(6,4\) | 512 | 512 | 0,16216 | 0,49741 | 0,04946 | 0,18339 |

Por exemplo, em \(M=5,D=2\), a mediana da distância de cobertura do ramo P foi aproximadamente cem vezes a de D. É uma diferença de **alocação e geometria**, e não prova de que uma das regiões matemáticas esteja ausente do problema.

### 8.2. Por que não existe uma proporção universal simples?

P é uma variedade de dimensão \(M-1\), enquanto D tem dimensão \(D-1\). Em dimensões maiores, cobrir adequadamente P pode exigir muitos mais pontos que cobrir D, mas a distribuição ótima também depende da extensão, curvatura, extremos, interfaces e orçamento total.

Ter **50% de pontos em cada ramo** não é necessariamente equilibrado em erro geométrico. Distribuir segundo uma regra fixa como **dois terços para P** tampouco possui justificativa universal. A geometria da PF não determina sozinha uma medida probabilística única para misturar componentes de dimensões distintas.

A conclusão não é que seja impossível estabelecer bons amostradores. É que cada **DPF5 configurado** precisa de uma política geometricamente justificada, com orçamento efetivo, extremos e critério de erro explicitados.

### 8.3. Comparação com sequências Sobol

Uma alternativa experimental utilizou sequências **Sobol embaralhadas deterministicamente**, com pontos especiais em fronteiras, preservando o **mesmo número de objetivos distintos em cada ramo** da amostra anterior. P recebeu \(M-1\) parâmetros angulares; D recebeu somente os \(D-1\) parâmetros essenciais.

Foram comparadas medidas de distância mediana, percentil 95 e maior distância observada em **três sementes independentes de sondagem**, para pedidos 128 e 512. A alternativa teve **resultados mistos**, mesmo com orçamento efetivo igual.

Alguns exemplos de melhoria da **distância mediana** (valor da implementação anterior menos valor do candidato; **positivo favorece Sobol**):

| \(M,D\) | Pedido | Três diferenças de mediana em P | Três diferenças de mediana em D |
|---|---:|---|---|
| \(3,2\) | 128 | +0,00262; −0,00102; +0,00247 | −0,00423; −0,00839; −0,00731 |
| \(5,2\) | 512 | +0,00435; +0,00546; +0,00815 | +0,00036; +0,00034; +0,00038 |
| \(6,3\) | 128 | −0,01026; −0,02384; −0,01700 | +0,00294; +0,00386; +0,00189 |
| \(6,3\) | 512 | +0,00281; −0,00250; +0,00244 | +0,00204; +0,00201; +0,00081 |

Portanto, **simplesmente substituir um amostrador por Sobol não resolve de forma geral a distribuição dos objetivos**. Uma melhoria na mediana também pode vir acompanhada de piora nos extremos ou no maior afastamento. Os pontos de fronteira incluídos são marcos úteis, mas não comprovam cobertura de **todas** as regiões fronteiriças.

### 8.4. Alocação entre ramos com orçamento total fixo

Outro experimento manteve a **quantidade total estratificada de objetivos distintos** e comparou a distribuição então existente com alternativas que destinavam **metade** ou **aproximadamente dois terços** dos pontos a P. Utilizaram-se três sementes de geração e três sementes de sondagem, com sondagens independentes por componente.

A tabela informa a **maior distância de sondagem observada, maximizada entre as sementes**, para 512 pontos pedidos; **menor é melhor**:

| \(M,D\) | Distribuição anterior | 50% para P | Aproximadamente 67% para P |
|---|---:|---:|---:|
| \(3,2\) | **0,07705** | 0,10917 | 0,08702 |
| \(5,2\) | 0,41784 | 0,34187 | **0,33189** |
| \(6,3\) | 0,45091 | 0,42947 | **0,38824** |
| \(6,4\) | 0,45091 | 0,42947 | **0,38824** |

Em \(M=3,D=2\), a distribuição anterior já destinava cerca de **91% dos objetivos distintos a P**, portanto uma política fixa de dois terços **retirava** pontos desse ramo e piorava a medida. Nas configurações maiores examinadas, aumentar a participação de P reduziu o maior afastamento observado. Com orçamento 128, os efeitos também variaram, por vezes de modo acentuado.

O experimento demonstra uma **dependência da geometria e do orçamento**. Não estabelece uma proporção universal ótima nem certifica a pior distância possível na PF contínua. Também não representa uma comparação estatística ampla de todos os métodos disponíveis.

## 9. Consequências para IGD, IGD+ e hipervolume

A pesquisa separou dois efeitos que não devem ser confundidos.

**Efeito da avaliação:** manter exatamente as mesmas **decisões** e calcular os objetivos com o avaliador existente e com a reconstrução publicada. Diferenças aqui significam que a definição do MOP muda os próprios resultados avaliados dos algoritmos.

**Efeito da referência:** congelar exatamente a mesma **população de vetores de objetivos** e calcular os indicadores com duas referências diferentes. Diferenças aqui são atribuídas à escolha do GT, **sem nova execução de MOEA**.

O segundo estudo verificou a identidade da população de objetivos com **SHA-256**, manteve 64 pontos de referência candidatos e 16 pontos de uma população fixa, utilizando implementações reais de **IGD, IGD+ e HV** do repositório.

No caso \(M=5,D=2\), semente 101, ao substituir somente a referência, observaram-se as seguintes diferenças (*reconstrução publicada menos avaliador anterior*):

| Quantidade | Diferença observada |
|---|---:|
| IGD | +0,02731 |
| IGD+ | +0,00270 |
| HV relativo/absoluto ao GT | +0,02663 |

Em \(M=5,D=2\), semente 202, as diferenças foram **+0,02933**, **−0,00076** e **+0,01801**, respectivamente. Para \(M=6,D=3\), semente 101, foram **+0,00967**, **−0,00020**, **+0,00562**; para \(M=6,D=4\), semente 202, **+0,03978**, **+0,01271**, **−0,00845**. Os sinais podem variar; os dados não implicam que todo indicador sempre aumente ou diminua.

A interpretação exige duas precisões adicionais:

- A implementação de **IGD e IGD+** examinada usa normalização **zero-to-one dependente da própria referência**. Assim, mudar GT pode afetar não só a localização dos pontos, mas também a normalização interna do indicador. A diferença não deve ser descrita como apenas uma distância geométrica sob limites fixos.
- O estudo de **HV** utilizou os mesmos limites **ideal = 0**, **nadir = 2** e ponto de referência normalizado **1,1** em ambos os cálculos. O hipervolume da **população** permaneceu igual; o volume associado ao **GT** mudou, alterando a razão entre os volumes. Não se deve afirmar que o volume de soluções do MOEA mudou quando apenas o GT foi substituído.

Esses resultados são sensibilidades numéricas de exemplos controlados, não previsões universais do desempenho futuro dos MOEAs. Para analisar o efeito completo de corrigir a função objetivo, seria necessário avaliar o histórico de decisões sob as duas definições ou executar estudos novos sob o problema corrigido.

## 10. Referências históricas, regressão e certificação

### 10.1. O que significa dizer que um baseline histórico ficou “errado”?

Um baseline tem pelo menos dois papéis distintos:

1. **Registro histórico de comportamento:** mostra o que determinada versão do programa efetivamente calculava. Mesmo que o código tenha posteriormente sido considerado matematicamente incorreto, o registro é verdadeiro como **evidência daquele comportamento e daquela versão**.
2. **Padrão-ouro científico da formulação pretendida:** só é válido se a definição matemática, a geração e a precisão da referência tiverem sido adequadamente verificadas.

Portanto, uma referência obtida de uma implementação que não respeita o artigo **não deve continuar a ser apresentada sem ressalvas como a PF correta publicada**. Isso não significa que todos os baselines do projeto estejam incorretos ou que todo teste de regressão baseado numa versão anterior deva ser apagado.

A preservação histórica exige **identificar a referência antiga como associada à fórmula e versão antigas** e introduzir uma eventual referência científica sucessora como artefato **novo, versionado e verificável**. Nunca se deve sobrescrever uma evidência histórica somente para que os testes fiquem verdes.

### 10.2. Corrigir o avaliador e substituir GT são intervenções diferentes

Uma correção das equações de **evaluation()** muda o problema matemático avaliado e pode alterar os resultados e trajetórias de MOEAs. Uma troca de **GT** muda a referência dos indicadores e pode afetar medições mesmo sem alterar as soluções dos algoritmos. As duas intervenções exigem comparações independentes e proveniência individual.

As ADRs têm papéis complementares:

- [ADR 0006](adr/0006-benchmarks-reimplementation.md): responsabilidade de cada MOP e implementação de PS/PF analíticos.
- [ADR 0017](adr/0017-comprehensive-calibration-and-testing-framework.md): requisitos científicos de GT e de calibração independente.
- **ADR 0018:** valores históricos de densidade de referência; a ADR 0057 substitui em princípio o ideal de uma densidade fixa universal, preservando os padrões efetivos até uma política operacional aprovada.
- [ADR 0056](adr/0056-versioned-certification-and-performance-reference-lifecycle.md): ciclos de vida distintos de certificado numérico, baseline de desempenho e certificado de desempenho; registros versionados e imutáveis, sem regeneração automática de referências.
- [ADR 0057](adr/0057-proposed-purpose-specific-gt-budgets.md): GT por MOP e finalidade; calibração explícita; seleção automática de orçamento aceita **em princípio**, mas sem limiares numéricos gerais definidos.

Um teste histórico pode verificar um **invariante matemático** que permanece válido depois de uma correção, ou pode registrar deliberadamente números associados à implementação anterior. É necessário conhecer sua finalidade antes de decidir se uma mudança de comportamento é correção ou regressão. **Qualquer alteração em teste de regressão preexistente exige aviso e autorização explícita**, e uma mudança contraditória a uma ADR aceita exige análise e autorização **antes** de ser realizada.

A pesquisa identificou como referências sensíveis, entre outras, **tests/stability_reference_gt.json**, **tests/stability_gt_geometry_reference.json**, **tests/unit/test_dpf3_emd_regression.py** e os sidecars calibrados. Esses arquivos podem ter propósitos diferentes e não devem ser substituídos em bloco. As referências persistidas merecem identificação de fórmula, configuração, origem e versão de software.

### 10.3. Calibração, certificados e FAIR são assuntos distintos

**GT é parte da API científica do MoeaBench, independentemente de FAIR.** A remoção temporária das métricas FAIR não retira a necessidade de PS/PF corretos nem transforma calibrate() em um método próprio daquela pesquisa.

Do mesmo modo, um **certificado numérico** prova que determinado conjunto de testes numéricos e invariantes foi satisfeito no ambiente examinado; não gera automaticamente uma nova referência histórica. Um **baseline de desempenho** é um estudo empírico versionado e aprovado; uma falha de desempenho não autoriza regravá-lo. A ADR 0056 separa explicitamente esses mecanismos.

## 11. O que é possível provar, medir e afirmar

A experiência com DPF3 e DPF5 estabelece uma hierarquia útil de evidências:

| Afirmação | Evidência necessária | O que não basta |
|---|---|---|
| Implementação respeita o artigo | Comparação independente das equações e casos de fronteira | Resultados visualmente plausíveis |
| Amostras pertencem à PF global | Prova analítica ou outro argumento global confiável | Filtragem por não dominância |
| Parametrização alcança toda a PF | Demonstração de completude e de todos os componentes | Identidade geométrica necessária isolada |
| GT finito tem cobertura limitada | Limite matemático no espaço de objetivos declarado, incluindo arredondamento se aplicável | Maior distância entre sondagens aleatórias |
| Quantidade solicitada foi satisfeita | Contagem efetiva de decisões e objetivos distintos | Valor passado a n_points |
| Precisão numérica está garantida | Análise ou limite rigoroso de erro para a implementação | Concordância em poucos pontos de alta precisão |
| Indicadores são comparáveis | Mesma referência, normalização documentada e proveniência | Nomes idênticos dos indicadores |
| Referência é historicamente válida | Arquivo e versão congelados, interpretação explícita | Recalcular retroativamente com código novo |

Para o **DPF3 publicado**, a parametrização global e um **limite de cobertura para uma grade angular ideal** foram demonstrados. Os ensaios numéricos de alta precisão quantificaram erros de ordem \(10^{-14}\) nas configurações estudadas, mas tal resultado **não** é uma certificação geral em aritmética IEEE. A grade tensorial também pode exigir \(k^{D-1}\) pontos, tornando custos altos inevitáveis com esse método específico.

Para o **DPF5 publicado**, a parametrização global de **ambos os componentes** e sua pertinência à PF foram demonstradas; as medições de cobertura de amostras finitas e as alternativas de alocação foram **empíricas** e dependentes da geometria. Da definição matemática, isoladamente, não decorre uma proporção universal de pontos entre componentes de diferentes dimensões.

Essas distinções estabelecem o que podemos declarar cientificamente sobre uma referência, sem transformar hipóteses ou medições amostrais em garantias universais.

## 12. Especificação científica de um gerador de GT confiável

Um gerador de referência apropriado para o MoeaBench deve:

1. **Partir das equações corretas** e das restrições do MOP, incluindo todas as configurações declaradas suportadas.
2. **Gerar decisões sobre uma PS caracterizada**, ou justificar a natureza aproximada/empírica da fonte quando solução analítica não existe.
3. **Preservar todos os componentes e extremos relevantes** da PF, inclusive interfaces e regiões de dimensão reduzida.
4. **Declarar cardinalidades:** pedido nominal, decisões geradas, objetivos retornados, duplicatas e quantidade efetiva de vetores distintos.
5. **Avaliar a cobertura** no espaço e métrica apropriados, distinguindo garantias analíticas de estatísticas de sondagem.
6. **Controlar precisão**, especialmente subtração de valores próximos, underflow, transformação inversa e singularidades da parametrização.
7. **Respeitar um custo computacional delimitado** em função da finalidade, da dimensão e da precisão exigida.
8. **Ser determinístico ou controladamente estocástico**, com sementes e ambiente identificáveis.
9. **Registrar a proveniência** do GT: identidade e parâmetros do MOP, versão da definição matemática, versão do código, método, orçamento efetivo, normalização e hipóteses de qualidade.
10. **Preservar referências históricas** e separar a alteração da equação de avaliação, do amostrador e dos padrões de calibração.

Nenhuma dessas propriedades obriga todos os MOPs a usar o mesmo algoritmo. Ao contrário, a geometria conhecida do MOP deve orientar a implementação do método.

## 13. Referências matemáticas e rastreabilidade da auditoria

**Fonte matemática primária:** Zhen e colaboradores, *Multiobjective Test Problems with Degenerate Pareto Fronts*, arXiv:1806.02706v2 (2022), seções IV-C, IV-E e IV-F, equações (15)–(16) e (19)–(21). [Artigo no arXiv](https://arxiv.org/abs/1806.02706) e [versão disponibilizada por autor](https://liangli-zhen.github.io/assets/pdf/arxiv2022_DPF.pdf).

**Demonstrações matemáticas completas:** [DPF3, PS/PF global](gt-dpf3-global-pareto-proof-issue119.md); [DPF5, PS/PF global](gt-dpf5-global-pareto-proof-issue119.md); [DPF3, cobertura angular D=2](gt-dpf3-d2-angular-cover-proof-issue119.md); [DPF3, cobertura angular em dimensão intrínseca geral](gt-dpf3-nd-cover-and-acceptance-issue119.md).

**Verificações da implementação e precisão:** [auditoria inicial de cardinalidade e dominância](gt-dpf3-dpf5-audit-issue119.md); [confronto inicial com as equações do artigo](gt-dpf-published-formula-triage-issue119.md); [reconstrução das fórmulas para D geral](gt-dpf-general-d-evidence-issue119.md); [núcleos numéricos e identidades esféricas](gt-dpf-numeric-kernel-evidence-issue119.md); [varredura em alta precisão](gt-dpf-high-precision-sweep-issue119.md); [análise fora da frente e isolamento de dependências](gt-dpf5-offpareto-issue119.md).

**Qualidade finita e efeitos nos indicadores:** [cobertura por ramo do DPF5](gt-dpf5-finite-sampler-coverage-issue119.md); [comparação com Sobol](gt-dpf5-branch-sobol-comparison-issue119.md); [alocação P/D em orçamento total fixo](gt-dpf5-branch-allocation-issue119.md); [efeitos sobre IGD, IGD+ e HV](gt-dpf5-paired-indicator-impact-issue119.md).

**Governança de alterações:** [inventário de impacto em produção](gt-dpf-production-impact-inventory-issue119.md), [plano de correção e migração versionada](gt-dpf-correction-migration-proposal-issue119.md) e [revisão da PR de pesquisa](gt-dpf-premerge-review-issue119.md). Esses arquivos são **registros especializados** com ensaios, histórico e alternativas; o presente documento contém os conceitos e resultados consolidados que devem permanecer compreensíveis independentemente daquela cronologia.

A [issue #119](https://github.com/monacofj/moeabench/issues/119) reúne o histórico do problema, e a [PR #139](https://github.com/monacofj/moeabench/pull/139) contém as reconstruções independentes, os dados reproduzíveis, novos testes e os relatórios que sustentam estas conclusões.

---

**Princípio orientador:** o GT deve representar o **problema matemático correto**, com **cobertura, precisão e proveniência apropriadas à finalidade**. Cada MOP é responsável por sua geometria e estratégia de geração; infraestrutura pode ser compartilhada quando a validade científica de seu emprego estiver demonstrada.