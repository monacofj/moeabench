# O Confronto das Verdades: Auditoria Científica do Conjunto Legacy2_optimal

O processo de evolução do framework MoeaBench, rumo à versão 0.7.5, exigiu um escrutínio rigoroso sobre o que definimos como "verdade fundamental" (Ground Truth). Após o estabelecimento inicial da v0.7.5, procedemos ao confronto sistemático com o conjunto **Legacy2_optimal**—uma coleção de frentes ótimas de alta precisão geradas por execuções extensivas do sistema anterior. Este documento narra as descobertas dessa auditoria e as interpretações geométricas que moldaram a consolidação matemática do projeto.

## A Harmonia Fundamental nos Problemas de Base

A análise das famílias DTLZ1 a DTLZ6 revelou uma convergência matemática notável entre a implementação atual e os dados do Legacy2_optimal. Em problemas de geometria esférica e linear simples, a discrepância (Gap) mante-se na ordem de $10^{-8}$ a $10^{-9}$. 

No caso do DTLZ1, o invariante de soma ($\sum f_i = 0.5$) foi respeitado com precisão absoluta, confirmando que a definição do hiperplano linear permanece estável entre as versões. De forma análoga, a "esfera perfeita" dos problemas DTLZ2 a DTLZ6 demonstrou que o cálculo do $g$-factor e a propagação das funções trigonométricas na v0.7.5 estão em plena harmonia com a intenção original do sistema. Esses resultados não são apenas números; eles representam o selo de qualidade que permite ao pesquisador confiar que a base do seu benchmark é sólida e reprodutível.

## O Enigma Geométrico do DTLZ8 e DTLZ9

A harmonia, contudo, é rompida ao adentramos nos problemas baseados em restrições e estruturas de blocos. O confronto revelou falhas significativas de invariância no Legacy2_optimal para os problemas DTLZ8 e DTLZ9.

No DTLZ8, onde a fronteira é definida pela interseção de restrições de hiperplanos, os pontos do Legacy2_optimal apresentaram desvios substanciais. Enquanto a v0.7.5 impõe que a relação entre o último objetivo e os demais satisfaça estritamente as equações de manifold (ex: $f_M + 4f_i = 1$), os dados legados demonstraram uma dispersão que sugere um "estacionamento" em zonas viáveis, mas não perfeitamente ótimas. 

Essas discrepâncias sugerem que, mesmo com alta fidelidade, algoritmos estocásticos podem sofrer de um viés de busca ou tolerância excessiva em restrições de desigualdade. Por esta razão, a v0.7.5 opta pelo rigor analítico puro, utilizando solvers guiados que garantem a satisfação das equações de repouso, elevando o patamar de exigência para qualquer algoritmo que pretenda convergir para essas frentes.

Essas discrepâncias sugerem que, mesmo com alta fidelidade, algoritmos estocásticos podem sofrer de um viés de busca ou tolerância excessiva em restrições de desigualdade. Por esta razão, a v0.7.5 opta pelo rigor analítico puro, utilizando solvers guiados que garantem a satisfação das equações de repouso, elevando o patamar de exigência para qualquer algoritmo que pretenda convergir para essas frentes.

## A Anatomia do Viés: O Caso DTLZ4 e as "Nuvens Esparsas"

Um dos momentos mais didáticos desta auditoria ocorreu durante a inspeção visual do DTLZ4 para três objetivos ($M=3$). Observou-se que os pontos da v0.7.5 pareciam "flutuar" de forma esparsa, deixando vazios no centro da superfície esférica. Uma análise profunda da função de avaliação revelou o culpado: o expoente de viés $x^{100}$.

Em sua formulação clássica, o DTLZ4 utiliza essas potências extremas para testar a capacidade do algoritmo de lidar com distribuições não-uniformes no espaço de decisão. Ao realizarmos uma amostragem uniforme no espaço de decisão ($X$), o mapeamento angular ($\Theta$) colapsa quase todos os pontos para as bordas e eixos da esfera ($f \approx 0$ ou $f \approx 1$). Os raros pontos que caem no centro da esfera tornam-se, visualmente, pontos isolados que parecem erros de cálculo—os chamados "floating points".

Para retificar essa distorção e fornecer um alvo de performance que cobrisse toda a superfície do problema, reimplementamos a lógica de amostragem analítica (`ps()`) do DTLZ4. A nova abordagem agora realiza o sorteio uniforme diretamente no espaço de coordenadas esféricas ($\Theta$) e inverte a relação de potência para encontrar o valor de $X$ correspondente. O resultado é uma frente de Ground Truth que combina o rigor matemático da esfera com uma densidade visual que permite métricas de performance (como IGD e Hypervolume) muito mais precisas e justas.

## Conclusão e Caminho Adiante

A auditoria do Legado2 cumpre seu papel didático ao nos mostrar que a precisão numérica não é o único pilar de um benchmark científico; a interpretação geométrica do espaço de busca é igualmente vital. Ao consolidarmos a v0.7.5 com solvers guiados e amostragem ajustada para o DTLZ4, não estamos apenas corrigindo arquivos, estamos refinando a linguagem matemática com a qual o MoeaBench se comunica com as futuras heurísticas de otimização.

---
*Escrito como parte do esforço de retificação científica da versão 0.7.5.*
