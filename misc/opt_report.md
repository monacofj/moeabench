# Relatórios de Auditoria Científica: Confronto de Verdades (v0.7.5 vs Legado)

## Introdução: A Natureza da Verdade em Otimização
Em pesquisa científica de algoritmos evolucionários multiobjetivo (MOEAs), a definição da "Frente de Pareto Real" é a âncora que valida toda métrica de convergência. Historicamente, muitas implementações, inclusive o conjunto de dados **Legado (MoeaBench/legacy)**, basearam suas verdades em processos de **busca e filtragem**. Consistia em rodar algoritmos exaustivamente, coletar os pontos mais próximos do ótimo e congelá-los como referência.

A versão **v0.7.5** do MoeaBench rompe com esse paradigma ao adotar a **Síntese Analítica**. Em vez de procurar a frente, nós a calculamos diretamente a partir das definições matemáticas originais dos papers de benchmark (Deb et al., 2002; Zhen et al., 2018). Este relatório detalha por que essa mudança não é apenas estética, mas uma retificação necessária da precisão científica.

## 1. Família DTLZ: Estabilidade e Invariantes
A família DTLZ baseia-se em geometrias fundamentais: hiperplanos lineares (DTLZ1) e superfícies esféricas (DTLZ2-4).

### DTLZ1: A Prova do Somatório
No DTLZ1, a soma dos objetivos ($\sum f_i$) deve ser exatamente $0.5$ em qualquer ponto da frente de Pareto. 
*   **O Confronto**: No Legado, encontramos somatórios em torno de $0.46$. Isso indica que a frente legada não era apenas ruidosa, mas estava "encolhida". 
*   **Veredito**: A v0.7.5 entrega o somatório exato de $0.5$. A versão atual está em conformidade absoluta com o artigo de Deb, enquanto o legado sofria de uma convergência incompleta ou erro de amostragem no kernel original.

### DTLZ2-4: Uniformidade Esférica
Problemas esféricos dependem da invariante $\sum f_i^2 = 1.0$.
*   **Análise**: O rastro Vermelho (Legado) apresenta uma distribuição de pontos que, embora correta em direção, falha na densidade. O rastro Azul (v0.7.5) é uma malha perfeita onde cada ponto respeita a invariante esférica com precisão de máquina.
*   **Veredito**: A v0.7.5 é superior por eliminar o "jitter" estocástico, fornecendo um padrão de comparação que não introduz viés de amostragem nas métricas (como IGD).

### DTLZ5 e DTLZ6: A Degeneração Controlada
Estes problemas introduzem degeneração na frente de Pareto (redução de dimensionalidade).
*   **Análise**: Enquanto o Legado sofria para manter a curvatura correta sob baixa dimensionalidade, a v0.7.5 utiliza a parametrização analítica exata que força os pontos a seguirem a "esfera espremida".
*   **Veredito**: A v0.7.5 garante que a frente seja uma curva ou superfície perfeitamente lisa, fiel ao mapeamento projetivo do artigo de 2002.

## 2. Paradoxos Geométricos: DTLZ7, DTLZ8 e DTLZ9
Aqui o confronto visual revela as interpretações mais profundas da otimalidade.

### DTLZ7: A Filtragem da Não-Dominância
O DTLZ7 possui uma frente desconectada em 4 ilhas. 
*   **A Discrepância**: O Legado mostra um rastro contínuo, conectando as ilhas. A v0.7.5 mostra apenas as ilhas isoladas.
*   **A Explicação**: As conexões presentes no legado são, matematicamente, **regiões dominadas** (sub-ótimas). Elas aparecem em processos de busca porque o algoritmo "passa por lá". A v0.7.5, ao aplicar um filtro de não-dominância analítico estrito, remove esses pontos. 
*   **Veredito**: O Azul está "certo" no sentido de Pareto-Ótimo. O Vermelho mostra a superfície total de resposta, o que pode confundir a análise de convergência.

### DTLZ8: O Desafio das Restrições
O DTLZ8 é um caso único onde a frente é definida por interseções de restrições complexas.
*   **Análise**: Por ser um problema de difícil síntese analítica pura, a v0.7.5 utiliza uma busca de alta fidelidade (NSGA-III com parâmetros massivos) congelada como "Frozen Truth". O legado apresentava falhas de cobertura em certas quinas da frente.
*   **Veredito**: Embora ambos sejam baseados em busca, a v0.7.5 fornece uma densidade e cobertura de bordas superior, representando melhor os limites das restrições lineares e não-lineares do problema.

### DTLZ9: A Linha vs A Nuvem
O DTLZ9 impõe a restrição $f_1 = f_2 = \dots = f_{M-1}$.
*   **Confronto**: O legado apresenta uma nuvem dispersa. A v0.7.5 apresenta uma linha (arco) unidimensional perfeita.
*   **Veredito**: O artigo original define o DTLZ9 com restrições que reduzem a dimensionalidade da frente. A v0.7.5 é a única que honra essa restrição estrutural. O legado tratava o problema como uma busca de caixa-preta, perdendo a essência da "curva de Pareto".

## 3. Família DPF: A Complexidade Retificada
Os problemas DPF (Degenerate Pareto Fronts) são projeções de frentes de baixa dimensão em espaços de alta dimensão.

### DPF1 e DPF3: A Pureza Linear
Diferente dos seus irmãos quadráticos, o DPF1 e DPF3 mantêm projeções lineares.
*   **Análise**: O rastro Vermelho (Legado) muitas vezes falhava em convergir para a base $g=0$, resultando em uma nuvem "acima" do ótimo real. O Azul (v0.7.5) aterrissa exatamente no limite inferior da função g.

### DPF2 e DPF4: O Resgate do Quadrado
A auditoria 3D revelou que o Azul estava inicialmente em uma magnitude diferente do Vermelho nestes dois problemas. 
*   **A Causa**: A v0.7.5 havia inicialmente linearizado as projeções por simplificação. No entanto, o artigo de Zhen et al. (2018) especifica projeções **quadráticas**.
*   **A Correção**: Com a restauração dos termos quadráticos na v0.7.5, a sobreposição com o legado tornou-se perfeita quanto à posição, mas o Azul agora mantém uma densidade uniforme (via amostragem estruturada em `linspace`), enquanto o Vermelho era ruidoso e disperso.

### DPF5: A Superfície Estruturada
No DPF5 ($D=3$), a frente deixa de ser uma linha e torna-se uma superfície 2D mergulhada em 3D.
*   **Análise**: O legado produzia uma "chuva de pontos" desorganizada. A v0.7.5 gera uma malha reticulada (Grid) que permite visualizar a continuidade bi-dimensional da superfície.
*   **Veredito**: A v0.7.5 é fundamental para diagnósticos visuais de continuidade, algo impossível de avaliar com a amostragem ruidosa do legado.

## Conclusão: O Novo Padrão Ouro
O confronto entre os dados **Legado** e a **v0.7.5** revela uma evolução da "Busca por Aproximação" para a "Síntese de Precisão". 

O Legado é um registro histórico de como os algoritmos se comportavam no passado, carregando consigo os ruídos da busca estocástica. A **v0.7.5 Ground Truth** é a manifestação limpa das equações dos papers originais. Para fins de auditoria científica e benchmark de alta fidelidade, a v0.7.5 é a referência definitiva, assegurando que o pesquisador está comparando seus resultados com a Geometria Real do problema, e não com uma amostragem histórica possivelmente degradada.
