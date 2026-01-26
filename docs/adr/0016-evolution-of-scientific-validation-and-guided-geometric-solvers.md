# ADR 0016: Evolução da Validação Científica e Solvers Geométricos Guiados

## Status
Aceito (Implementado na v0.7.6)

## Contexto
Durante a consolidação da v0.7.5 (ADR 0015), identificamos que a dependência de "Proxies Heurísticos" (Tier B) para problemas restritos como o DTLZ8 e a amostragem uniforme no espaço de decisão para problemas enviesados como o DTLZ4 introduziam artefatos geométricos inaceitáveis. O confronto com os dados de alta precisão do **Legacy2_optimal** revelou que a "verdade" baseada em buscas estocásticas sofria de viés de convergência e dispersão espacial, comprometendo métricas sensíveis à densidade (IGD e Hypervolume).

## Decisões Técnicas

### 1. Migração para Solvers Analíticos Guiados (DTLZ8)
Abandonamos a abordagem de "União de Múltiplas Sementes" do NSGA-III para o DTLZ8. 
*   **Decisão**: Implementamos um motor de reconstrução de variedade que inverte as restrições lineares do problema. 
*   **Lógica**: O solver amostra a curva central e as "hastes" laterais do DTLZ8 de forma determinística, garantindo que cada ponto gerado pertença estritamente à interseção das restrições ativas. 
*   **Resultado**: O DTLZ8 passa do Tier B (Probabilístico) para o Tier A (Analítico), eliminando o ruído estocástico na definição do Ground Truth.

### 2. Retificação Angular das Variedades Enviesadas (DTLZ4)
O DTLZ4 utiliza uma potência $x^{100}$ que colapsa amostras uniformes do espaço de decisão para os eixos da esfera.
*   **Decisão**: Substituímos a amostragem em $X$ pela amostragem uniforme no espaço angular ($\Theta$).
*   **Lógica**: O gerador sorteia ângulos uniformemente no triângulo esférico ($\Theta \in [0, \pi/2]$) e realiza a inversão matemática ($x = \sqrt[100]{2\theta/\pi}$) para encontrar os parâmetros de decisão. 
*   **Consequência**: Garantimos uma densidade de pontos visualmente íntegra e estatisticamente justa, eliminando os "pontos flutuantes" isolados que distorciam auditorias de performance.

### 3. Protocolo de Confronto Legado (VBar-Audit)
Estabelecemos a auditoria contra o `legacy2_optimal` como uma barreira de validação científica.
*   **Decisão**: Toda mudança no motor matemático de um MOP deve ser validada por um "Confronto Triplo" (v0.7.5 vs Legado1 vs Legado2) via `topo_shape`.
*   **Rigor**: Invariantes de paridade ($\sum f = c$ ou $\sum f^2 = c$) são verificados em cada auditoria para garantir que a precisão numérica mantenha um erro residual inferior a $10^{-8}$.

## Interpretacão Narrativa e Consequências
Esta mudança marca a transição do MoeaBench de um "Framework de Otimização" para um "Instrumento de Metrologia Científica". 

*   **Rigor Superior**: A v0.7.6 agora impõe uma verdade matemática que é, em muitos casos, superior aos dados produzidos por algoritmos de estado da arte rodando por milhares de gerações.
*   **Estabilidade de Métricas**: Métricas de convergência agora operam sobre alvos de densidade uniforme, reduzindo variações espúrias nos resultados de IGD e HV causadas por amostragens pobres da frente ótima.
*   **Auditabilidade**: A substituição de arquivos CSV "congelados" por solvers analíticos aumenta a transparência do framework, permitindo que qualquer pesquisador reproduza a frente ideal sem depender de sementes aleatórias ou dados externos.

---
*Documentado em 2026-01-25 para refletir a consolidação da Fase Z/Z2.*
