<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Índice da documentação do MoeaBench

Este índice apresenta a função dos principais documentos do projeto e oferece um percurso de leitura. Para compreender uma funcionalidade, comece pelo documento técnico ou guia correspondente; para conhecer **por que** uma arquitetura foi escolhida, consulte as decisões (ADRs); para conferir dados, provas, hipóteses e resultados de uma investigação, consulte os relatórios especializados.

A documentação **de referência** explica conceitos e contratos e deve permanecer útil independentemente do andamento de uma issue. Os **relatórios de pesquisa** preservam evidências e reproduções de experimentos. As **ADRs** registram decisões, seu status e suas eventuais substituições; não se deve tratar uma proposta como decisão aprovada.

## Uso e referência geral

| Documento | O que contém | Quando consultar |
|---|---|---|
| [Guia do usuário](userguide.md) | Instalação, primeiros experimentos e utilização das operações do MoeaBench | Para aprender a usar o software |
| [Referência da API](reference.md) | Interfaces, classes, métodos e parâmetros documentados | Para consultar contratos de programação |
| [Resumo da API](api_sheet.md) | Mapa conciso de operações e fluxos usuais | Para localizar rapidamente uma chamada |
| [Ground Truth](gt.md) | Fundamentos de PS/PF/GT, responsabilidade de cada MOP, geração de referências, matemática do DPF3/DPF5, qualidade de amostragem, precisão, indicadores e preservação histórica | Para compreender como definir e avaliar cientificamente uma referência |
| [Contribuição](CONTRIBUTING.md) | Orientações para alterações e colaboração | Para desenvolver ou revisar contribuições |
| [Índice das ADRs](adr/README.md) | Todas as decisões arquiteturais, status e relações de substituição | Para identificar normas vigentes antes de alterar contratos |

A [página inicial do repositório](../README.md) apresenta a visão geral do MoeaBench, exemplos, citação e referências bibliográficas. O presente **index.md** é um índice de documentação, não uma lista bibliográfica e não substitui o README da raiz.

## Ground Truth: da teoria à evidência reproduzível

O documento principal é **[gt.md](gt.md)**, que contém a explicação permanente e completa dos conceitos e das conclusões científicas. Os documentos abaixo são **relatórios especializados da investigação [#119](https://github.com/monacofj/moeabench/issues/119), reunida na [PR #139](https://github.com/monacofj/moeabench/pull/139)**. Eles fornecem provas detalhadas, condições experimentais, números, resultados negativos, comparação com a implementação auditada e rastreabilidade.

### Fundamentos matemáticos e provas

| Documento | Conteúdo e contribuição |
|---|---|
| [Prova global da PS/PF do DPF3](gt-dpf3-global-pareto-proof-issue119.md) | Limiares caóticos ordenados, conservação da ordem das projeções e demonstração de que a PS publicada é exatamente o conjunto onde g=0 |
| [Prova global da PS/PF do DPF5](gt-dpf5-global-pareto-proof-issue119.md) | Parametrizações hiperesféricas dos ramos P e D, prova de pertinência e completude da frente publicada e análise da dominância entre componentes |
| [Cobertura angular do DPF3 com D=2](gt-dpf3-d2-angular-cover-proof-issue119.md) | Construção finita por ângulos, limite de Hausdorff em aritmética exata e distinção da fidelidade em float64 |
| [Cobertura angular do DPF3 com dimensão geral](gt-dpf3-nd-cover-and-acceptance-issue119.md) | Extensão do limite matemático para D maior, singularidades de cardinalidade, custo tensorial, estimativas de arredondamento e critérios científicos |

### Auditoria das equações e precisão numérica

| Documento | Conteúdo e contribuição |
|---|---|
| [Diagnóstico inicial de PS/PF](gt-dpf3-dpf5-audit-issue119.md) | Contagens pedidas e devolvidas, duplicatas e dominância interna observada nos avaliadores auditados |
| [Confronto com as fórmulas publicadas](gt-dpf-published-formula-triage-issue119.md) | Primeira identificação da ordenação dos limiares DPF3 e da diferença de condição na fronteira dos ramos DPF5 |
| [Reconstrução DPF5 para D geral](gt-dpf-general-d-evidence-issue119.md) | Equações independentes dos ramos beta/P e gamma/D em dimensões essenciais superiores, com g=0 e g>0 |
| [Núcleos numéricos DPF3/DPF5](gt-dpf-numeric-kernel-evidence-issue119.md) | Cancelamento em 1−cos, identidades trigonométricas estáveis, invariantes esféricos e discrepâncias entre funções de avaliação |
| [Varredura em alta precisão](gt-dpf-high-precision-sweep-issue119.md) | Valores de referência em precisão elevada, underflow, limites de representabilidade e verificações por dimensão |
| [Verificação fora da PF do DPF5](gt-dpf5-offpareto-issue119.md) | Comparações com g>0, equações escaladas e isolamento de dependências opcionais nos testes de pesquisa |

### Discretização, comparação de amostradores e impacto nos indicadores

| Documento | Conteúdo e contribuição |
|---|---|
| [Cobertura finita do DPF5 por ramo](gt-dpf5-finite-sampler-coverage-issue119.md) | Medidas independentes da cobertura de P e D, cardinalidade efetiva e diferença entre a fronteira implementada e a publicada |
| [Amostragem Sobol com orçamento efetivo igual](gt-dpf5-branch-sobol-comparison-issue119.md) | Experimento controlado por ramo; melhoras e pioras conforme orçamento e geometria |
| [Alocação do orçamento entre P e D](gt-dpf5-branch-allocation-issue119.md) | Estudo de 50%, aproximadamente 67% e distribuição existente, com múltiplas sementes e orçamento total efetivo constante |
| [Efeito da referência nos indicadores](gt-dpf5-paired-indicator-impact-issue119.md) | Isolamento dos efeitos de correção do avaliador e substituição do GT sobre IGD, IGD+ e HV relativo, mantendo populações fixas |

### Integração, governança e rastreabilidade

| Documento | Conteúdo e contribuição |
|---|---|
| [Inventário de impacto de produção](gt-dpf-production-impact-inventory-issue119.md) | Componentes, arquivos históricos, referências, consumidores e testes potencialmente afetados por mudanças matemáticas |
| [Proposta de correção e migração](gt-dpf-correction-migration-proposal-issue119.md) | Sequenciamento e requisitos de aprovação para correções no avaliador, amostradores, referências versionadas e orçamento de calibração |
| [Revisão pré-merge da PR #139](gt-dpf-premerge-review-issue119.md) | Escopo da PR de pesquisa, salvaguardas, comparação de alterações e verificações de CI realizadas |

Os relatórios históricos podem conter enunciados como “a pertinência global ainda não foi demonstrada” em um estágio inicial. **O resultado matemático consolidado está em [gt.md](gt.md) e nas provas globais posteriores.** A documentação histórica deve ser lida com essa distinção; não é apropriado interpretar observações provisórias como conclusões finais.

## Decisões arquiteturais relevantes para GT

O [índice completo de ADRs](adr/README.md) inclui os registros aceitos, propostos, substituídos e parcialmente substituídos. Para alterações relacionadas a Ground Truth, os pontos de entrada mais diretamente relevantes são:

| ADR | Tema | Relação com GT |
|---|---|---|
| [0006](adr/0006-benchmarks-reimplementation.md) | Reimplementação dos benchmarks | Avaliação e PS/PF analíticos como responsabilidade do MOP |
| [0016](adr/0016-evolution-of-scientific-validation-and-guided-geometric-solvers.md) | Soluções geométricas orientadas por conhecimento | Evolução da construção analítica de referências |
| [0017](adr/0017-comprehensive-calibration-and-testing-framework.md) | Calibração e testes científicos | Referências analíticas e papel dos baselines |
| [0018](adr/0018-consolidated-scientific-report-and-gt-density-refinement.md) | Densidade histórica de GT | Orçamentos e decisões históricas, com a revisão parcial descrita na 0057 |
| [0021](adr/0021-numerical-integrity-and-validation-pipeline-float64-policy.md) | Integridade numérica | Política de representação e validação em float64 |
| [0056](adr/0056-versioned-certification-and-performance-reference-lifecycle.md) | Certificações versionadas | Preservação de referências históricas e separação de certificados |
| [0057](adr/0057-proposed-purpose-specific-gt-budgets.md) | Orçamentos de GT por finalidade | Geração por MOP e calibração sob demanda, com detalhes numéricos distintos das decisões em princípio |

A [ADR 0053](adr/0053-dpf5-mixed-dimensional-reference-measure.md) documenta **pesquisa proposta/não certificada** relacionada a medidas sobre componentes de dimensões diferentes e a estudos FAIR históricos. Não a confundir com uma política de GT aprovada para produção.

## Como usar este índice

Para **compreender** a ciência do Ground Truth, leia [gt.md](gt.md). Para **implementar** métodos ou alterar APIs, compare os contratos das [ADRs](adr/README.md) com a documentação de [API](reference.md). Para **conferir uma demonstração ou reproduzir números**, vá ao relatório especializado correspondente e aos scripts/testes da PR #139. Para **entender por que uma decisão de arquitetura foi tomada**, consulte o histórico e o status da ADR, incluindo os adendos mais recentes.

O índice organiza o conhecimento; não confere status de decisão aprovada a experimentos e não transforma registros históricos em contratos operacionais.
