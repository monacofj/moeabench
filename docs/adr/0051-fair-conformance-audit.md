<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# ADR 0051: FAIR Conformance Audit, Scientific Oracles, and Validation Log

**Status:** Proposed — investigation in progress  
**Date:** 2026-10-21  
**Issue:** [#75](https://github.com/monacofj/moeabench/issues/75)  
**Branch:** `fix/75-fair-conformance`  
**Normative references:** monograph; `docs/fair_metrics.md`; ADRs 0026, 0037, 0048, 0049 (and their supersession relationships).

## Purpose and working agreement

This is both an architectural decision record and an **append-only investigation log**. The original explanation and plan in Section 1 are transcribed **ipsis litteris** from the discussion; subsequent evidence, failed attempts, fixes, numerical results, test executions, and unresolved issues belong in Section 2. Do not rewrite the original explanation to make new observations seem pre-existing. Do not change scientific definitions just to make current regression targets pass. Fixes remain proposed until independently tested.

## 1. Original explanation and plan — literal transcript (2026-10-21)

Eu resolveria a [issue #75](https://github.com/monacofj/moeabench/issues/75) como uma **auditoria de conformidade científica**, não como uma simples correção do `example_13.py`.

O objetivo é demonstrar que cada métrica FAIR implementa sua definição matemática e que os Q-scores resultantes têm a interpretação pretendida. Para isso, precisamos de testes que não dependam da própria implementação que estamos tentando validar.

Já examinei os principais módulos no `main` do MoeaBench e encontrei evidências concretas de problemas.

### 1. Correções necessárias

**Primeiro: seleção do Ground Truth.**

Em `diagnostics/auditor.py`, o código recebe o GT explícito, mas posteriormente consulta `_gt_registry` e pode substituí-lo por um GT registrado. Isso explica a discrepância entre os 500 pontos fornecidos e os 10.000 utilizados.

A correção deverá garantir que um GT explícito tenha precedência, sem impedir o uso automático de referências quando o usuário não fornece uma.

**Segundo: cálculo de \(s_K\).**

Em `diagnostics/utils.py`, a resolução assume `1.0` quando o MOP não fornece `s_fit` ou `s_k`:

```python
res = getattr(mop, 's_fit', getattr(mop, 's_k', 1.0))
```

Entretanto, a documentação define:

\[
s_K=\operatorname{median}_{u\in U_K}
\min_{\substack{v\in U_K\\v\ne u}}\|u-v\|_2,
\qquad U_K=\operatorname{FPS}(GT,K).
\]

O próprio MoeaBench já possui `get_resolution_factor_k()` para calcular essa grandeza. Precisamos utilizá-lo corretamente, preservando a possibilidade de o usuário fornecer explicitamente `s_k`.

**Terceiro: `ideal_res` e Closeness.**

A implementação de `fair.closeness()` começa assumindo `ideal_res = 0.0`, tenta recuperar uma calibração e, caso ocorra alguma exceção, simplesmente mantém zero.

Isso pode esconder uma falha de calibração e produzir um resultado numérico aparentemente legítimo.

Precisamos estabelecer quando zero é matematicamente válido, quando o resíduo deve ser recuperado da calibração e quando sua ausência deve produzir um erro explícito.

**Quarto: coerência das referências e dos Q-scores.**

É necessário verificar se as métricas, suas referências ideais, os baselines aleatórios e as distribuições de calibração foram construídos nas mesmas coordenadas e sob o mesmo protocolo.

Aqui eu não corrigiria Q-scores isoladamente. Corrigiria primeiro a geometria e a calibração que os alimentam.

Há ainda um achado importante na leitura do código: em `diagnostics/calibration.py`, a calibração de `ideal_res` usa um segundo FPS extraído do próprio GT e mede a distância desses pontos ao mesmo GT.

Como os pontos selecionados pertencem ao conjunto original, essas distâncias são **necessariamente zero**. Portanto, essa construção não mede o resíduo de discretização descrito no ADR 0037.

Além disso, `fair.closeness()` tenta chamar `base.get_baseline_data()`, mas não encontrei essa função no módulo `baselines.py`. A exceção é silenciada pelo `except`, o que permite que `ideal_res` permaneça zero sem qualquer aviso.

Isso nos dá um defeito concreto, com uma explicação matemática, e não apenas uma suspeita derivada de um resultado estranho.

### 2. Como eu testaria a correção

Usaria três níveis complementares.

#### A. Testes matemáticos independentes

Criaria pequenos conjuntos sintéticos para os quais conhecemos antecipadamente o resultado, calculado por fórmulas independentes do MoeaBench.

_Nota de transcrição: a resposta original incluía neste ponto uma ilustração interativa das frentes (GT, soluções sobre a frente e soluções deslocadas), que não é prosa Markdown e não foi incluída no ADR._

Esquema dos testes de proximidade: a distância geométrica deve ser previsível e aumentar quando deslocamos a população da frente.

Por exemplo, para uma frente linear e uma população deslocada perpendicularmente por uma distância conhecida \(d\), podemos verificar diretamente as distâncias, o cálculo de \(s_K\) e o Closeness resultante.

Também construiria uma frente analítica contínua e um GT discretizado, gerando pontos ideais independentes dos pontos presentes no GT. Assim verificamos se `ideal_res` representa efetivamente o erro de discretização.

Esses testes permitiriam detectar uma implementação errada mesmo que ela produzisse sempre o mesmo resultado.

#### B. Testes de propriedades científicas

Além de resultados numéricos exatos, verificaria propriedades esperadas das métricas.

| Experimento controlado | Expectativa |
|---|---|
| População ideal versus população afastada | Closeness deve piorar com o afastamento |
| Frente completa versus frente truncada | Coverage deve indicar perda de cobertura |
| Frente uniforme versus frente com lacunas | Gap deve detectar a fragmentação |
| Distribuição uniforme versus aglomerada | Regularity e Balance devem responder às respectivas irregularidades |
| Repetir o mesmo experimento com a mesma configuração | Resultados numericamente reproduzíveis |
| Mudar coerentemente a escala de população, GT e calibração | Closeness em unidades de \(s_K\) deve ser preservado |
| Fornecer um GT explícito diferente do registrado | O GT fornecido deve ser respeitado |

Eu tomaria cuidado com a afirmação de que toda métrica deve melhorar monotonicamente quando a frente melhora. Isso não é verdade em geral: uma frente pode melhorar a convergência e piorar a distribuição. Por isso, os testes devem isolar uma propriedade de cada vez.

#### C. Testes dos Q-scores e das calibrações

Aqui usaria populações representando situações conhecidas: uma aproximação ideal, uma aproximação próxima da frente, uma população moderadamente deslocada e uma população gerada pelo modelo nulo da calibração.

Verificaria não apenas se os Q-scores permanecem entre zero e um, mas se respondem coerentemente às mudanças controladas e se suas referências favoráveis e desfavoráveis são efetivamente as especificadas.

Os valores esperados seriam calculados por um pequeno oráculo de teste independente, sem chamar as funções do MoeaBench para gerar a resposta esperada.

### 3. Sequência de implementação

Eu trabalharia numa branch `fix/75-fair-conformance`, em quatro etapas, cada uma acompanhada dos respectivos testes.

**Etapa 1 — Estabelecer as especificações normativas**

Confrontar monografia, `docs/fair_metrics.md` e ADRs vigentes. Produzir uma tabela com a definição matemática, os parâmetros, as unidades e as referências de cada métrica. Havendo contradições entre documentos, resolvê-las explicitamente antes de programar.

**Etapa 2 — Corrigir a infraestrutura compartilhada**

Resolver a precedência do GT, o cálculo de \(s_K\), o sistema de coordenadas e a recuperação das calibrações. Verificar também a compatibilidade entre um GT explicitamente fornecido e os baselines pré-computados. Uma referência diferente pode exigir recalibração; não devemos misturar referências incompatíveis.

**Etapa 3 — Validar individualmente as métricas**

Corrigir cada métrica e testar os valores físicos e os Q-scores separadamente. Não alterar as fórmulas para fazer os testes passarem; usar resultados analíticos e casos controlados como critérios de correção.

**Etapa 4 — Validar a integração completa**

Rodar o `example_13.py` e os demais exemplos FAIR, comparar os resultados, verificar os relatórios e executar a bateria de regressão do MoeaBench.

### 4. Como garantir que não estragamos o que já funciona?

Esse é um cuidado especialmente importante, porque alterar uma referência pode afetar várias métricas simultaneamente.

Eu adotaria três regras.

Primeiro, **preservar tudo que for comprovadamente correto**. Headway, por exemplo, deve continuar passando pelos testes existentes e por testes independentes baseados no ADR 0049.

Segundo, não utilizar os números atualmente produzidos como *ground truth* dos novos testes. Testes de regressão numérica são úteis, mas congelar resultados de uma implementação potencialmente errada apenas perpetuaria o defeito.

Terceiro, separar testes rápidos de testes dispendiosos. Os testes matemáticos e de integração devem rodar normalmente no CI. A regeneração completa das calibrações, que pode ser demorada, deve ocorrer apenas se necessária, em uma bateria separada. Isso também permitirá avaliar se os baselines distribuídos precisam ser versionados novamente.

Há uma limitação importante na suíte atual: `tests/integration/test_example13_consistency.py` verifica que auditoria e visualização recebem o mesmo experimento e a mesma referência, mas utiliza mocks. Portanto, **esse teste não detectaria o GT sendo substituído posteriormente dentro da auditoria**. Precisamos acrescentar um teste numérico de ponta a ponta.

### 5. Critérios para considerar a issue resolvida

Eu só consideraria a #75 pronta para merge quando tivermos demonstrado que:

- As definições matemáticas das seis métricas FAIR estão implementadas corretamente.
- A escolha do GT, \(K\), \(s_K\) e das calibrações é consistente, reproduzível e verificável.
- Os Q-scores comportam-se como esperado nos casos controlados.
- O Example 13 apresenta resultados explicáveis, inclusive quando algum Q-score é baixo.
- A suíte existente passa sem regressões indevidas.
- A documentação descreve exatamente o comportamento verificado.

**Minha recomendação é não começar pelos Q-scores.** O erro mais perigoso parece estar na infraestrutura compartilhada de referências e calibração. Corrigi-la primeiro pode resolver várias discrepâncias simultaneamente e revelar quais problemas realmente pertencem às fórmulas individuais.

Ao final, teremos algo mais significativo do que um Example 13 funcionando: teremos evidências verificáveis de que a implementação FAIR corresponde à metodologia científica que pretendemos publicar.

## 2. Investigation and validation log — append new entries here

### 2026-10-21 — Start, confirmed by code inspection (no test execution yet)

- **SUCCESS — Work isolated:** created branch `fix/75-fair-conformance` from `main` for issue #75.
- **CONFIRMED CODE PATH — Explicit GT precedence:** `moeabench/diagnostics/utils.py::_resolve_diagnostic_context` initially takes `ref`; `moeabench/diagnostics/auditor.py::audit` subsequently overwrites `GT` whenever an appropriately dimensioned `_gt_registry` entry exists. This is a code-level proof of precedence loss, not yet an executed reproduction of the exact DTLZ1 run.
- **CONFIRMED CODE PATH — s_K:** `_resolve_diagnostic_context` supplies `s_k=1.0` as a default. `audit()` only calculates `get_resolution_factor_k(GT, K_target)` when `s_k_mop <= 1e-12`. Therefore, the `1.0` placeholder is treated as genuine calibration; this conflicts with the documented FPS-based scale.
- **CONFIRMED MATHEMATICAL ERROR — ideal_res calibration:** `calibration._generate_baselines` sets `u_ideal_sample=get_ref_uk(gt,k,seed=999)`, a subset of `gt`, and then computes each sample point's distance to `gt`. All distances must equal zero (apart from numerical roundoff), so this cannot estimate off-sample finite-GT residue promised by ADR 0037.
- **CONFIRMED CODE PATH — swallowed calibration failure:** `fair.closeness` calls `base.get_baseline_data(...)` under `except: pass`; that helper is absent from the inspected current `baselines.py` module. Result: `ideal_res=0` without a visible error.
- **DISCOVERED LIMITATION — existing test:** `tests/integration/test_example13_consistency.py` mocks the audit and checks only that calls share `gt`, not whether the audit actually uses it internally.
- **OPEN — Coordinate mismatch:** reported in the original investigation but the complete input-to-baseline scale provenance still requires a numerical reproduction before classifying particular code paths as broken.
- **OPEN — Normative ambiguity:** ADR 0037 says independent fresh ideal samples; `docs/fair_metrics.md` also describes FPS draws from a finite GT. Their intended relationship must be resolved before changing formulas or regenerating frozen baselines.
- **NOT DONE:** no implementation correction, no regression numbers, no full Example 13 run, and no CI result claimed at this stage.

### Rules for subsequent log entries

Record each attempt with date, exact command or code entry point, environment, expected vs observed values, outcome (**PASS**, **FAIL**, **BLOCKED**, **INCONCLUSIVE**), explanation, and commit/PR link if any. Retain failed attempts as evidence. Update this ADR on the branch after each meaningful investigation or modification. Mark the ADR **Accepted** only after issue acceptance criteria and regressions are met.


### 2026-10-21 — Reproduction probes committed and independent numerical check

- **SUCCESS — Test cases added:** `tests/unit/test_fair_conformance_issue75.py` introduces an independent FPS-resolution oracle, strict expected failures (XFAIL) for the implicit `s_K=1.0` fallback and GT-registry override, and a positive mathematical demonstration of the zero-by-construction subset residue. XFAIL indicates a known defect, **not** validation of correct behavior. The tests have been committed but have **not been executed against an installed MoeaBench package**.
- **PASS — Independent numerical check (outside MoeaBench):** Python/NumPy/SciPy direct calculation on `GT = [(x, 1-x) for x in np.linspace(0, 1, 41)]`, seed-zero FPS, `K=11`: median nearest-neighbor spacing `s_K=0.1767766952966369`, distinct from the currently configured implicit fallback `1.0`.
- **PASS — Independent numerical check:** For `K=10`, seed-999 FPS subset of the *same* `GT`, the median distance to `GT` is exactly `0.0`. On independent midpoints of this same analytic line, the median nearest-GT distance is `0.017677669529663664`. This confirms the claimed mechanism cannot measure an off-sample residue.
- **LIMITATION — Not a full application reproduction:** these numerical checks ran as small stand-alone calculations; they are not equivalent to running `example_13.py` nor the MoeaBench pytest suite. No real DTLZ1 Q-score has been recalculated yet.
- **ENVIRONMENT:** local isolated Python had NumPy `2.3.5`, SciPy `1.17.0`, pytest `9.0.2`; pymoo is not installed and the repository could not be cloned in that runtime (network/DNS unavailable). Current testing therefore remains **BLOCKED** until an environment containing MoeaBench and its dependencies is available. Do not report the XFAIL tests as executed.

