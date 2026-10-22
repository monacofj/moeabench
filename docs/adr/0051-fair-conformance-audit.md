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



### 2026-10-21 — Normative cross-check against the dissertation

- **SUCCESS — Normative source recovered:** inspected `monografia-nova.pdf`, Chapter 6, PDF pages 114–118, directly rather than relying only on existing ADRs or on earlier discussion. The dissertation makes the physical definitions explicit (equations 6.1–6.8) and gives the Q-score ECDF formula (6.9).
- **CONFIRMED — Mathematical contracts:** the dissertation's `s_K` is the median nearest-neighbor distance in an FPS subset `U_K`; Closeness uses median `P→GT` divided by `s_K` minus `epsilon_ideal`; Coverage mean `GT→P`; Gap 95th percentile `GT→P`; Regularity Wasserstein-1 of NN-spacing distributions; Balance **square root** of Jensen–Shannon divergence of regional occupancy histograms; Headway ratio of final and initial GD95. In the inspected code, the `jensenshannon(...,base=2)` implementation for Balance is therefore **correct**; do not replace it with unrooted JS divergence.
- **IMPORTANT QUALIFICATION — epsilon_ideal:** the dissertation explicitly allows `epsilon_ideal=0` in a regime where the discretization residue is zero. Hence zero by itself does not prove a bug. The **defect is the purported estimator**, which is structurally forced to zero even in settings for which an independent ideal reference would have nonzero distance. Any fix must distinguish legitimate exact-zero residual from a missing/invalid estimate.
- **DISCOVERED SPECIFICATION TENSION:** ADR 0037 describes a fresh analytical ideal front; `docs/fair_metrics.md` describes ideal draws from FPS subsets of the same finite GT. Both cannot independently support a positive off-sample residue. Before altering calibration, decide which definition governs and document the method for independently sampling ideal points.
- **CONFIRMED — Q-score definitions in dissertation:** Coverage, Gap, Regularity, Balance use a calibrated ECDF mapping between favorable and unfavorable references; Closeness uses Wasserstein against a half-normal perturbation baseline; Headway uses a log-linear mapping. These mappings will be tested separately, not assumed interchangeable.
- **BLOCKED — Automated suite:** no GitHub commit statuses were reported for the draft PR head at inspection time, and the current configured GitHub workflow is specific to branch `issue-67-environment-verification` or manual dispatch. This is not a successful CI run.

### Provisional normative conformance matrix (2026-10-21)

| Physical metric | Dissertation definition | Units/reference | Current status after source audit |
|---|---|---|---|
| Closeness | `max(0, median(P→GT)/s_K − epsilon_ideal)` | FPS resolution units; independent finite-GT residue | **Broken reference/resolution plumbing; ideal residue estimator invalid for off-sample residue** |
| Coverage | `mean(GT→P)` | Objective-space Euclidean distance | Formula matches code; common GT/scale still needs verification |
| Gap | `percentile95(GT→P)` | Objective-space Euclidean distance | Formula matches code; common GT/scale still needs verification |
| Regularity | `W1(NN(P),NN(U_K))` | Wasserstein distance over objective-space NN separations | Formula matches code; `U_K`/GT provenance needs verification |
| Balance | `sqrt(JS(H_P,H_UK))` | Dimensionless distance of regional occupancy histograms | SciPy `jensenshannon` formula matches thesis; reference centroids/histogram provenance needs verification |
| Headway | `GD95(P_final→GT)/GD95(P_initial→GT)` | Dimensionless longitudinal ratio | Formula matches ADR 0049; must retest under corrected GT selection and preserve undefined-initial handling |

A formula-level match is **not** an end-to-end correctness certification. The physical reference and coordinate system can invalidate a correctly implemented formula.

### Next execution gate

On an installed project environment, run:

```bash
python -m pytest tests/unit/test_fair_conformance_issue75.py -rx -q
python -m pytest tests/integration/test_example13_consistency.py -q
```

Record actual pass/XFAIL/XPASS/failure output here. Only then begin narrow production fixes. Regenerate frozen baseline artifacts **only if** required by a documented reference/calibration change, and separately verify that the new values follow an independent oracle.


### 2026-10-21 — Targeted CI initiated

- **SUCCESS — Draft review opened:** [PR #76](https://github.com/monacofj/moeabench/pull/76) tracks the investigation, tests, and this ADR without proposing an unvalidated production patch.
- **SUCCESS — CI workflow created:** added temporary branch-only `.github/workflows/fair-conformance-investigation.yml` to execute the issue-specific unit probes and the existing Example 13 integration contract against an installed MoeaBench package, using Python 3.14.4. **This temporary workflow must be removed from the final PR** after tests are integrated in the ordinary test suite.
- **IN PROGRESS — First remote CI run:** [Actions run 39656387447](https://github.com/monacofj/moeabench/actions/runs/39656387447) was observed with state `in_progress` (not yet a PASS or FAIL). An open PR is not itself evidence of numerical correctness.


### 2026-10-21 — First real CI reproduction (confirmed)

- **PASS — GitHub Actions environment:** [run 39656387447](https://github.com/monacofj/moeabench/actions/runs/39656387447) completed with conclusion `success`; installed MoeaBench `0.16.1` and dependencies in Python `3.14.4`.
- **PASS — Mathematical oracle tests:** `2 passed` (independent non-unit FPS resolution and proof that a same-GT FPS subset has zero distance whereas analytical midpoints do not).
- **EXPECTED FAIL — Two implementation bugs actually reproduced:** `2 xfailed`: `test_resolver_computes_documented_resolution_from_explicit_gt` (`s_K=1.0` fallback) and `test_audit_preserves_explicit_gt_when_registry_has_another_reference` (registry overrides explicit GT). These are not successes of the product: strict XFAIL records real, previously known defects without destabilizing the investigation branch.
- **PASS — Existing Example 13 interface regression:** `tests/integration/test_example13_consistency.py`: `1 passed`. This only tests the call contract with mocks, not end-to-end Q-score validity.
- **PASS — Stronger XFAIL guards:** the two expected-failure tests now specify `raises=AssertionError`; a runtime exception unrelated to the asserted contract will be a test **failure**, not an accepted XFAIL. [run 39656554029](https://github.com/monacofj/moeabench/actions/runs/39656554029) completed `success` on this improved version: `2 passed, 2 xfailed`, plus `1 passed` in the Example 13 interface test.
- **NOT YET VERIFIED — Complete scientific correctness:** neither CI run tested an entire DTLZ1 optimization or recalculated the final Q-scores. Do not treat successful execution of a suite containing XFAILs as a product conformance certification.
- **INFRASTRUCTURE NOTE:** the temporary branch workflow was restricted to changes in the FAIR implementation/tests/workflow itself, so routine ADR diary updates need not start additional runs.


### 2026-10-21 — Audit-report impact registered before production changes

- **USER REQUIREMENT — Potential artifact regeneration:** the FAIR fixes may require regenerating the canonical `calibration/audit_report.html`; this dependency was added to issue #75 as a distinct validation gate.
- **CONFIRMED ARTIFACT DEPENDENCY:** `calibration/audit_report.html` currently points to `reports/audit_report_v0.14.0.html`. The relevant report inputs are versioned `calibration/reports/audit_v*.json` files. `calibration/scripts/generate_visual_report.py` resolves the input JSON and produces the versioned HTML; `calibration/scripts/audit_calibration.py` produces numerical audit results. `docs/CONTRIBUTING.md` explicitly instructs regeneration of baseline/report artifacts when scientific indicators are affected.
- **DECISION — Not regenerated yet:** preserve prior audit versions. Once the definitions and baselines are validated, reproduce current artifacts, identify changed numerical outputs and provenance, and generate a new versioned report only if warranted. Do not overwrite the historical report or blindly regenerate 10k GT baselines.
- **CRITICAL OPEN QUESTION:** an explicit GT with 500 points and a registered baseline built on a different 10,000-point and possibly differently normalized GT cannot safely be assigned the latter's Q-score baseline without a compatibility check or recalibration. The first plumbing correction will restore the explicit GT; the science of baseline compatibility remains a separate required verification before declaring Q-scores correct.


### 2026-10-21 — First production corrections and intermediate CI failures

- **CHANGE — GT precedence:** `moeabench/diagnostics/auditor.py` no longer substitutes a registered calibration GT when `ground_truth` is supplied explicitly. Inferred/default GT may still be replaced by the registered GT so default audits use their canonical calibration reference. The effective `gt_source` is recorded in `diagnostic_context`.
- **CHANGE — FPS resolution:** `moeabench/diagnostics/utils.py` now derives absent `s_k` from `get_resolution_factor_k` on the selected GT and baseline K policy rather than defaulting to `1.0`; the aggregate audit recomputes `s_k` after final GT selection. Explicit positive `s_k` remains authoritative.
- **EXPECTED DEVELOPMENT FAILURE — CI runs [39658692509](https://github.com/monacofj/moeabench/actions/runs/39658692509) and [39658769457](https://github.com/monacofj/moeabench/actions/runs/39658769457):** the two earlier tests were still marked `xfail(strict=True)` and **XPASS(strict)** after the code defects were corrected, so those workflows correctly failed (`2 failed, 2 passed`). This is evidence of changed behavior, but not evidence of a completed regression pass. The XFAIL decorators were removed in the subsequent test commit.
- **TEST EXPANSION — Explicit API overrides:** added unit assertions for explicit `s_k` and for observing the effective GT/source/scale in FAIR-only audit metadata.
- **VERIFICATION IN PROGRESS:** a new branch CI build includes the corrected tests and runs the full unit and basic integration suites. Scientific Q-score/GT-baseline compatibility is **still open** and is not asserted to be solved by these two plumbing changes.


### 2026-10-21 — Regression suite run and CI fixture failure

- **PASS — Corrected conformance tests:** [Actions run 39658837761](https://github.com/monacofj/moeabench/actions/runs/39658837761) passed once the obsolete XFAIL decorators had been removed and explicit overrides tested.
- **PASS — Unit regression:** [Actions run 39658876164](https://github.com/monacofj/moeabench/actions/runs/39658876164) passed `276/276` unit tests after the GT/s_K plumbing changes.
- **FAIL — Integration suite environment:** the same run failed with `8 passed, 15 errors`. All 15 errors came from a missing required integration fixture `examples/example_full_exp1.zip`, NOT from a test assertion about FAIR metrics. This is an environment/preparation failure and does not certify integration success or a product regression.
- **CI FIX — Restored fixture preparation:** the branch's temporary GitHub Actions workflow now generates `example_full_exp1.zip` and `example_full_exp2.zip` using short DTLZ2 NSGA-II/NSGA-III runs, following the established repository workflow `.github/workflows/environment-verification.yml`, before executing the basic integration suite.
- **STILL OPEN — End-to-end Q-score baseline compatibility and full Example 13:** numerical validity is not yet certified; GT precedence and scale are only the first layer of the audit.


### 2026-10-21 — Integration recovered; Q-score reference compatibility gate

- **PASS — All unit and basic integration regressions after fixture fix:** GitHub Actions [run 39659239196](https://github.com/monacofj/moeabench/actions/runs/39659239196) completed successfully: issue probes `6 passed`, Example 13 mocked interface test `1 passed`, full unit suite `276/276 passed`, and basic integration suite `23/23 passed`. The preceding `15 errors` were therefore resolved by restoring missing CI fixtures; no mathematical change was made to resolve them.
- **SCIENTIFIC CORRECTNESS FINDING — Q baseline provenance:** restoring explicit GT precedence exposed a deeper incompatibility: the provided 500-point analytical reference and the frozen 10k-point registered calibration GT are different physical references. Blindly combining physical measurements from one with quality anchors calibrated on the other can produce scientifically misleading Q-scores.
- **CHANGE — Fail-closed clinical interpretation:** `audit()` now compares a selected GT with the registered GT (when such provenance exists) and records `baseline_gt_compatible`. For `quality=True` and a mismatched reference, the FAIR physical metrics remain available, but no Q-scores are fabricated and status is `MISSING_BASELINE`, with explanatory guidance. For `quality=False`, the caller can still deliberately measure against any explicit GT.
- **CHANGE — Example 13 source of truth:** the example now requests `mb.clinic.audit(exp)` with its registered calibration reference and passes `audit.diagnostic_context["GT"]` to `mb.view.topology`, rather than generating another, possibly incompatible `exp.optimal_front()` and passing that into the clinical audit. The interface regression was changed to enforce the same GT object for audit and visualization. A dedicated conformance unit test checks mismatch rejection and preservation of physical measurements.
- **LIMITATION:** these new changes require CI validation (at the time of this note), and the underlying `ideal_res` methodology and calibration artifacts are not yet repaired. The current Q-scores must **not** be declared scientifically certified solely because they are within `[0,1]`.


### 2026-10-21 — Full DTLZ1 Example 13 pipeline exposed a fundamental scale mismatch

- **PASS — Real algorithm execution and numeric capture:** [GitHub Actions run 39659720148](https://github.com/monacofj/moeabench/actions/runs/39659720148) completed successfully with the Example 13 parameters: DTLZ1 (M=3), NSGA-II population 100, 280 generations, seed 1; this run additionally passed `7` issue-specific unit tests, `1` mocked interface test, `277/277` unit and `23/23` basic integration tests.
- **OBSERVED — Fundamental numerical inconsistency:** `P.shape=(100,3)`, registered `GT.shape=(10000,3)`; `median(sum(P))=0.5452085204545316` while `median(sum(GT))=1.004812790875524`. The computed `s_K=0.08847951537468546` at effective `K=100` matched an independent call to the FPS routine, but the coordinate spaces appear **incompatible**.
- **OBSERVED — Q-score collapse:** `Q_CLOSENESS=0`, `Q_COVERAGE=0`, `Q_GAP=0`, `Q_REGULARITY=0`, `Q_BALANCE=0`, `Q_HEADWAY=0.9986921404471868`; physical `CLOSENESS=3.0195288163289904`. These are **recorded symptoms, not reference targets for tests**.
- **ROOT-CAUSE LEAD:** `moeabench/diagnostics/clinic/generate_calibration.py` and `generate_references.py` normalize the GT with `(gt - ideal)/(nadir - ideal)`; `calibration/scripts/generate_baselines.py` consumes `calibration_package.npz['gt_norm']`. The clinical `audit()` currently compares objective values of the actual optimizer output to the registered **normalized GT** without applying the matching transform to the population. This precisely explains why a roughly half-sum physical DTLZ1 population is evaluated against a roughly unit-sum registered reference. Confirm transformation provenance and per-objective bounds before correcting any score.
- **DECISION — Do not patch scores or recompute `audit_report` yet:** first establish the actual normalization metadata (`ideal`, `nadir`) and ensure all input populations, initial generations, GT references, physical metrics, and Q-baselines use one coordinate system. Regenerate report/baselines only after these invariants and the independent analytical tests are validated.


### 2026-10-21 — Verified source of normalization drift; first calibrated-coordinate correction

- **CONFIRMED — exact calibration metadata:** [Actions run 39660219372](https://github.com/monacofj/moeabench/actions/runs/39660219372) confirmed `calibration_package.npz['gt_norm']` is **exactly equal** to the DTLZ1 registered reference. Its normalization bounds were `ideal=[3.10853240e-06, 5.66859151e-06, 1.98573640e-05]`, `nadir=[0.49471033, 0.49559658, 0.49998148]`. Raw optimizer objective sums were near `0.5089`, registered GT sums near `1.0048`; transforming the raw population yielded objective sums near `1.0237`. Median raw-population to normalized-GT distance was `0.28718182553465516`; after transforming the population it was `0.011754545757937142`.
- **CHANGE — Traceable normalization:** `_resolve_diagnostic_context` now gives precedence to registered canonical GT for automatic calls and applies the saved `(P-ideal)/(nadir-ideal)` transform to both final and initial MOP populations **only when the full GT exactly matches the package's normalized GT**. Explicit unrelated GTs are not automatically normalized. Context exposes the normalization metadata. The calculation is not based on a guessed factor of two.
- **PASS — Post-normalization real run:** [Actions run 39660695325](https://github.com/monacofj/moeabench/actions/runs/39660695325) passed `277/277` unit and `23/23` basic integration tests and ran real DTLZ1/NSGA-II (seed 1). Reported `Q_CLOSENESS=0.7560986200725877`, `Q_COVERAGE=1.0`, `Q_GAP=1.0`, `Q_REGULARITY=0.0`, `Q_BALANCE=0.050000000000000044`, `Q_HEADWAY=0.9998873950510557`, physical `CLOSENESS=0.1652224416805805`. These are observations, **not** gold-standard values. The previous run used a different raw population distribution despite the same experiment seed; do not attribute their numerical differences solely to normalization without a controlled fixed-population comparison.
- **TEST LOG CORRECTION — Diagnostic double normalization:** the CI diagnostic script in that run mistakenly reapplied `ideal/nadir` to `context['P_final']`, which is *already* calibrated. Its printed 'P normalized median' and 'median P-to-GT normalized' values from the second transform must be ignored. The subsequent workflow has been corrected to compare `exp.front(-1)` (raw) with `context['P_final']` (normalized) and assert their exact transformation relation.
- **FOUND ADDITIONAL VIOLATION — Marker scaling hack:** `moeabench/view/topo.py` was silently forcing `s_k=0.02` for DTLZ1 when computing clinical marker classes, expressly to produce more attractive marker symbols. This violates the normatively defined FPS resolution and makes the visualization disagree scientifically with the diagnostic. **REMOVED** this ad-hoc override. Visualization now detects a display GT obtained by reversing the canonical normalization and passes its calibrated counterpart to marker evaluation.
- **CHANGE — Correct plotted geometry:** `examples/example_13.py` now inverse-transforms the stored calibrated GT to raw objective coordinates before overlaying it with the raw experiment's point cloud; otherwise the plot depicts objects from incompatible coordinate systems. Regression tests now cover both normalized and unnormalized display references. CI validation of this last set of changes is pending.
- **OPEN — Residual and calibrated artifacts:** `ideal_res` is still generated using a same-GT subset and recovered by a missing helper under `except: pass`. The frozen baseline JSON and versioned audit report may therefore need scientifically justified regeneration. This remains a separate issue-75 acceptance gate.


### 2026-10-21 — Visual marker integration regression detected and corrected

- **FAIL — First post-normalization integration run:** [Actions run 39660942624](https://github.com/monacofj/moeabench/actions/runs/39660942624) passed `8` focused tests and `278` unit regressions but reported `6 failed, 17 passed` in integration. All failures were marker-alignment tests in `tests/integration/test_topology_marker_alignment.py`; lightweight test plot-source objects expose `mop` but not the `pop()` API expected by the full diagnostic context resolver, causing marker classification to be swallowed/fallback to no marker.
- **CORRECTION — Narrow resolver precondition:** `view.topology` now only uses the full calibrated coordinate resolver for true Experiment/Run-like sources with a callable `pop()`; other sources preserve the existing marker classification path. This deliberately avoids treating simple source metadata as proof of full experiment semantics.
- **SOURCE CORRECTION — Removed false DTLZ1 sensitivity override:** no hardcoded `s_k=0.02` may silently change clinical marker meaning. Marker semantics must reuse the same scientific calibration scale as FAIR, while plotted objects remain in compatible coordinates.
- **TESTS PENDING:** the next branch CI run after this correction will decide whether the six marker regressions are fixed; a source patch alone is not a passing test.


### 2026-10-21 — Calibrated plotting and marker regression gate passed

- **PASS — Visualization regressions restored:** [Actions run 39661368040](https://github.com/monacofj/moeabench/actions/runs/39661368040) passed `8` issue probes, `2` Example 13 interface cases, `278/278` unit tests and `24/24` basic integration tests. The six marker regressions from run 39660942624 are no longer present.
- **PASS — Real plot geometry and clinical symbols:** [Actions run 39661479682](https://github.com/monacofj/moeabench/actions/runs/39661479682) independently passed the same regression suites and ran the real 280-generation DTLZ1 pipeline. It verified that `P_calibrated == (P_physical-ideal)/(nadir-ideal)`, plotted the inverse-transformed GT alongside raw population coordinates and asserted that `100` point-wise clinical marker symbols were assigned (no swallowed marker error).
- **OBSERVED — Plausible aligned coordinates, not a Q oracle:** that run recorded `median(sum(P_physical))=0.5082769538375382`, `median(sum(P_calibrated))=1.024478214606263`, `median(sum(GT_calibrated))=1.004812790875524` and `Q_CLOSENESS=0.9169435371300113`. Differences from earlier runs under the same declared seed must be investigated separately; no hard-coded expected Q was introduced.
- **CHANGE — Recovery of nonzero ideal residue:** introduced `baselines.get_baseline_data(problem,k,metric)` so `fair.closeness()` can retrieve the actual `ideal_res` record; eliminated bare `except: pass` and now reject negative/non-finite residues. Added independent tests for a nonzero residue and invalid calibration. **Important:** the estimator that writes `ideal_res` to frozen baselines is still structurally zero and has not yet been corrected; this change only repairs retrieval and validation.
- **CI PENDING — Residue tests:** the last two new tests and accessor change have not yet received a final run result at the moment of this entry.


### 2026-10-21 — Final validation checkpoint of the current investigation branch

- **PASS — Latest focused + broad regression suite:** [Actions run 39661691468](https://github.com/monacofj/moeabench/actions/runs/39661691468) completed successfully after introducing the `ideal_res` accessor and independent nonzero/invalid-residue tests: `10 passed` in the focused issue-75 suite, `2 passed` for the Example 13 interface variants, `280/280` unit tests and `24/24` basic integration tests. Real DTLZ1/NSGA-II quality computation and 100 visualization marker assignments also completed.
- **CAUTION — Reproducibility not yet demonstrated:** the same nominal Example 13 seed=1 has produced different Q-Closeness observations in independent CI runs (e.g. `0.9169435371300113` in run 39661479682 versus `0.5593203805865916` in run 39661691468). Those numbers were not used as immutable oracles. Investigate seed propagation, process/global RNG, baseline geometry/calibration stochasticity and CI dependency provenance before declaring numerical stability.
- **OPEN — Calibrated ideal residue is not yet scientifically validated:** its stored generation formula still samples ideal points **from the GT itself**, so `ideal_res=0` by construction. Correcting the read path is necessary but not sufficient; investigate and implement genuinely independent ideal sampling when the MOP has a true analytical front, or document an explicit zero/undefined policy for empirical GTs. This may invalidate distributed baseline records.
- **OPEN — Regeneration/versioning gate:** `calibration/audit_report.html` and versioned audit JSON/HTML plus any baselines affected by the science changes must be regenerated and compared only after the `ideal_res` definition, coordinate system, and reproducibility are validated. The existing versioned `v0.14.0` report is **not** updated in this draft PR, and historical versions should be preserved.
- **STATUS — Not accepted:** ADR 0051 stays Proposed; PR #76 remains Draft; no merge or release claim.


### 2026-10-22 — Independent residue design and NSGA-II RNG audit

- **CONFIRMED SPECIFICATION CONFLICT — `ideal_res`:** ADR 0037 requires a *fresh analytical* reference outside the discrete GT. The previous implementation used FPS of the same GT at another seed, yielding exactly zero by definition. There is no evidence that such zero was a valid measured discretization floor.
- **CONFIRMED DETERMINISTIC-SAMPLER PITFALL:** DTLZ1's analytical `ps(n_points)` restarts a fixed RandomState(42) for every request. Two separately requested samples of the same size therefore repeat exactly; requesting more points and excluding all GT coincidences is necessary for any empirical finite-GT correction. DTLZ2's built-in analytical sampler is also deterministic, with a repeated prefix in M>=3.
- **CHANGE — Independent analytical ideal sample:** `calibration.independent_analytical_reference(mop,gt)` deterministically draws additional PF points and removes the overlapping GT prefix. `estimate_ideal_residue(gt,fresh,s_K)` validates sample dimensionality, finiteness, positive scale and independence before computing median nearest-GT distance divided by `s_K`.
- **CHANGE — Provenance and unavailable policy:** `calibrate_mop()` and the offline `generate_baselines.py` supply a verified independent reference for analytical GTs. For externally supplied or empirical GTs with no independent analytical reference, correction remains zero **and is explicitly labeled** `unavailable_no_independent_ideal` rather than being passed off as an observed zero. The offline generator requires reproducing the exact packaged `gt_norm` before trusting the analytical MOP sample.
- **CHANGE — Baseline ECDF parity:** the entire Closeness blur ECDF and its anchors are transformed with `u' = max(0,u-ideal_res)`, exactly as physical Closeness is. Correcting physical measurements without correcting their reference ECDF would otherwise create a new scientific inconsistency.
- **CHANGE — Non-destructive migration tool:** `calibration/scripts/migrate_closeness_residue.py` supports previewing per-K epsilon and Q-anchor differences from the frozen `baselines_v0.14.0.json`. It refuses to overwrite the source, double-apply a correction or migrate mismatched GTs; it stages a replacement only when an explicit separate output path is given. **No new certified baseline or audit report has been published.**
- **CONFIRMED REPRODUCIBILITY DEFECT — NSGA-II:** `NSGA2deap` initialized a private `self.rng` but called DEAP selection/crossover/mutation functions which use Python's module-global `random`. The purported run seed consequently did not control the whole evolutionary search.
- **CHANGE — RNG isolation:** these DEAP stochastic calls now execute using the run-private RNG stream with the process RNG saved/restored. New tests compare complete independently executed same-seed runs despite deliberate global RNG perturbation, verify distinct-seed differences and check no global RNG state leakage. This correction is limited to serial runs; the RNG swap would require further design before concurrent evolution.
- **FAILED CI (recorded, not hidden):** earlier runs [39791858292](https://github.com/monacofj/moeabench/actions/runs/39791858292), [39791934666](https://github.com/monacofj/moeabench/actions/runs/39791934666), [39792221190](https://github.com/monacofj/moeabench/actions/runs/39792221190) and [39792353644](https://github.com/monacofj/moeabench/actions/runs/39792353644) hit an existing mocked `_generate_baselines` implementation in `test_dtlz2_calibrate_uses_corrected_analytical_gt` which did not accept the new `independent_ideal` parameter. This was a test-double compatibility failure, not a numerical assertion. The test double has been updated **and now also checks** that new samples lie on the true DTLZ2 PF but not in GT. Revised CI still requires confirmation.
- **TESTS PENDING:** report reproducibility results and dry-run migration deltas from [Actions run 39792974536](https://github.com/monacofj/moeabench/actions/runs/39792974536) and later runs once completed. Do not call CI successful until their conclusions are available.
- **REPORT DEPENDENCY:** the official 22 MB `baselines_v0.14.0.json` is retained unchanged; the canonical HTML audit report references `v0.14.0`. A scientifically checked replacement baseline, recalculated benchmark/audit JSON, and versioned HTML will require their own validation before the report alias can be updated.


### 2026-10-22 — Baseline migration blockers and audit_report pipeline review

- **FAILED CI — legacy offline generator imported a removed module:** [run 39792917699](https://github.com/monacofj/moeabench/actions/runs/39792917699) and [run 39792974536](https://github.com/monacofj/moeabench/actions/runs/39792974536) (on earlier commits) failed in the new migration smoke test with `ModuleNotFoundError: No module named 'moeabench.diagnostics.fr'` from `calibration/scripts/generate_baselines.py`. This previously undetected legacy import prevented running the official baseline generator with current MoeaBench. Corrected the import to the current `fair` module and a further legacy `base.version()` call to `system.version(show=False)`. Full replacement generation is not yet executed.
- **FAIL-SAFE — certified alias protection:** full `generate_baselines.py` now requires `MOEABENCH_BASELINE_OUTPUT` to point to a distinct staged output path and rejects attempts to overwrite the frozen canonical alias. The new incremental migration script requires an explicit separate `--output` for writing and rejects double applications of residue correction. Updated `docs/CONTRIBUTING.md` accordingly.
- **SCIENTIFIC COMPATIBILITY — archived GT sampling:** numerical inspection of built-in DTLZ1 analytical generation with current seed=42 revealed different ideal/nadir extrema from the historical `calibration_package.npz` reported by CI. Exact point-for-point reproduction of an archival GT is not a safe criterion for analytical manifold equivalence across sampler revisions. The staged generator now checks median distance between the newly generated analytical PF and archived GT relative to archival local nearest-neighbor spacing; the archived baseline GT itself must still match the packaged `gt_norm` exactly before migration. This detects grossly different manifolds without rejecting statistically different valid surface samples.
- **CONFIRMED REPORT REGENERATION BLOCKER:** `calibration/scripts/audit_calibration.py` still imports removed `moeabench.diagnostics.fr`, calls the removed `QualityAuditResult.summary()`, uses an artificial random initial population for longitudinal Headway, and writes/resumes the hard-coded historical `audit_v0.13.1.json`. Thus **the published audit report cannot safely be rebuilt by simply rerunning the legacy scripts**. Before versioning a new `audit_report`, the audit source must be brought into scientific/API parity, baseline references and raw input coordinates must be reconciled, and the old versioned artifacts must be retained for comparison.
- **NEXT GATE:** validate the DTLZ1 impact report and exact same-seed runs in CI, then decide whether to extend migration to other bundled MOPs. Do not alter the 22 MB canonical `baselines_v0.14.0.json` or the audit alias without independent numerical confirmation and a new versioned release decision.


### 2026-10-22 — Additional custom-GT physical invariant

- **CORRECTNESS GAP FOUND:** although `audit(quality=True)` refused Q-scores for an explicit GT incompatible with the registered baseline, `fair.closeness(...,quality=False)` could still retrieve the frozen baseline's `ideal_res` using only a matching problem name. An epsilon calibrated for GT A must **never** be subtracted from distances measured against GT B; even the physical metric would then be biased.
- **FIX — Reference-specific epsilon:** `baselines.reference_compatibility(problem,gt)` compares the complete selected GT with the canonical registered calibration GT. The physical `fair.closeness` retrieves stored `ideal_res` only when compatibility is positively established; for unrelated references it uses the uncorrected physical distance, unless an explicit `ideal_res` is supplied by the caller. This conservatively preserves meaningful physical metrics without borrowing evidence for a different reference.
- **TESTS ADDED:** independent regression demonstrates that an incompatible GT never calls `get_baseline_data`, while an explicit correction still gives the analytically predicted `max(0,u-epsilon)` distribution. Existing tests with a known matching reference explicitly mock that compatibility rather than relying on unrelated baseline global state.
- **VERIFICATION STATUS:** recent GitHub Actions were queued at this checkpoint; tests and full numerical migration have **not** been certified green for the latest patch. Do not update official baselines or audit report on this evidence alone.


### 2026-10-22 — First successful DTLZ1 dry-run and same-seed replay

- **PASS — Reference run 39794475828:** [GitHub Actions](https://github.com/monacofj/moeabench/actions/runs/39794475828) completed successfully on commit `34787615`, including the branch's FAIR probe suite, broad unit/basic integration suites, independent DTLZ1 residue preview, and two complete NSGA-II/DTLZ1 executions with seed 1.
- **PASS — Strict same-seed replay:** the second 280-generation DTLZ1/NSGA-II run, executed after deliberately perturbing global Python and NumPy RNG states, produced an exactly equal final objective array; the full `Q_CLOSENESS` numerical score was exactly equal as well. This confirms reproducibility of this configuration with the new run-private DEAP RNG. Additional algorithms/versions remain outside this exact claim.
- **PASS — Independent finite-GT residue:** for the frozen DTLZ1 registered GT and `K=100`, independent analytical PF sampling yielded `ideal_res = 0.028020957844485463` in FPS-resolution units. The legacy value was zero by construction. The dry-run migration completed with `failed = {}` and did not modify the certified baseline.
- **QUANTIFIED ANCHOR CHANGE — K=100:** `rand50` for the blinded Closeness baseline changes from `2.022782510492534` to `1.9947615526480484` after the same residue correction. These numbers are *staged predictions*, not yet values published by a certified baseline.
- **TEST-GATE — Same-population old/new comparison:** new run [39796130631](https://github.com/monacofj/moeabench/actions/runs/39796130631) stages a candidate JSON in `/tmp` and then computes FAIR and Q twice on the identical completed experiment using `use_baselines`. It asserts `u_new = max(0, u_old-epsilon)`, unchanged remaining five FAIR physical metrics and their Q-scores, and a finite new Q-Closeness score. This run was queued when this note was recorded. A passing previous run does not imply this new check passes.
- **SAFETY — Non-destructive comparison:** the 22 MB certified `baselines_v0.14.0.json`, its `baselines.json` alias, all historical audit JSONs and `calibration/audit_report.html` remain unchanged. The staged JSON is an ephemeral CI artifact and is not a new official baseline.


### 2026-10-22 — DTLZ1 fixed-population baseline comparison passed; correction to earlier K attribution

- **PASS — [Actions run 39796130631](https://github.com/monacofj/moeabench/actions/runs/39796130631):** `15` issue-specific probes, `2` Example 13 interface tests, `288/288` unit tests, `24/24` basic integration tests and two equal-seed full DTLZ1/NSGA-II runs. The baseline comparison used exactly the **same completed experiment** twice: frozen canonical and staged non-destructive candidate baseline.
- **PASS — Corrected Closeness and preservation:** the actual experiment's snapped baseline cardinality was `K=50`; staged `ideal_res=0.028020957844485463`; physical Closeness changed `0.7628883764289277 → 0.7348674185844423`, exactly following `u_new=max(0,u_old−epsilon)`. Coverage, Gap, Regularity, Balance and Headway, and their five Q-scores, were numerically unchanged. There were no missing clinical calibration records.
- **INCONCLUSIVE — Q sensitivity for this population:** `Q_CLOSENESS=0.0` under both frozen and staged baselines. This result is not evidence that the correction is ineffective; the Wasserstein quality map has a clipping gate and this particular approximation was classified at the worst end. A separate synthetic, fixed-distribution test is needed to demonstrate the changed calibration and the Q response in a nondegenerate range.
- **CORRECTION TO PRIOR LOG ENTRY:** earlier ADR notes incorrectly assigned `ideal_res=0.028020957844485463` and the `rand50=2.022782510492534→1.9947615526480484` change to **K=100**. These figures actually correspond to **K=50**. For **K=100**, the independently computed `s_K=0.08847951537468546`, `ideal_res=0.04113807806761828` and `rand50=2.0401773038848496→1.9990392258172314`. The erroneous historical attribution is retained above for traceability; this correction is authoritative.
- **OTHER VERIFIED RESIDUES:** `K=150: epsilon=0.051032004238763994`; `K=200: epsilon=0.058975789112873` (DTLZ1, current independent analytical sample). These are staged values, not yet certified replacements.
- **CERTIFIED ARTIFACTS UNCHANGED:** original `baselines_v0.14.0.json`, the canonical baseline alias and the audit report have not been overwritten.


### 2026-10-22 — Pointwise clinical markers and legacy audit input inventory

- **NEW SCIENTIFIC INCONSISTENCY DETECTED:** `q_closeness_points` in the clinical marker plot still divided raw point-to-GT distances by `s_K` and compared them with `rand50`. Once the frozen baseline has a nonzero `ideal_res`, however, `rand50` and scalar FAIR Closeness are corrected while marker inputs were not. This would cause the visual markers and FAIR physical layer to disagree on the same point.
- **CHANGE — Consistent pointwise calibration:** 2D point clouds are now evaluated via `fair.closeness(...).raw_data`, reusing GT-specific correction and consistent coordinates. The per-point log-linear marker scores are bounded to `[0,1]`. Added a synthetic analytic test that compares the vector of marker scores against the physical distances after nonzero `ideal_res`; this change requires its own green CI result.
- **TEST — Nondegenerate Q sensitivity:** a controlled Wasserstein example independently establishes `Q_CLOSENESS=0.5` for samples at 1 versus random reference at 2, and `Q_CLOSENESS=1/1.8\approx0.5556` after subtracting an epsilon of 0.2 from both sides. The DTLZ1 optimization's `Q_CLOSENESS=0` under both baselines was therefore documented as a clipped, inconclusive case, not mistaken for a calibration no-op. Test execution is pending.
- **AUDIT REPORT DATA INVENTORY — Script added:** `calibration/scripts/audit_report_preflight.py` enumerates historical final front CSVs, genuine generation-zero snapshots, intermediate checkpoints, GT files and `calibration/reports/baselines.csv`. It reports which stored runs can support longitudinal Headway *without substituting random initial populations*. Added a synthetic unit test proving `gen100\neq gen0` and a CI step retaining the structured inventory. No historical reports or source data are modified.
- **PARTIAL REPOSITORY INSPECTION:** `calibration/data/calibration_data` contains final and `gen100,gen200,...` example files; the `calibration/reports` directory listing does not contain `baselines.csv`, despite the legacy audit script requiring it. A full inventory needs CI rather than assuming that the first 1000 directory entries represent all files. The report alias remains frozen until a scientifically valid generation pipeline is established.
- **STATUS:** [Action run 39796130631](https://github.com/monacofj/moeabench/actions/runs/39796130631) validated the preceding calibration changes. Later marker and audit-input checks are queued and must not be called passing until their own results are available.


### 2026-10-22 — Historical audit regeneration decision and FAIR-only prototype

- **PASS — Actual archival preflight in [Actions run 39796767703](https://github.com/monacofj/moeabench/actions/runs/39796767703):** a full inventory (not merely the first directory page) found `42` studies and `985` archived final-front run records, `0` actual generation-zero population snapshots and therefore `985` Headway-ineligible runs. The legacy script's required `calibration/reports/baselines.csv` is also absent; the current historical audit cannot be rebuilt as-is. The same run passed `290/290` unit and `24/24` basic integration tests.
- **SCIENTIFIC DECISION — Separate physical retrospective analysis from clinical certification:** archived final fronts and their matched 3D GTs can support five physical FAIR statistics, but not calibrated clinical Q-scores or longitudinal HEADWAY. We **will not** present an artificial random initial population, a generation-100 checkpoint or an unrelated global calibration as genuine evidence.
- **IMPLEMENTATION — Non-destructive partial prototype:** `calibration/scripts/audit_fair_archive.py` reads a matched historical GT and actual saved final population for each run, applies minimization non-dominated filtering, constructs its own same-GT FPS `U_K`, `s_K` and Balance reference histogram, and computes physical CLOSENESS, COVERAGE, GAP, REGULARITY, BALANCE. CLOSENESS is explicitly marked **uncorrected** for finite-GT ideal residue, because archived raw GTs have no verified independent analytical residue. HEADWAY is null unless a **genuine** `gen0` file is present; Q-scores are always null without compatible calibration. The resulting JSON has a deliberately separate schema and a noncertified flag. It cannot overwrite any `calibration/reports/*` file.
- **TESTS — Positive and negative cases:** synthetic three-objective fronts test expected zero Closeness, finite other metrics, explicit missing HEADWAY/Q, positive genuine-initial Headway, and fail-closed behavior when an expected GT is missing. No historical metric is claimed correct until the same calculation runs on actual archived CSVs in CI.
- **CI GATE:** [run 39798432210](https://github.com/monacofj/moeabench/actions/runs/39798432210) executes these tests and processes two stored `DTLZ1_NSGA2` fronts. It preserves an independent provisional JSON artifact, alongside the GT preflight and DTLZ1 calibration-impact files. CI completion and actual numerical output remain pending in this entry.
- **OPEN RELEASE WORK:** a future certified `audit_report` needs a new independently reviewed source-data protocol, compatible baseline calibration, and actual longitudinal run histories (or explicit unavailable Headway). The legacy report script is not presently fit for reproducing current FAIR methodology.


### 2026-10-22 — Real historical FAIR-only run succeeded (two archived fronts)

- **PASS — [Actions run 39798432210](https://github.com/monacofj/moeabench/actions/runs/39798432210), checked while real Example 13 was finishing:** `17/17` issue-specific tests, `2/2` Example 13 interface cases, `295/295` unit tests and `24/24` basic integration tests passed. The provisional independent archive auditor processed `2` genuine `DTLZ1_NSGA2` final fronts with `0` input errors, `2` genuinely missing Headway values and no clinical Q-score claims. The stage uploaded an isolated JSON artifact labeled `fair-75-historical-physical-only`. Do not confuse successful individual steps with overall workflow conclusion until that has been verified.
- **NUMERICAL EVIDENCE — Historical DTLZ1/NSGA2 run00:** `front_rows=200`, non-dominated `K=200`, `s_K=0.028141702067655178`; `CLOSENESS=0.1929059121773103` (uncorrected finite-GT residue), `COVERAGE=0.01787265392250496`, `GAP=0.03697431047414172`, `REGULARITY=0.014169125640813957`, `BALANCE=0.216334448910683`, `HEADWAY=null`, all `Q=null`. GT: `DTLZ1_3_optimal.csv`, SHA256 of numerical array `a25509f27461fdcc88f2911e8ca5c518bb19f54feb4bde4b1c67b837def1895e`. These values are **provisional physical metrics in historical CSV coordinates**, not certified clinical quality.
- **SUCCESS — Independent source handling:** only genuine final fronts and the named DTLZ1 historical GT were read; generation-100 snapshots were not mistaken for the initial population. The report is not passed to the existing HTML renderer, which expects an incompatible fully calibrated JSON schema.
- **REMAINING SCIENTIFIC CAVEATS:** the historical raw GT is not the same as the normalized registered clinical calibration reference; positive analytical `ideal_res` has not been separately certified for this historical CSV. Rebuilding historical Headway still requires the original run histories or new reproducible optimization runs with preserved `gen0`. The official audit reports and canonical baselines remain untouched.


### 2026-10-22 — Nondegenerate Q response and final checkpoint of this work session

- **PASS — Independent Q-Closeness sensitivity oracle:** CI [run 39797053285](https://github.com/monacofj/moeabench/actions/runs/39797053285) executed the added synthetic distribution test successfully as part of `17` focused probes, `291/291` unit regressions and `24/24` basic integration tests. The analytically known example yields `Q=1/2` for observed/blurred values `1/2`, and `Q=1/1.8` after applying the same `epsilon=0.2` correction to both. This specifically demonstrates a response outside the degenerate zero-clipping region, whereas the DTLZ1 optimization at snapped `K=50` remained clipped at Q=0 before and after calibration.
- **PASS — Historical data audit prototype:** CI [run 39798432210](https://github.com/monacofj/moeabench/actions/runs/39798432210) completed all its available named test and provisional audit steps with `17` issue probes, `295/295` unit tests, `24/24` basic integration tests, `2` archived DTLZ1_NSGA2 runs processed and `0` archive input errors. The provisional JSON artifact is uploaded separately and never substituted for certified audit JSON. The real Example 13 diagnostic step also completed successfully at the last job-level inspection; the workflow conclusion was still propagating in the GitHub summary.
- **REGRESSION EVIDENCE — Visualization parity:** the preceding clinical-marker change and subsequent unrelated additions passed the full basic integration suite, including topology marker alignment. The changed pointwise Closeness uses the same physical corrected values as scalar Closeness.
- **TRACKING:** [Issue #75](https://github.com/monacofj/moeabench/issues/75) has a dated checkpoint linking both CI runs and the historical run00 metric values. ADR 0051 remains Proposed, PR #76 remains Draft, and baseline/report release artifacts are not changed.
- **NEXT SCIENTIFIC GATE:** decide the provenance and regeneration method for `P_0` across the historical datasets (new recorded MOEA runs or explicitly omitted longitudinal scores), and certify new GT-specific baseline material before rendering/versioning a replacement `audit_report`. The five historical physical metrics alone are not a certified clinical Q-score report.


### 2026-10-22 — Implemented genuine-generation-zero pilot protocol

- **NEW PROTOCOL — reproducible fresh optimization:** `calibration/scripts/audit_longitudinal_pilot.py` executes a short DTLZ1/NSGA-II study on the current MoeaBench implementation with an explicit run seed and true retained generation-zero and final populations. It uses the exact normalized registered GT and same-package ideal/nadir coordinates and stores SHA256 hashes of GT, P0 and Pf in both native and calibrated coordinates.
- **INDEPENDENT VALIDATION — longitudinal Headway:** the script independently computes `GD95(Pf→GT)/GD95(P0→GT)` using a KD-tree and verifies agreement with `mb.clinic.audit(exp).fr.metrics['HEADWAY']`, without calling the FAIR Headway implementation as its oracle. It also independently validates the transform `(P_raw-ideal)/(nadir-ideal)` for P0 and Pf. A real P0 is **mandatory**, not substituted from a synthetic random population or an unrelated archive.
- **REPRODUCIBILITY TEST:** `tests/unit/test_longitudinal_pilot_issue75.py` includes an analytic perpendicular-offset example with expected Headway ratio `0.25`, plus a complete twice-executed NSGA-II run with identical seed and exact result comparison. It requires P0/Pf hashes to differ (actual evolution), physical Headway to match the oracle, and provisional status.
- **CI PILOT:** temporary branch workflow [run 39799497479](https://github.com/monacofj/moeabench/actions/runs/39799497479) runs a 60-generation, population-64, seed-7 DTLZ1 pilot and retains its JSON as a separate `fair-75-true-p0-longitudinal-pilot` artifact. It explicitly refuses to write under `calibration/reports`. This is *not* a recreation of an archived trial and cannot by itself certify a complete replacement clinical `audit_report`.
- **STATUS WHEN RECORDED:** the new run was queued; no PASS or FAIL is claimed until the actual tests finish. The prior [run 39798432210](https://github.com/monacofj/moeabench/actions/runs/39798432210) completed `success` for the archival physical-only path.


### 2026-10-22 — Genuine P0 and candidate-baseline pair confirmed, report preview initiated

- **PASS — Entire true-P0 pilot against frozen baseline:** [GitHub Actions run 39799497479](https://github.com/monacofj/moeabench/actions/runs/39799497479) completed with success, including `297/297` unit tests and `24/24` basic integration tests. For DTLZ1/NSGA-II, population=64, generations=60, seed=7: true generation-zero `P0=64` points and final non-dominated front `Pf=61`. The physically measured `HEADWAY=0.12798362165781485` **equaled** the independent `GD95(Pf,GT)/GD95(P0,GT)` oracle. SHA256 digests of registered GT, calibrated/raw P0 and Pf are recorded in the separately uploaded pilot artifact. The pilot is a *new experiment*, not an attempt to masquerade as one of the 985 historical runs.
- **PASS — Fixed-run candidate-baseline comparison:** [GitHub Actions run 39799845693](https://github.com/monacofj/moeabench/actions/runs/39799845693), completed job successfully: on the same P0/Pf and GT, staged `ideal_res=0.028020957844485463` transformed physical Closeness from `133.37693550984966` to `133.34891455200517`. The other five FAIR metrics and their Q-scores remained numerically identical. `Q_CLOSENESS=0` for both candidates, which is **not** a calibration validation beyond the clipping limit. `Q_HEADWAY=0.8262538800687846` but the other five clinical scores were zero in this deliberately short, underconverged pilot. Treat the experiment as a provenance/geometry test rather than a converged search-quality benchmark.
- **REPORT GENERATOR — a new provenance-separated preview:** `calibration/scripts/render_fair_evidence_preview.py` renders a standalone, explicit **NOT CERTIFIED** HTML preview from two separately labeled JSON sources: the genuine-P0 clinical pilot, and archived fronts with only physical FAIR and no invented Headway/Q. It refuses the official `calibration/reports` destination and canonical HTML alias, validates each input's declared provenance and missing evidence status, and escapes all displayed strings. Tests verify it refuses unsupported archival Q-scores and pilot Headway lacking genuine P0.
- **CI REPORT PREVIEW:** the latest branch workflow [run 39800276405](https://github.com/monacofj/moeabench/actions/runs/39800276405) includes the provisional HTML preview as an independent CI artifact, after generating/validating both JSON sources. Its result was still pending at the time of this note. No official `audit_report` or versioned certified baseline was modified.


### 2026-10-22 — First complete provisional two-cohort report generated and validated

- **PASS — Latest branch CI [run 39800276405](https://github.com/monacofj/moeabench/actions/runs/39800276405)** completed the entire workflow successfully after the new report preview and tests: `17/17` focused issue-75 probes, `2/2` Example 13 interface cases, `301/301` unit tests, and `24/24` basic integration tests. The real DTLZ1/NSGA-II numerical checks passed, and the independent same-seed replay still gave an identical final front and Q-Closeness.
- **OUTPUT — Provisional HTML:** `calibration/scripts/render_fair_evidence_preview.py` rendered `fair-75-provisional-evidence.html` (4094 bytes) from the two JSON source artifacts. The [Actions run 39800276405](https://github.com/monacofj/moeabench/actions/runs/39800276405) retained it as artifact `fair-75-provisional-evidence-preview` (GitHub artifact ID `12435771153`). The file is clearly titled **NOT CERTIFIED**. Historical runs are deliberately reported with unavailable HEADWAY and Q-scores, not synthetic or retrospectively fabricated values.
- **SUPPORTING ARTIFACTS:** the same run retained `fair-75-historical-physical-only` (ID `12435646518`), `fair-75-audit-report-preflight` (ID `12435262685`), `fair-75-dtlz1-residue-impact` (ID `12434838364`) and `fair-75-true-p0-longitudinal-pilot` (ID `12434484199`). These provide inspectable numerical input/output provenance for the HTML, rather than merely a presentation.
- **PROTOCOL ENFORCEMENT — Tests:** preview tests reject historical Q-scores lacking calibration, reject longitudinal pilot data without validated actual P0, test rendering of unavailable fields and prevent publishing the provisional report to the canonical release path. The live report stage completed with `success`, as did all supporting computations.
- **SEPARATION REMAINS REQUIRED:** the historical archive and freshly optimized pilot are **not** pooled into an apparently homogeneous clinical audit. The short seed-7 DTLZ1 pilot had `HEADWAY=0.12798362165781485` from true P0 but was not converged (Closeness `133.37693550984966`), leaving five Q-scores at zero. Neither a successful preview nor passing tests is a certificate that those quality scores establish optimal search performance.
- **NON-DESTRUCTIVE:** the original `baselines_v0.14.0.json`, frozen `baselines.json` reference, `calibration/audit_report.html`, and historical `audit_v*.json` files remain untouched. ADR 0051 stays Proposed and PR #76 stays Draft; no merge/release authorization is implied.
- **NEXT ACCEPTANCE GATES:** independently certify revised GT-specific calibration across the benchmark suite, choose how much of the historical dataset can legitimately be reexecuted with actual P0, and design/version the final official report schema with explicitly unavailable metrics where data genuinely cannot be recovered. Remove/merge the temporary investigation CI workflow into permanent tests before the PR can be merged.


### 2026-10-22 — Multi-MOP computational cost pilot initiated

- **BLOCKER DISCOVERED — Invalid legacy full Headway calibration:** `calibration/scripts/generate_baselines.py::calculate_metrics` calls `fair.headway(samples, gt_norm, s_k=s_fit, raw=True)` for a standalone random or ideal sample. The `raw=True` branch reports the final `GD95/s_K`, **not** the longitudinal ratio `GD95(P_final)/GD95(P_initial)`. Thus the historical full-generator HEADWAY baseline is scientifically invalid for the current FAIR definition. The sidecar `calibration._calc_all()` similarly calls Headway with no initial population, yielding an undefined statistic. **Do not certify either full generator until paired P0/Pf calibration is designed and validated.**
- **NEW BENCHMARK — Legitimate five-metric trial:** created `calibration/scripts/benchmark_fair_calibration_cost.py` to measure actual costs, using packaged GTs, the same FPS/K scale and centroid protocol, and *full planned per-K draw counts* (200 random + 30 ideal, excluding HEADWAY). It calculates the independent ideal residue and corrected Closeness blur baseline, plus separate random/ideal distributions for Coverage, Gap, Regularity and Balance. The result is a **noncertified, non-destructive timing/evidence JSON**, not a replacement for the six-metric official baseline.
- **CHOICE OF TEST CASES:** DTLZ1 (simplex, planar), DTLZ2 (spherical) and DTLZ5 (degenerate curved/frontier), with `K=20` and full 200/30 sampling counts. This deliberately tests distinct manifold geometries without multiplying a preliminary experiment across 44 different K values and 14 MOPs before knowing cost or failures.
- **INDEPENDENT TESTS:** `tests/unit/test_fair_calibration_cost_issue75.py` asserts no Headway is derived from one-population samples, verifies physical metric finiteness, preserves fail-closed behavior for incompatible references, and checks invalid sample parameters before expensive computation.
- **CI LAUNCHED:** [Actions run 39804447513](https://github.com/monacofj/moeabench/actions/runs/39804447513) executes the full unit/basic integration suite and then a three-MOP calibration timing trial, saving an inspectable artifact `fair-75-three-mop-calibration-cost`. At the time of this entry the workflow was **in progress**, not yet a passing result.
- **INVARIANTS:** no official `baselines_v0.14.0.json`, `baselines.json`, versioned audit JSON, or `calibration/audit_report.html` was modified. ADR remains Proposed and PR #76 remains Draft pending evaluation.


### 2026-10-22 — First timing run failed at CLI import (not numeric calibration)

- **PASS — Existing tests:** GitHub Actions [run 39804447513](https://github.com/monacofj/moeabench/actions/runs/39804447513) passed its focused, unit, basic integration and prior DTLZ1 staging/report steps before entering the new timing benchmark.
- **FAIL — New benchmark CLI bootstrap:** direct invocation of `calibration/scripts/benchmark_fair_calibration_cost.py` raised `ModuleNotFoundError: No module named 'calibration'` while importing the shared generator helper. None of the three MOP timing results was measured on this attempt; do **not** call this a physical or algorithmic failure, and do not invent timings.
- **FIX — Import-path consistency:** the benchmark now explicitly adds its repository root to `sys.path` before package imports, following the already validated `migrate_closeness_residue.py` CLI approach. The corrected test run must be checked before recording any computational costs.


### 2026-10-22 — Measured full-draw K=20 calibration costs across three MOPs

- **PASS — [GitHub Actions run 39804855951](https://github.com/monacofj/moeabench/actions/runs/39804855951):** all targeted tests, `304/304` unit tests, `24/24` basic integration tests and the three-problem benchmark passed. Saved `fair-75-three-mop-calibration-cost` JSON (artifact ID `12436952621` from an earlier run is **not** applicable to this trial; fetch the actual artifact from run 39804855951 for the authoritative record).
- **MEASURED — K=20, N_random=200, N_ideal=30:** DTLZ1 `3.542953798 s` with independent `ideal_res=0.015395137929985012`; DTLZ2 `3.686852346 s` and `ideal_res=0.02096790311655496`; DTLZ5 `4.112692919 s` and `ideal_res=0.000538819487576958`. All three returned `residue_source=independent_analytical_pf`. These **per-problem wall times** include package reading, GT-normal estimation, FPS/clusters, blur/residue and all four other single-population physical metric samples, but expressly exclude longitudinal HEADWAY and any full MOEA optimization.
- **PERFORMANCE BREAKDOWN:** DTLZ1: GT preprocessing `0.4977s`, blur+residue `1.4260s`, random draws `1.1456s`, ideal draws `0.4398s`. DTLZ2: `0.4710+1.5988+1.1467+0.4360s` respectively. DTLZ5: `0.4500+2.1105+1.0864+0.4382s`. The difference is modest for planar/spherical/degenerate fronts at this K and M=3.
- **CAUTION — Do not extrapolate linearly across K and MOPs:** the suite has 44 K values and 14 packaged MOPs, but some steps are amortizable across K while other draw costs increase with population size or objective dimension. A large-K `K=100` follow-up was added to CI to bound scaling before deciding GitHub versus local batch execution.
- **FAILURE LOG — Prior CI bootstrap:** [run 39804447513](https://github.com/monacofj/moeabench/actions/runs/39804447513) had failed before starting calibration with `ModuleNotFoundError: calibration`. The fixed standalone script reran successfully in 39804855951. The error was a CLI path issue; no mathematical correction was made to force results.
- **SCIENTIFIC LIMITATION:** a successful five-physical-metric timing pilot is **not** a certified replacement six-metric clinical baseline. The offline full generator's HEADWAY protocol remains incompatible with its present longitudinal definition and must be redesigned around true P0/Pf before certifying this score.


### 2026-10-22 — K=100 measured scaling and remote execution decision

- **PASS — [GitHub Actions run 39805359181](https://github.com/monacofj/moeabench/actions/runs/39805359181):** targeted scientific checks plus `304/304` unit and `24/24` basic integration tests passed; DTLZ1, DTLZ2, DTLZ5 successfully completed two full-draw trials at `K=20` and `K=100`, with 200 random and 30 ideal populations each. No certified baseline artifacts were changed.
- **K=20 in same runner:** DTLZ1 `2.460463993s`, DTLZ2 `2.629698032s`, DTLZ5 `2.815228041s`. Differences from the previous 3.54–4.11s run show normal execution noise/setup differences; use same-job measurements for scaling.
- **K=100 in same runner:** DTLZ1 `2.425126727s`, DTLZ2 `2.608054958s`, DTLZ5 `2.123364131s`. **No significant increase** for these three 3-objective problems at K=100 under this five-metric calibration protocol. The method includes normal estimation, FPS and clustering, blur/residue and exactly 200/30 physical-metric samples, but explicitly excludes longitudinal Headway and re-running MOEAs.
- **Independent residue from frozen GT:** at `K=100`, DTLZ1 `0.04113807806761828`, DTLZ2 `0.054159062473055554`, DTLZ5 `0.004156630066974807`. All used `independent_analytical_pf`, not FPS subsets of the same GT. These quantities are provisional until the revised baselines are independently validated and versioned.
- **COMPUTING DECISION:** staged, per-problem calibration of the five one-population FAIR components is practical on GitHub Actions. Naively multiplying roughly 2–4 seconds per K by 44 K and 14 packaged problems suggests an order of tens of minutes in total, **not a reliable precise extrapolation**: geometry, dimensionality, caching and reference density vary. Batch by problem and keep the per-job time below workflow limits. The full six-metric clinical report remains blocked by normative Headway baseline design, not measured CPU cost.
- **NEXT GATE:** exercise a reversible Closeness-only migration on DTLZ1, DTLZ2, DTLZ5 across the full available K grids, verify no changes to the other 11 problems or other FAIR metrics, and stage only a numerical impact report. The canonical baselines and audit_report are unchanged.


### 2026-10-22 — Expanded staged Closeness migration with independent full-JSON verification

- **NEW TOOL:** `calibration/scripts/verify_closeness_migration.py` independently checks old versus staged baselines at every K. It enforces identical top-level metadata, problem lists, GT references and unchanged entries for all non-targeted MOPs and all five non-Closeness metrics, while verifying the clipped `max(0,u-epsilon)` transform for Closeness ECDF samples and anchors. It explicitly checks finite positive epsilon and `independent_analytical_pf` provenance. Invalid target lists or any unrelated drift fail closed.
- **UNIT TESTS:** `tests/unit/test_verify_closeness_migration_issue75.py` tests valid transformation and rejects corruption of a different metric, another MOP or the corrected ECDF itself. This oracle does not call the migration implementation.
- **CI EXPERIMENT:** [run 39806053838](https://github.com/monacofj/moeabench/actions/runs/39806053838) stages a **full-K Closeness-only** correction for DTLZ1, DTLZ2, DTLZ5 from the frozen v0.14.0 JSON. It saves only numerical impact and verification summaries as workflow artifacts. The large candidate is confined to `/tmp` and cannot replace the official baseline or published `audit_report`.
- **STATUS ON ENTRY:** run in progress; no success or K-count/result asserted before actual CI validation.


### 2026-10-22 — Full-K, three-MOP Closeness migration verified without metric drift

- **PASS — [Actions run 39806053838](https://github.com/monacofj/moeabench/actions/runs/39806053838):** `308/308` unit regressions and `24/24` basic integration tests passed; existing real DTLZ1/NSGA-II reproducibility and FAIR checks also completed successfully.
- **PASS — Complete archived K grids:** non-destructively migrated Closeness profiles for DTLZ1 `44/44` K values, DTLZ2 `44/44`, and DTLZ5 `44/44`. The independent JSON verifier reported `PASS_PROVISIONAL_CLOSENESS_ONLY` with `untouched=11` other MOPs.
- **SCIENTIFIC INVARIANTS:** only `closeness` records changed. The verifier independently reconstructed the nonnegative clipped ECDF correction and checked rand50/p95 anchors, positive finite ideal residues and `independent_analytical_pf` provenance. It required equality of all other physical/Q baseline fields, GT references, full problem collection and source metadata. All checks passed.
- **COST IN CONTEXT:** three-problem full-grid residue migration ran in GitHub Actions without breaching the job budget. This is *not* a complete re-calibration of all FAIR reference distributions: Closeness-only correction reuses frozen v0.14.0 ECDF samples, and longitudinal HEADWAY still lacks a scientifically valid paired-trajectory generator.
- **TEST ARTIFACTS:** the run retains `fair-75-three-mop-migration-impact` containing the staged numerical changes and independent verification report. The full 22 MB candidate existed under `/tmp` only, never overwrote the certified v0.14.0 baseline, and was not installed as the public default.
- **CLARIFICATION OF PRIOR K=20 ARTIFACT ID:** GitHub confirmed `12436952621` is the cost JSON artifact for [run 39804855951](https://github.com/monacofj/moeabench/actions/runs/39804855951); the earlier ADR caveat that this ID might be from another run was unnecessary.
- **RELEASE GATE REMAINS OPEN:** the original generator would silently calibrate HEADWAY from GD95 of one population when `raw=True`, not genuine P0/Pf drive. Do not run it for a certified six-metric release. A separate longitudinal calibration design, broader independent checks, a versioned complete baseline, and the definitive `audit_report` schema/rebuild are required. ADR 0051 remains Proposed and PR #76 remains Draft.


### 2026-10-22 — ADR 0049 reconciliation: Headway quality anchors are analytical, not a random ECDF

- **NORMATIVE BASIS:** accepted ADR 0049 defines `HEADWAY = GD95(P_final→GT)/GD95(P_initial→GT)`. Its physical value is unbounded above and requires both populations of the **same** trajectory. The runtime `q_headway` is already `Q=1−clip(log1p(HEADWAY)/log(2),0,1)`: analytical Q anchors are `1` when HEADWAY=0 (complete error removal) and `0` when HEADWAY=1 (no progress), also zero for worsening search (HEADWAY>1). Q requires **no random-ECDF empirical calibration**; an empirical paired-run distribution can be used to test realistic behavior, not to replace this normative mapping.
- **INVALIDITY CONFIRMED — Two existing generators:** both offline `calibration/scripts/generate_baselines.py` and custom sidecar `moeabench/diagnostics/calibration.py` were building `headway` distribution records from *one* random or ideal final population. Offline passed `raw=True`, which ADR 0049 defines as the different GD95/s_K physical view; sidecar passed no P0 at all and obtained an undefined scalar. Both could produce scientifically misleading baseline records, even if all other five FAIR calculations pass.
- **FIX — Remove pseudo-calibration:** both generators now **omit** HEADWAY from one-population random/ideal sampling and instead store explicit transparent metadata `{ideal:0.0, no_progress:1.0, method:'analytic_anchors_adr0049', status:'no_empirical_longitudinal_baseline'}`. They no longer fabricate `headway.rand_ecdf` or `headway.rand50`. This is intentionally a *prospective new staged-baseline semantic change*; the published frozen baseline artifact and official report are unmodified. Existing clinical `q_headway` needs no behavior change.
- **INDEPENDENT ORACLES — Analytic cases:** introduced `calibration/scripts/validate_headway_anchors.py` and `tests/unit/test_headway_anchors_issue75.py`. A 3-objective manifold displaced in a perpendicular objective gives expected Headway ratios `0` (perfect final), `1/4` (partial improvement), `1` (unchanged), `2` (regression), alongside exactly corresponding Q values `1`, `1−log(1.25)/log(2)`, `0` and `0`. Missing genuine P0 must yield NaN for both Headway and Q; a zero-error initial state tests the ADR 0049 edge-case rule.
- **INDEPENDENT REAL RUN CHECKS:** validator also runs actual NSGA-II/DTLZ1 experiments for seeds 7 and 8 using a fixed GT and genuine saved P0/Pf, tests observed Headway against an independent KDTree GD95 oracle, tests Q against the independent log-linear formula and records P0/Pf/GT SHA256. Dedicated tests assert Q_HEADWAY **never** consults the historical empirical Headway baseline.
- **CI PENDING:** [run generated by commit 8ff3007e](https://github.com/monacofj/moeabench/actions) has been triggered by the targeted workflow. Results and any failed tests must be recorded explicitly before declaring this correction ready. No release/merge permitted on an unverified code change.


### 2026-10-22 — ADR0049 Headway analytic anchors validated in real trajectories

- **PASS — Full [Actions run 39809623180](https://github.com/monacofj/moeabench/actions/runs/39809623180):** `313/313` unit tests and `24/24` basic integration tests passed; direct CLI `validate_headway_anchors.py` validated independent analytical controls plus two complete DTLZ1/NSGA-II trajectories (population 32, 20 generations, seeds 7 and 8) with the registered normalized GT and genuine matching generation-zero populations.
- **INDEPENDENT CONTROL NUMBERS:** Headway `0 → Q=1.0`, Headway `0.25 → Q=0.6780719051126376`, Headway `1 → Q=0.0`, Headway `2 → Q=0.0`; missing P0 produced undefined (NaN) both physically and clinically, not a synthetic zero or one. Analytic degenerate-initial (GD95(P0)=0) behavior matched ADR 0049 exactly.
- **INDEPENDENT REAL TRAJECTORIES:** seed `7`: Headway `0.4776167657885824`, Q_HEADWAY `0.4367278589251462`; seed `8`: Headway `0.5997337234010647`, Q_HEADWAY `0.32216821232392656`. Each physical metric matched its independent cKDTree GD95 ratio, and Q matched the independent log-linear mapping. The CI artifact `fair-75-headway-adr0049-anchor-validation` includes SHA256 digests of the actual P0, Pf and calibrated GT for auditing.
- **SCIENTIFIC DECISION — No empirical random-anchor needed:** ADR 0049 and the existing runtime `q_headway()` already specify analytic anchors 0 and 1. The historic offline `headway.rand50/rand_ecdf` and sidecar pseudo-calibration were scientifically incompatible with that definition; prospective generators now store only transparent analytic 0/1 metadata rather than inventing a longitudinal ECDF from a single population. This preserves runtime Q behavior, including clipping for real regressions above 1, and does not modify frozen historical baseline JSON.
- **ADDED FURTHER RELEASE GATE:** following the green run, added a direct unit test of `calibration/scripts/generate_baselines.py::generate_ecdf_for_problem` using a small 3-objective GT and reduced draw counts, asserting its actual emitted HEADWAY record has analytic anchors and *no* random ECDF, while all five other FAIR records remain finite. This extra test is not included in run 39809623180 and requires fresh CI validation before declaring the branch latest-state fully green.
- **RELEASE STATUS:** baseline v0.14.0 and the published `audit_report` are intact, ADR 0051 remains Proposed, PR #76 remains Draft. A change to the prospective baseline record schema is documented and should be checked in the next full integration pass before release.


### 2026-10-22 — Final offline Headway generator contract regression passed

- **PASS — [Actions run 39810162946](https://github.com/monacofj/moeabench/actions/runs/39810162946):** `314/314` unit tests and `24/24` basic integration tests passed. A new direct test called `generate_ecdf_for_problem` (the actual **offline full baseline record writer**) with a deterministic small 3-objective GT and bounded draws, confirming its HEADWAY record now has precisely ADR0049 analytic `ideal=0` and `no_progress=1` anchors plus explicit `no_empirical_longitudinal_baseline` provenance, **without** legacy `rand50`/`rand_ecdf`. The other five metric records remained finite.
- **PASS — Full validation repeated:** the script independently rechecked both mathematical Q/Headway anchors and genuine seed-7/seed-8 NSGA-II P0/Pf trajectories, plus prior DTLZ1 Example 13, three-MOP Closeness-only migration and frozen-vs-staged baseline comparisons. The same-seed final front and Q-Closeness continued to repeat exactly.
- **SCIENCE STATUS:** this closes the specific error in *how prospective Headway baselines are created*. Q_HEADWAY itself was not altered: the accepted ADR0049 analytic log-linear scoring is preserved. It does **not** certify the frozen versioned baseline's existing historical `headway` record, which retains old pseudo-calibration data for historical reproducibility only. The new analytic record is a deliberate forward schema change requiring documentation/consumer compatibility review before release.
- **STILL OPEN:** clinical calibration for other five FAIR quantities beyond the staged DTLZ1/2/5 corrections, resolving historical reports with missing P0, integrating the new report preview into a versioned release, and retiring the branch-only workflow after tests are permanent. PR #76 remains Draft; no certified baseline or audit_report alias was overwritten.


### 2026-10-22 — Downstream Headway marker compatibility and all-MOP analytical screening

- **COMPATIBILITY AUDIT:** source searches found that production `q_headway()` already uses ADR0049 analytic 0/1 anchors and does **not** require stored `headway.rand50` or `headway.rand_ecdf`. The obsolete `calibration/scripts/audit_calibration.py` *does* consult these deprecated fields but is already designated scientifically invalid/unfit for regenerating a certified report. A second **live** consumer was identified: `q_headway_points()`, used to classify pointwise visualization diagnostics, still borrowed legacy `headway.rand50` and converted it with `s_K`, thus incompatible with ADR0049 and new sidecar metadata.
- **FIX — Pointwise Headway contributions:** `moeabench/diagnostics/qscore.py::q_headway_points` now scales each point's nearest-GT error by the actual `GD95(P_initial→GT)` from the same diagnostic context, then applies the same log-linear analytic 0/1 score anchors. This is documented as a **pointwise contribution**, not a replacement definition of population HEADWAY. Missing genuine P0 yields an array of NaNs, not an invented random-population baseline; the ADR0049 zero-initial-error special case is handled. No `headway.rand50`/ECDF lookup remains.
- **INDEPENDENT REGRESSIONS:** `tests/unit/test_headway_pointwise_issue75.py` validates homogeneous partial progress (Headway=0.25), heterogeneous per-point ratios, zero-error P0 and missing-P0 behavior; monkeypatches both baseline lookup APIs to throw, proving the visual marker path never uses the obsolete stored HEADWAY records.
- **ALL-MOP SCREENING:** `calibration/scripts/screen_analytical_residues.py` inventories the 14 registered frozen calibration packages, requires exact baseline↔packaged GT identity, tests for a non-overlapping analytical ideal sample, and measures positive finite epsilon at K=20 and K=100. Unsupported MOPs are named with explicit failure reasons; they are **not** assigned an invented correction. The numerical local-manifold compatibility check is only a screening filter, not scientific proof of equivalent sampling distributions or a baseline certification. No baseline candidate is written by this inventory.
- **TESTS:** `tests/unit/test_screen_analytical_residues_issue75.py` uses synthetic GT packages to independently prove that a supported analytical sample is accepted while missing/mismatched GT or unsupported analytical PF remain explicit failures, never silent successes.
- **CI GATE:** the branch investigation workflow now includes both test modules and an independent 14-MOP inventory artifact `fair-75-all-mops-analytical-screen`. Results have not yet been observed at the time of this entry.
- **INVARIANTS:** the published baseline file and `calibration/audit_report.html` are still untouched. PR #76 stays Draft, ADR 0051 stays Proposed.


### 2026-10-22 — Entire 14-MOP inventory and post-Headway consumer compatibility

- **PASS — [Actions run 39847454128](https://github.com/monacofj/moeabench/actions/runs/39847454128):** `320/320` unit and `24/24` basic integration tests passed, including all new q_headway_points and synthetic screening tests, real DTLZ1/NSGA-II same-seed replay and three-MOP full-K migration.
- **SCREENING RESULT — 11/14 analytically compatible (not yet certification):** independent non-overlapping analytical PF points were found for `DPF1`, `DPF2`, `DPF4`, `DTLZ1`, `DTLZ2`, `DTLZ3`, `DTLZ4`, `DTLZ5`, `DTLZ6`, `DTLZ7`, `DTLZ9`. Their frozen baseline GT matched the corresponding packaged GT exactly and the screening-stage independent nearest-GT residues were finite and positive at both K=20 and K=100. Full per-problem values and GT hashes are in CI artifact `fair-75-all-mops-analytical-screen`. This is **screening**, not an exhaustive proof of unbiased analytical front sampling.
- **UNSUPPORTED — 3/14:** `DPF3`, `DPF5` and `DTLZ8` did **not** have an independently verified analytical PF in the current frozen-package screening. The failure result is `No independently verified analytical PF for archived GT`. These remain *uncorrected and explicitly unsupported* for an independent-residue migration, never assigned invented zeros and not silently included in certified Q-score baselines.
- **FULL-K MIGRATION EXPERIMENT NOW IN CI:** candidate Closeness-only migration and independent full-JSON verification were extended to all **11 screened MOPs**, preserving the remaining 3 intact. `calibration/scripts/verify_closeness_migration.py` requires all other metrics, GT references and metadata to remain numerically equal and independently replays the ECDF and anchor corrections. This larger migration remains `/tmp`-only; workflow [39848065197](https://github.com/monacofj/moeabench/actions/runs/39848065197) began while this note was prepared.
- **RUNTIME COMPATIBILITY — q_headway_points:** prior q_headway_points borrowed legacy `headway.rand50`. It now uses true-P0 GD95 as denominator and ADR0049 analytic anchors, and returns NaN without P0. The fixed logic and four independent tests passed in run 39847454128. Documentation in `docs/fair_metrics.md`, `docs/reference.md` and `docs/userguide.md` now explicitly differentiates a per-point contribution from aggregate HEADWAY and forbids fabricated random baselines.
- **EXPLICIT LEGACY FAIL-CLOSED:** `baselines.get_baseline_values/get_baseline_ecdf` now detect *new* `analytic_anchors_adr0049` Headway records and raise an informative `UndefinedBaselineError` instead of silently treating missing legacy `rand50/rand_ecdf` keys as calibration data. The historical frozen-v0.14.0 records remain structurally untouched for reproducibility; `q_headway` consumes only the analytic mapping and does not access these records. Regression tests verify both fail-closed behavior and availability of unrelated metrics. These last two reader-compatibility changes still await the result of workflow [39848192493](https://github.com/monacofj/moeabench/actions/runs/39848192493) at this checkpoint.
- **REPORT STATUS:** no canonical baseline or `calibration/audit_report.html` mutation; only branch code, separate tests, docs and GitHub Actions artifacts. ADR remains Proposed / PR #76 Draft.


### 2026-10-22 — Verified 11-MOP, 484-K Closeness-only migration and Headway-reader compatibility

- **PASS — Full latest [GitHub Actions run 39848192493](https://github.com/monacofj/moeabench/actions/runs/39848192493):** `321/321` unit regressions, `24/24` basic integration tests, independent full-K migration verification, DTLZ1 Example 13 actual optimization, clinical plot geometry, and exact same-seed final-front / Q-Closeness replay all completed successfully.
- **PASS — 484 individual K records (11 × 44):** the complete staged correction covered `DPF1`, `DPF2`, `DPF4`, `DTLZ1`, `DTLZ2`, `DTLZ3`, `DTLZ4`, `DTLZ5`, `DTLZ6`, `DTLZ7` and `DTLZ9`. Independent `verify_closeness_migration` reported `PASS_PROVISIONAL_CLOSENESS_ONLY`, confirmed exactly `44` corrected K records in **each** selected MOP and `untouched=3` other problems. All other metrics and baseline reference arrays/metadata were required equal to their frozen v0.14.0 values. The fully corrected *candidate* was written only to `/tmp` during the CI job, and the signed numerical verification + impact reports are in artifact `fair-75-eleven-mop-full-k-verification`. This is a mathematically verified **deterministic correction of frozen samples**, not a regenerated/certified full six-metric clinical baseline.
- **PASS — Headway-reader backward compatibility gate:** `tests/unit/test_headway_anchors_issue75.py::test_new_analytic_headway_record_refuses_legacy_rand50_api` passed. New analytical HEADWAY records intentionally have no empirical `rand50/rand_ecdf`. Generic `get_baseline_values` / `get_baseline_ecdf` requests for such records now fail with explicit ADR0049 `UndefinedBaselineError` explanations rather than consuming undefined values. `q_headway` itself remains anchored analytically at 0/1; other five metric readers remain unaffected. The historical frozen-baseline JSON and older legacy records are not migrated or modified in place.
- **UNSUPPORTED PROBLEMS:** `DPF3`, `DPF5` and `DTLZ8` are still not cleared for analytical `ideal_res` by the current frozen-package screening. Source inspection suggests possible sampler limitations requiring independent investigation: DPF3 filters Pareto-front candidates by nondomination and may yield sample sizes different from requested counts; DPF5 uses a grid Pareto-set sampler where fresh point identity and historical-front compatibility are nontrivial; DTLZ8 preferentially uses a fixed high-fidelity Pareto-set file, which may prevent construction of an independent new sample. These are **hypotheses from source inspection**, not verified explanations of each failure, and must not be silently "fixed" by relaxing GT compatibility checks. The 3 corresponding frozen records were unchanged in the staged candidate.
- **RELEASE GATE:** even a passing 484-K candidate may not be published: the analytical compatibility check is a screening test; the 3 MOPs lack independent residue calibration; revised HEADWAY analytic metadata is a forward baseline schema distinction; and the definitive, scientifically complete `audit_report` requires its own source-data/release protocol. PR #76 remains Draft, ADR 0051 remains Proposed, baseline aliases and published HTML remain intact.


### 2026-10-22 — Unsupported MOP root-cause diagnostics initiated

- **MOTIVATION:** the current analytical screening accepted 11 MOPs but withheld Closeness residue correction from `DPF3`, `DPF5` and `DTLZ8`. Source inspection suggests different plausible causes; an absent independent reference does *not* by itself prove that the MOP has no analytical front. Diagnosing this matters before deciding whether to regenerate missing GTs or intentionally leave some Q-scores uncertified.
- **NEW DATA-ONLY PROBE:** `calibration/scripts/diagnose_unsupported_pf.py` loads each archived calibration GT and ideal/nadir transform, invokes the current MOP's `pf()` with the archived requested count and with an extra 1024 points, records requested-vs-returned cardinalities, exact/near-exact overlaps with archived GT, nearest-GT median/p95 mismatch relative to archived nearest-neighbor spacing, elapsed time and exceptions. This is **not** a replacement GT generator, and it cannot modify baselines/reports or relax the analytical compatibility screen.
- **SOURCE HYPOTHESES TO CHECK:** `DPF3.pf()` applies non-dominated filtering and may return a different point count than requested; `DPF5.ps()` uses a floor-square-grid size for M=3 and its sampler geometry can differ between requested counts; `DTLZ8.ps()` preferentially reads a static high-fidelity Pareto-set file and may return the same points even when requesting an extension. These statements are current-code observations, not validated root causes until measured on the archived GT package.
- **INDEPENDENT UNIT TESTS:** validate that a missing calibration package fails explicitly and that a known continuous toy analytical front reports non-overlapping candidate points and correct cardinalities.
- **CI PENDING:** [Actions run 39849133507](https://github.com/monacofj/moeabench/actions/runs/39849133507) runs the full branch validations followed by an evidence-only three-problem sampler diagnosis, uploading artifact `fair-75-unsupported-pf-diagnostics`. No success, failure or scientific interpretation is inferred before the results are inspected.


### 2026-10-22 — Root causes measured for DPF3, DPF5 and DTLZ8, and conservative grid-sampler acceptance

- **PASS — Evidence-only [Actions run 39849133507](https://github.com/monacofj/moeabench/actions/runs/39849133507):** `323/323` unit and `24/24` basic integration tests passed. The new source-data diagnostic recorded actual requested/returned analytical PF sizes, GT overlaps and nearest-GT mismatch in units of the frozen GT's local nearest-neighbor spacing. Numerical evidence is in artifact `fair-75-unsupported-pf-diagnostics`.
- **DPF3 — REJECTION REQUIRES MORE THAN CARDINALITY:** archived GT shape `(1464,3)`; current sampler returned `243` points for 1464 requested, and `407` for an extended request of 2488. Median nearest-GT mismatch was only `0.0604` and `0.0791` local spacings, **but p95 mismatch was `201.24` and `221.56` local spacings**. A median-only compatibility criterion could therefore mistakenly approve the drastically mismatched tail despite a plausible center. Keep this analytical sample **unsupported** until a stronger explanation of the archived-vs-current PF geometry is found.
- **DPF5 — A NORMAL GRID CARDINALITY EFFECT:** archived GT shape `(3367,3)`; current grid sampler returned `3364` points for 3367 requested and `4356` for 4391 requested. Its median mismatch was `0.326` and `0.311` local spacings and its p95 was only `0.562` and `0.526`, respectively; extended sampling yielded `4288` fresh points beyond the stored GT. Exact returned cardinality is not a scientific requirement for a valid analytical surface sample. DPF5 merits **screening-level consideration**, not automatic publication.
- **DTLZ8 — TRUE NONINDEPENDENCE OF THE CURRENT SAMPLER:** archived GT shape `(1000,3)`; both a request for 1000 points and a request for 2024 returned the **same 1000 points**. All 1000 coincide with frozen GT, giving `0` independent points. A different sampler protocol is necessary; do not treat repeated static PF content as a fresh analytical reference.
- **CODE CHANGE — Conditional robust compatibility for variable-cardinality PF samplers:** `_analytical_ideal_from_package` no longer rejects samples solely because their cardinality differs from the archived GT; it first demands finite matching objective dimensions, adequate sample size, and median geometric agreement, and **additionally demands p95 nearest-GT mismatch <= 5 archival median-NN spacings whenever returned cardinality differs**. This should admit a DPF5-style compatible grid while rejecting DPF3's extreme tail, and should still reject DTLZ8 if there are zero fresh points. The 5-spacing tolerance is a conservative **screening heuristic**, not a theorem or full-front equivalence certificate. A synthetic independent test covers both acceptable grid rounding and pathological off-front tails.
- **FRESH CI REQUIRED:** this gate was introduced *after* run 39849133507. Follow-up workflows [39849834587](https://github.com/monacofj/moeabench/actions/runs/39849834587) and [39849996192](https://github.com/monacofj/moeabench/actions/runs/39849996192) are in progress; the actual new supported-MOP count and all regressions must be verified before any staged candidate is expanded. Frozen calibrated artifacts are unchanged.


### 2026-10-22 — DPF5 recovered with robust geometry; screened coverage rises from 11/14 to 12/14

- **PASS — Full [Actions run 39849834587](https://github.com/monacofj/moeabench/actions/runs/39849834587):** `323/323` unit + `24/24` basic integration tests, prior full-K Closeness comparisons and real DTLZ1 Example 13 replay passed with the revised variable-cardinality analytical sampler eligibility check.
- **SCREENING NOW 12 SUPPORTED / 2 UNSUPPORTED:** `DPF5` qualifies as a **screening-level** analytical reference despite grid-size rounding. Archived GT `3367` points; current sampler's `3364` same-count-requested points had p95 mismatch `0.5621` local GT-neighbor spacings and its extended `4356` points had p95 `0.5260`, with `4288` fresh points independent of the archived GT. Its independently derived provisional residues are `K=20 epsilon=0.02037207393558536` and `K=100 epsilon=0.053362698905108405`. By comparison, `DPF3` is still rejected for >200 local-neighbor-spacing mismatch at p95, while `DTLZ8` offers zero fresh analytical points from its static lookup.
- **SOURCE CHANGE:** `_analytical_ideal_from_package` no longer demands equal sample *size* when returned points have matching objective dimensionality and close agreement with archived GT both in median and 95th percentile. The stronger tail rule is applied to the variable-cardinality case rather than relaxing the overall geometric gate. The synthetic test `test_unequal_sample_cardinality_accepted_only_with_compatible_tail` covers both DPF5-like acceptable grid rounding and a deliberately inconsistent far-away tail.
- **CI EXPANSION:** [workflow after commit 6b630f72](https://github.com/monacofj/moeabench/actions) now stages and independently verifies **12 × 44 = 528** Closeness K records, with the remaining **2 MOPs** required completely unchanged. The staged v0.14.0 candidate stays in `/tmp`, never as the canonical installed baseline. This larger migration still requires its own successful run before claiming all 528 checks passed.
- **CERTIFICATION LIMIT:** the analytical manifold proximity heuristic and fresh-sample independence are necessary checks but not a guarantee of statistically representative Pareto-front sampling; Q-score scientific acceptance and the definitive report remain separate gates.


### 2026-10-22 — 528-K, 12-MOP correction verified with DPF5 and unchanged unsupported MOPs

- **PASS — Full [Actions run 39850449482](https://github.com/monacofj/moeabench/actions/runs/39850449482):** `324/324` unit regressions, `24/24` basic integration tests, 14-MOP independent analytical screening, true-P0 Headway tests, existing three-MOP cost benchmark, real DTLZ1 Example 13 clinical computation/visual markers and exact same-seed replay all passed.
- **PASS — Independent 12 × 44 = 528-K migration verification:** screened-compatible problems `DPF1`, `DPF2`, `DPF4`, `DPF5`, `DTLZ1`, `DTLZ2`, `DTLZ3`, `DTLZ4`, `DTLZ5`, `DTLZ6`, `DTLZ7`, `DTLZ9` each yielded exactly `44` independently checked corrected Closeness records. `verify_closeness_migration` returned `PASS_PROVISIONAL_CLOSENESS_ONLY` with `untouched=2`: `DPF3` and `DTLZ8` stay unchanged. All other physical/clinical baseline records, entire registered GTs, problem collection and metadata were required equal to frozen v0.14.0 data.
- **ARTIFACT PROVENANCE:** independent numerical correction results and verification in Actions artifact `fair-75-twelve-mop-full-k-verification` (artifact ID `12456952561`); current sampler investigation in `fair-75-unsupported-pf-diagnostics` (ID `12456424402`). Full candidate baseline was created in CI `/tmp` only, not committed or installed as a default reference.
- **SCIENTIFIC DISPOSITION:** DPF5 passed a *conservative screen* for fresh analytical points despite grid-cardinality rounding. DPF3 remains unsupported due the very large p95 mismatches relative to archived GT NN spacing despite a deceptively small median. DTLZ8 repeats exactly the frozen 1000-point static PF when requesting more points and provides no independent sample. Neither should receive an invented `ideal_res` or an unqualified Q-score certification.
- **KNOWN LIMITATIONS:** The 528 corrected records are a provisional deterministic finite-GT residue correction of legacy Closeness ECDF/anchors—not a full regeneration of other metrics or a new certified six-metric baseline. The screening threshold does not prove representativeness of fresh analytical samples across the whole Pareto surface. The historical `audit_report` cannot be reissued as scientifically complete while true archived P0 populations are absent. Prospective ADR0049 HEADWAY anchors and its reader compatibility have separate passing tests.
- **RELEASE/REVIEW STATUS:** ADR 0051 remains Proposed and PR #76 Draft; no merge, version bump, or change to the canonical v0.14.0 baselines, audit JSON or `calibration/audit_report.html` has been made.


### 2026-10-22 — Scientific challenge to ordered-prefix independent ideal_res estimation

- **RISK IDENTIFIED:** `independent_analytical_reference()` currently returns the **first** `n_fresh=1024` non-GT points from a deterministic analytical `pf(n_points=len(GT)+1024)` request. DPF5 uses an ordered 2D grid, so this prefix may concentrate in one region of the Pareto surface. For DTLZ1/2 the `pf` implementation resets a deterministic random generator, so repeating the same call with a different external seed does **not** constitute independent replication. A positive, geometry-compatible residue alone does not prove it is representative.
- **NEW NONCERTIFIED SENSITIVITY STUDY:** `calibration/scripts/study_ideal_residue_robustness.py` challenges DTLZ1 (simplex), DTLZ2 (hypersphere) and DPF5 (grid). It obtains **one enlarged analytical candidate pool**, keeps **all** non-overlapping fresh points rather than truncating it, and computes four **disjoint random subsamples** of sizes 256, 512, and 1024. This design measures **partition/subsampling sensitivity of a single deterministic analytical reservoir**, **not** sampling variance across independent analytical generators. The study explicitly states this limitation. For each sample it records (a) `ideal_res` at K=20 and K=100 in FPS-scale units, (b) five-bin objective-1 regional occupancy, (c) nearest-GT mismatch and archived GT-to-sample coverage, and (d) Q_CLOSENESS computed by the *actual runtime scoring code*, using fixed synthetic observation distributions (0.35× and 0.65× the frozen v0.14.0 normal-blur ECDF) and applying **the same candidate residue** to both observations and reference ECDF.
- **CONTROL — Production prefix as comparator:** independently reproduces the existing first-1024 algorithm and checks whether its `epsilon` lies within the range of four random disjoint 1024-point subsets from the enlarged reservoir. Records relative variance, Q-score envelope and explicitly raises warning flags **without arbitrarily declaring calibration scientifically certified**. Frozen GT arrays, baseline JSON and official audit reports are read-only; an isolated CI JSON holds all measured data.
- **INDEPENDENT UNIT TESTS:** `tests/unit/test_ideal_residue_robustness_issue75.py` provides a controlled synthetic 3D manifold with an intentionally spatially biased ordered prefix; verifies deterministic repeatability, five-region occupancy, real Q-closeness sensitivity, non-overlapping candidate requirements and fail-closed insufficient sample size. The CI runs the focused tests before the full unit regression suite.
- **CI PENDING:** new [GitHub Actions run after commit 4a5a4059](https://github.com/monacofj/moeabench/actions) must pass before any numerical claims for DTLZ1/2/DPF5. The earlier 528-K migration is **mathematically verified but its representativeness is unproven**; do not release on its numerical success alone.


### 2026-10-22 — First robustness challenge exposes estimator dependence; Q protocol corrected

- **PASS — First [Actions run 39867962980](https://github.com/monacofj/moeabench/actions/runs/39867962980):** the new analytically controlled prefix-bias unit tests passed, as did `328/328` unit and `24/24` basic integration suites. The 3-MOP robustness study ran on enlarged fresh analytical PF reservoirs; it produced independent disjoint-subset comparisons without modifying historical frozen calibration files.
- **RESULT — Ordered-prefix `ideal_res` outside all four N=1024 replica estimates at both K=20 and K=100 for DTLZ1, DTLZ2, DPF5.** At **K=100** (physical FPS units), legacy prefix vs median of disjoint subsets: DTLZ1 `0.0411380781 vs 0.0439991444` (subset range `0.043034602..0.046621690`, relative range 8.15% of median); DTLZ2 `0.0541590625 vs 0.0546697767` (range `0.054513723..0.055570554`, 1.93%); DPF5 `0.0533626989 vs 0.0527203822` (range `0.051024338..0.052923799`, 3.60%). This is a **sensitivity signal, not proof of statistical bias or a new accepted residue estimate**, because the four disjoint subsets come from one deterministic analytical pool. No publication or baseline swap is authorized.
- **INVALID Q sensitivity comparison in that initial run:** the first implementation inadvertently used the **frozen K=20 Closeness ECDF for both K=20 and K=100** when calculating candidate Q values. Therefore the run's Q ranges must **not** be used as quantitative K=100 quality evidence, even though its epsilon/residue comparisons are valid. This scientific mismatch was identified and corrected before drawing Q conclusions.
- **FAILED CI (preserved):** [run 39868124192](https://github.com/monacofj/moeabench/actions/runs/39868124192) failed the new focused test while the code expected a dictionary of K-specific Closeness ECDFs but the synthetic test still supplied a single array. This is a test fixture contract mismatch; it did not expose a numerical regression in production. The tests now supply separate K=20/K=100 reference distributions and include a fail-closed missing-K case.
- **CHANGE — K-matched Q inputs:** the robustness script now reads `rec[str(K)]['closeness']['rand_ecdf']` for each K independently, and `_fixed_q` uses the *real* runtime `qscore.q_closeness` function against the correctly residue-adjusted ECDF, with the exact same correction applied to its frozen synthetic evaluation distribution. `tests/unit/test_ideal_residue_robustness_issue75.py` was updated to verify K-wise reference selection and reject absent calibrated K. The last CI run [39868152173](https://github.com/monacofj/moeabench/actions/runs/39868152173) is pending as of this entry.
- **PROTOCOL LIMITATION:** these are randomized partitions of one fixed deterministic sampler-generated pool; they are **not independent analytical draws with different PF-generator seeds**. They can reveal ordered-prefix artefacts and within-pool variance but cannot independently establish that any sample is representative of the full true Pareto front. For DPF5 the reservoir must use *all* fresh candidates from the larger PF call, not a second first-N prefix; that correction is included.


### 2026-10-22 — Corrected K-specific Q-score sensitivity completed on three MOPs; full screened battery initiated

- **PASS — Corrected [Actions run 39868152173](https://github.com/monacofj/moeabench/actions/runs/39868152173):** `329/329` unit and `24/24` basic integration tests, the complete 12-MOP provisional Closeness migration, genuine-P0 longitudinal Headway, and same-seed Example 13 replay passed. Unlike the first robustness attempt, the Q experiments were now calibrated against **each K's own frozen ECDF**, not against a shared K=20 reference.
- **K=100 RESIDUE RANGES — four disjoint N=1024 reservoir subsets:** DTLZ1 `0.043034602..0.046621690` (median `0.043999144`; 8.15% range/median), DTLZ2 `0.054513723..0.055570554` (median `0.054669777`; 1.93%), DPF5 `0.051024338..0.052923799` (median `0.052720382`; 3.60%). The current ordered first-1024 estimator gave `0.041138078`, `0.054159062`, `0.053362699`, respectively, **outside all four same-size subset ranges for each problem**. This is evidence of sensitivity to analytical subset selection, not sufficient statistical proof of an unbiased alternative.
- **K-MATCHED Q_CLOSENESS — fixed synthetic observations scaled from each frozen K-specific blur ECDF:** DTLZ1 N=1024: Q range `0.661468169..0.662419780` at factor 0.35, `0.356223278..0.356745848` at factor 0.65. DTLZ2: `0.664507112..0.664786897` and `0.357887881..0.358042348`. DPF5: `0.663334002..0.663825256` and `0.357264785..0.357534525`. Largest spread for these two controlled evaluations was below `0.001` absolute Q units. This **does not** guarantee stable Q-score for arbitrary actual MOEA fronts, different seeds/generators, or near clipping thresholds.
- **PERFORMANCE/INDEPENDENCE CAVEAT:** the four replicate subsets are disjoint *within a sample-size block* but originate from one deterministic larger `pf()` result; they must not be presented as independent PF generator draws or given unearned statistical confidence intervals. Samples at different sizes reuse points. The enlarged DPF5 candidate pool retains all fresh points to prevent a second ordered-prefix truncation.
- **EXPANSION — [Actions run 39868719932](https://github.com/monacofj/moeabench/actions/runs/39868719932):** a dedicated separate GitHub Actions job now exercises the same challenge on **all 12 analytically screened MOPs**, instead of re-running the 12-MOP analysis only after the heavy historic/report suite. It retains `fair-75-all-screened-residue-robustness` as a separate noncertified JSON artifact and prints region-coverage counts for the first-1024 prefix and the complete analytical pool. Both strong and weak findings must be recorded before proposing any production change to the estimator.
- **RELEASE GATE:** current 528-record migration remains *provisional*; the core sampler still uses a first-N prefix. Neither baseline v0.14.0 nor any official audit report was changed. ADR 0051 Proposed / PR #76 Draft.


### 2026-10-22 — Complete 12-MOP challenge exposed severe PF ordering bias; new deterministic reservoir policy

- **PASS — Dedicated 12-MOP robustness job, [Actions run 39868719932](https://github.com/monacofj/moeabench/actions/runs/39868719932):** all 12 screened analytical reference MOPs produced at least 4 disjoint 1024-point subsamples from a larger deterministic analytical reservoir. All diagnostic Q evaluations used **K-specific frozen baseline ECDFs** and the same physical residue correction for fixed observations and their reference ECDF. The job completed successfully and retained `fair-75-all-screened-residue-robustness` for full numerical traceability.
- **NEW CONCRETE FAILURE — Regional concentration of the production first-1024 estimator:** archived GT first-objective quantile bins (5 equally populated GT regions) received counts `[1024,0,0,0,0]` for each of **DPF1, DPF2, DPF4** and `[0,0,0,0,1024]` for **DTLZ9**, whereas the full enlarged analytical reservoirs occupied all five bins (`~3637` candidate points per bin in these four cases). The original estimator was *not representative of the returned candidate surface* in those MOPs. DPF5's prefix and reservoir also had unbalanced occupancies, though less extreme. This is a stronger defect than ordinary finite-sample variation, and cannot be repaired by changing a threshold or relabeling Q.
- **N=1024 K=100 notable old-prefix vs random-subset medians:** DPF2 `0.00141631→0.00186025`; DPF4 `0.00304053→0.00210644`; DTLZ9 `0.00143849→0.00201599` in addition to DTLZ1/2/DPF5 values recorded above. These are *observed within-pool sensitivity measurements*, **not certified new baseline values**. Q score changes under fixed scaled-blur synthetic observations stayed relatively small but cannot guarantee stability near Q classification boundaries or on actual MOEA populations.
- **PRODUCTION FIX — replace ordered prefix with deterministic whole-reservoir sampling:** `moeabench/diagnostics/calibration.independent_analytical_reference` now requests `len(GT)+4×n_fresh` analytical points, excludes all archived GT coincidences, removes exact repeated candidate objective vectors, then samples `n_fresh` distinct points **without replacement** using a *private* `numpy.default_rng(sampling_seed=123)`. It returns unsupported when insufficient genuinely new points exist. This selection is repeatable without mutating either global Python or NumPy random state. The policy directly addresses the verified ordering counterexamples, but still **does not establish** that the *underlying* analytical PF sampler is unbiased.
- **LEGACY CONTROL PRESERVED IN ROBUSTNESS STUDY:** `study_ideal_residue_robustness.py` reconstructs the historical first-1024 selection explicitly even after the runtime helper's policy changes; it separately reports the new reservoir policy's epsilon and region counts. Otherwise the experiment would quietly compare the corrected algorithm to itself and lose reproducibility of its original negative finding.
- **NEW INDEPENDENT TESTS:** an objective-sorted synthetic analytical grid must yield a deterministic new 250-point sample with occupancy of all five front regions; different private sampling seeds must change the sample; neither module-global RNG state may change; repeated candidate points must not be overrepresented. These tests explicitly fail the former first-N selection algorithm. Earlier independent physical/Q and baseline-preservation tests remain.
- **REVALIDATION PENDING:** these production and test changes landed after the 12-MOP pilot success. Latest focused, unit, integration, regional coverage and candidate baseline migrations must pass under the revised sampler before its corrective numerical outputs can be accepted. Official v0.14.0 baselines, installed aliases and published audit report remain unchanged; PR #76 Draft / ADR 0051 Proposed.
