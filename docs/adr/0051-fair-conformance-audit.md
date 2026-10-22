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
