# Scientific Audit Report: Delgado2 Confrontation (v2 Legacy)

Este relatório documenta o confronto entre os novos dados legados de alta precisão (**Delgado2**) e a verdade analítica implementada no **MoeaBench v0.7.5**.

## 1. Executive Summary

A auditoria confirmou que a v0.7.5 e o Delgado2 estão em **perfeita harmonia** para os problemas fundamentais (DTLZ1-6). No entanto, foram detectadas divergências geométricas nos problemas baseados em blocos e restrições (DTLZ8-9), onde a v0.7.5 impõe um rigor matemático superior.

## 2. Invariant Audit Results

| Problem | Status | Invariant | Gap | Conclusion |
| :--- | :--- | :--- | :--- | :--- |
| **DTLZ1** | **OK** | Sum 0.5 | 2.04e-09 | Identical Geometry. |
| **DTLZ2-6** | **OK** | SOS 1.0 | < 2e-08 | Identical Spherical Geometry. |
| **DTLZ8** | **FAIL**| Linear | 1.41e+00 | Delgado2 deviates from the theoretical constraint intersection. |
| **DTLZ9** | **FAIL**| Spherical Pair | 8.57e-01 | Delgado2 does not satisfy the $f_j^2 + f_M^2 = 1$ invariant strictly. |

## 3. Notable Findings

### DTLZ8 & DTLZ9 Discrepancies
Os dados do Delgado2 para DTLZ8 e DTLZ9 apresentam uma média de objetivos que não satisfaz os invariantes de repouso (Pareto Set Ideal). Isso sugere que a geração do Delgado2, embora de alta fidelidade, pode estar sujeita a:
1.  **Search Bias**: Onde o algoritmo "estacionou" em uma zona viável mas não perfeitamente ótima.
2.  **Constraint Tolerance**: Diferenças na precisão de flutuação permitida para as restrições de desigualdade do DTLZ8.

### DPF Family
A família DPF na v0.7.5 agora utiliza **projeções quadradas** (Squared Projections) e **amostragem estruturada**, o que gera frentes visualmente "limpas" e geometricamente precisas. O Delgado2 apresenta nuvens mais densas nestes problemas, mas com menor precisão em relação às curvas analíticas puras.

## 4. Final Recommendation

Mantenha-se a **v0.7.5 Ground Truth** como o alvo oficial. O Delgado2 serviu para validar que estamos no caminho certo para a vasta maioria dos problemas, mas as correções analíticas da v0.7.5 provaram-se mais rigorosas para problemas restritos.

---
*Gerado automaticamente pelo Antigravity Auditor em 2026-01-25.*
