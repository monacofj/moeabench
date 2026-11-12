# Issue #81 — Sensibilidade do Q-Closeness à compressão da referência desfavorável

**Data:** 12/11/2026. **Branch:** `research/81-dpf5-calibration-protocol`. **PR #86: Draft, sem merge.** Pesquisa não certificada; nenhuma calibração oficial, lei científica, tolerância ou novo formato de persistência aprovado.

## 1. Dados reutilizados e reconstrução

Reutilizamos as **24 calibrações completas e 48 execuções reais** de NSGA-II/SPEA2 da bateria anterior: K40/100/200, oito calibrações por K, 16 execuções por K, quatro sementes e dois orçamentos. Em cada condição há **384 células pareadas população×calibração**; são células correlacionadas, não 384 experimentos independentes. Os quatro métodos e cinco resoluções produzem 7.680 avaliações de Q, também pareadas.

Não houve novas otimizações, ajustes de sigma ou amostragens ideais. A distribuição completa de cada calibração foi reconstruída por replay **apenas do blur final**, usando o sigma, a semente de blur, epsilon, s_K, GT, normalização e código geométrico preservados. A recuperação a partir das 200 amostras isoladamente seria impossível: a fidelidade depende dos parâmetros/sementes e do código/GT preservados.

Conferências executadas:

- Hashes do arquivo original, dos 24 blurs, das 24 ECDFs atuais e das 48 populações conferidos; distâncias de Pf à GT reproduzidas exatamente.
- As 24 ECDFs de 200 valores e seus 384 Q reproduzem exatamente os dados originais.
- Os 384 Q completos reproduzem exatamente a análise anterior.
- Os 61 hashes de arquivos protegidos permaneceram iguais antes/depois.
- Oracle independente do SciPy para todas as condições: discrepância máxima **3,3306690738754696×10⁻¹⁶**; nenhuma divergência de sinal de gaps entre os dois cálculos.
- Os cinco arquivos de dados gerados foram reproduzidos **byte a byte** em uma segunda execução. **74 testes passaram (tempo observado registrado em `tests.log`).** Um teste impede novas otimizações, ajustes e amostragens ideais.

Os vetores completos reconstruídos agora estão preservados em `reconstructed-full-references.npz`, ordenados, com todas as multiplicidades e zeros, sem deduplicação. Os hashes por vetor, parâmetros e sementes estão em `summary.json`; os hashes dos arquivos estão no manifest. Os campos legados de condição única do JSON antigo não foram usados: cada linha fornece algoritmo/orçamento reais.

## 2. Definições precisas

Sejam `z[0]≤...≤z[n−1]`, n=3367, os valores corrigidos `max(0,d(blur,GT)/s_K−epsilon)` de **uma calibração**. Cada valor completo tem massa 1/n. As representações reduzidas têm m valores, cada um com massa 1/m; nenhum peso foi aprendido das populações dos MOEAs.

| Método | Definição, j=0,...,m−1 | Justificativa |
|---|---|---|
| Atual, compatibilidade | `z[floor(j(n−1)/(m−1))]`; implementação exata `linspace(...).astype(int)` | Reproduzir o produtor atual, incluindo os extremos. |
| Extremos lineares | Interpolação linear dos índices em `h=j(n−1)/(m−1)` | Controle do efeito do truncamento de índices; equivale à interpolação de quantis tipo 7 na malha de extremos. |
| Quantis centrais | `Q_emp((j+1/2)/m)=z[ceil(n(j+1/2)/m)−1]` | Mediana da função quantil em cada intervalo de probabilidade; minimiza W1 para m átomos de pesos iguais. Índices calculados com aritmética inteira. |
| Médias por intervalo | `m ∫_[j/m,(j+1)/m] Q_emp(p) dp` | Constante que minimiza o erro quadrático da função quantil em cada intervalo; preserva sua média. Integral calculada pelas sobreposições exatas das células empíricas. |

A justificativa W1 decorre de `W1=∫|Q_emp(p)−Q_reduzida(p)|dp`: para cada intervalo de massa 1/m, a constante que minimiza a integral absoluta é uma mediana. Para a integral quadrática, o minimizador é a média. Essas propriedades não garantem uma ordenação idêntica de erros de Q em todas as populações: Q é uma função das distâncias e da referência ideal.

Epsilon permanece **exatamente o da calibração original em todas as resoluções**; sigma e os pontos negativos também. A referência completa usa os mesmos 3367 valores, sem agregação entre calibrações. **m=3367 e distribuição completa são a mesma lei**, uma condição de identidade, não evidências independentes.

O mapeamento Q, inclusive o gate `d_i≥delta → Q=0`, foi mantido. Fora do gate, `Q=d_r/(d_i+d_r)`, onde `d_i=W1(pop_corrigida,0)`, `d_r=W1(pop_corrigida,referência)` e `delta=W1(0,referência)`. A interpretação dos valores como amostras de pesos iguais também foi mantida.

## 3. Qual erro é introduzido pelos 200 valores atuais?

O erro é `Q_m−Q_completo` na **mesma população e mesma calibração**. O erro relativo é `|Q_m−Q_completo|/|Q_completo|`. Não houve Q completo zero neste painel; o script conta explicitamente denominadores zero sem inventar um epsilon para o erro relativo.

| Valores | Erro absoluto médio | P95 absoluto | Erro absoluto máximo | Erro relativo máximo |
|---:|---:|---:|---:|---:|
| 200 | 0.000557437243 | 0.00127627598 | 0.00254904843 | 0.350688952% |
| 400 | 0.000227921872 | 0.000518208289 | 0.00108484487 | 0.149249072% |
| 800 | 6.91160071e-05 | 0.000179977872 | 0.000355674386 | 0.0489324081% |
| 1600 | 1.75648039e-05 | 4.41831672e-05 | 0.00011234428 | 0.0150054748% |
| 3367 | 0 | 0 | 0 | 0% |

O pior caso atual/200 foi K40, NSGA-II G40, semente43, calibração2: **Q completo=0,7268687566860689**, Q comprimido=0,7294178051111964, erro=+0,002549048425127487, relativo=0,35068895198482286%. Isso não deve ser confundido com a mudança candidato-versus-baseline-histórico, que também envolve a revisão das referências e de epsilon.

`summary.json` contém erros assinados, absolutos e relativos, média, RMS, mínimo/máximo e quantis 0/10/25/50/75/90/95/99/100%, globais e por K. `q-by-cell.csv.gz` contém cada avaliação individual.

## 4. Efeito do método de representação

| Método, 200 valores | Erro absoluto médio de Q | Erro absoluto máximo de Q | W1 médio da referência |
|---|---:|---:|---:|
| Atual: índices truncados | 0.000557437243 | 0.00254904843 | 0.0257130312 |
| Extremos + interpolação linear | 0.000585021286 | 0.00268354891 | 0.0261346911 |
| Quantis centrais da ECDF | 4.27739679e-05 | 0.0002552786 | 0.0131980085 |
| Médias por intervalo de probabilidade | 8.09394843e-06 | 0.000229295303 | 0.0135482392 |

A interpolação linear não resolveu o problema: seu erro máximo em 200 valores foi maior que o atual. A inclusão do máximo com massa 1/200, em vez de 1/3367, dá ao maior valor original **16,835 vezes** sua massa original. A regra atual/200 deslocou a média da referência para cima entre **0,01199762 e 0,03081639**, em unidades normalizadas. Quantis centrais evitam impor um átomo na ponta máxima; médias por intervalo preservam a média até arredondamento numérico, desvio máximo observado de 4,44×10⁻¹⁶.

Para fidelidade de **Q neste painel**, médias por intervalo tiveram o menor erro médio em todas as resoluções. Isso não é dominância universal: em m800 o maior erro dos quantis centrais foi **0,0000825658**, menor que o das médias por intervalo (**0,0001645243**). Para fidelidade **W1 da distribuição de referência**, quantis centrais são a escolha matematicamente ótima no conjunto considerado de m átomos com pesos iguais; seu W1 médio em m200 foi 0,01319801, contra 0,01354824 das médias por intervalo.

Preservar a média é útil porque `delta=mean(referência)`, e preserva d_r quando as diferenças completa e comprimida entre funções quantis mantêm o mesmo sinal em cada intervalo integrado. Não garante preservar toda distância W1 nem todo Q: cruzamentos das funções quantis ainda podem produzir erro. Há uma condição especial neste painel: **200/400/800/1600 são múltiplos de K40/K100/K200**, e cada população avaliada tem exatamente K valores. A função quantil da população é, portanto, constante em cada intervalo de compressão. Pela desigualdade de Jensen, `|mean(R_bin)−F_bin| ≤ mean(|R_bin−F_bin|)`: as médias por intervalo não aumentam d_r. Como preservam delta e d_i não muda, também não aumentam Q, salvo arredondamento. Isso explica a direção dos erros observados; não é uma garantia universal de menor erro. Em um contraexemplo testado, população `[0,0.9,1.8,2.7]`, referência `[0,1,2,3]`, m2, o Q sobe de 0,1 para 0,25 sob médias por intervalo, porque o alinhamento falha. Esse é um teste matemático sintético, não uma nova população experimental. As médias por intervalo não foram ajustadas aos MOEAs.

## 5. Evolução com a resolução e ordenações

A tabela abaixo conta células **par×calibração** com inversão estrita em relação à distribuição completa, somando as três grades por K; não há comparações entre K diferentes. Todas pertencem ao mesmo par de K200. Cada condição tem 360 pares dentro dos Ks × 8 calibrações, ou 2880 comparações correlacionadas.

| Método | 200 | 400 | 800 | 1600 | 3367 |
|---|---:|---:|---:|---:|---:|
| Atual: índices truncados | 6 | 2 | 0 | 0 | 0 |
| Extremos + interpolação linear | 6 | 2 | 0 | 0 | 0 |
| Quantis centrais da ECDF | 0 | 0 | 0 | 0 | 0 |
| Médias por intervalo de probabilidade | 0 | 0 | 0 | 0 | 0 |

Com o método atual e com a interpolação linear há **um par que inverte entre calibrações** em m200 e m400; em m800/m1600/m3367 não há inversões entre calibrações nem discordâncias com a completa. Quantis centrais e médias por intervalo não apresentaram essas inversões em nenhuma resolução testada. Não houve empates exatos ou gaps dentro de um ULP em nenhuma condição. O diagnóstico de um ULP é apenas resolução numérica de ponto flutuante, não um limiar científico de equivalência prática.

Par detalhado: **NSGA-II G40 seed19 menos SPEA2 G150 seed19, K200**:

| Método | Valores | Gap mínimo | Gap máximo | Calibrações com ordem diferente da completa |
|---|---:|---:|---:|---:|
| Atual: índices truncados | 200 | -0.000242991827 | 0.000505154673 | 6/8 |
| Atual: índices truncados | 400 | -0.000435099232 | 5.92829709e-05 | 2/8 |
| Atual: índices truncados | 800 | -0.0005266082 | -0.000112374019 | 0/8 |
| Atual: índices truncados | 1600 | -0.000565106921 | -0.000167689649 | 0/8 |
| Atual: índices truncados | 3367 | -0.000567990396 | -0.000168535585 | 0/8 |
| Extremos + interpolação linear | 200 | -0.000242570994 | 0.000504936527 | 6/8 |
| Extremos + interpolação linear | 400 | -0.000431741461 | 6.74647207e-05 | 2/8 |
| Quantis centrais da ECDF | 200 | -0.00061793737 | -0.00018469487 | 0/8 |
| Médias por intervalo de probabilidade | 200 | -0.000567990396 | -0.000168535585 | 0/8 |

As três inversões de K200 em relação ao Q **histórico**, já descritas na ADR 0055, persistem em todos os métodos/resoluções e na distribuição completa. Portanto, não são todas explicáveis pela compressão. Os 120 pares por K, os oito gaps de cada par e os índices das discordâncias estão em `pairwise.json.gz`; os dez pares com menores gaps completos de cada K são destacados no sumário apenas como recurso de inspeção, sem definir significância prática.

O erro máximo global do método atual cai com a resolução, mas a melhora **não é monotônica para toda célula**: em 1, 7 e 19 células o erro aumenta, respectivamente, nas transições 200→400, 400→800 e 800→1600. As grades não são aninhadas e os erros podem mudar de sinal. Nos quantis centrais, o erro máximo global aumenta de 0,0000825658 em m800 para 0,0000877460 em m1600, embora o erro médio caia. Nas médias por intervalo, os pequenos aumentos observados nas transições ficaram no nível de arredondamento; nenhuma célula aumentou além da soma de dois ULPs dos Q comparados. Nenhuma dessas verificações estabelece tolerância de aceitação.

## 6. Separação da variabilidade de calibração e de representação

| K | Amplitude média entre calibrações, distribuição completa | Maior amplitude, completa | Maior amplitude, atual/200 |
|---:|---:|---:|---:|
| 40 | 0.00646573602 | 0.0189230387 | 0.0188294308 |
| 100 | 0.00884103832 | 0.0165935341 | 0.0171740568 |
| 200 | 0.00405780836 | 0.00707644927 | 0.00691201151 |

A distribuição completa foi **estável quanto às ordenações neste painel**, inclusive no par mais próximo, cujo menor gap absoluto foi 0,0001685356. Ela não torna os Q constantes entre calibrações: as amplitudes máximas completas continuam entre 0,00708 e 0,01892 conforme K. Essa estabilidade observada não é prova de estabilidade para novas sementes, outros MOPs ou populações mais próximas.

Para cada população fixa, calculamos `Q_m=Q_completo+e_m` e a identidade descritiva entre as oito calibrações:

```text
Var(Q_m) = Var(Q_completo) + Var(e_m) + 2 Cov(Q_completo,e_m).
```

As variâncias usam divisor 8, sem inferência de amostragem. A tabela mostra médias dessas quantidades sobre as 16 populações de cada K para atual/200:

| K | Média Var(Q completo) | Média Var(erro de representação) | 2 × média Cov(Q completo, erro) | Média Var(Q atual/200) |
|---:|---:|---:|---:|---:|
| 40 | 5.97165279e-06 | 2.28734866e-08 | -3.59331118e-07 | 5.63519516e-06 |
| 100 | 9.43510565e-06 | 4.52831442e-08 | 2.8750124e-07 | 9.76789004e-06 |
| 200 | 2.41448468e-06 | 2.09361647e-08 | 5.03542017e-08 | 2.48577504e-06 |

O termo de covariância pode ser negativo: em K40 a compressão reduz a variabilidade aparente. Portanto, não seria correto atribuir uma porcentagem causal da variância à compressão simplesmente subtraindo variâncias. Nos vetores individuais, a mudança da amplitude entre calibrações causada pela compressão atual/200 variou K40 −0,00009361..−0,00001313; K100 −0,00029163..+0,00058052; K200 −0,00016444..+0,00050397.

A variabilidade das execuções dos otimizadores foi mantida separada: o sumário guarda a amplitude entre as quatro sementes, dentro de algoritmo/K/orçamento, para cada calibração fixa e cada representação. As mesmas populações e calibrações aparecem em todas as condições; sementes entre orçamentos/K são compartilhadas. Não houve novas observações independentes de otimização.

## 7. Recomendação técnica e decisões pendentes

**Recomendação técnica de fidelidade, restrita a este DPF5:** preservar e usar a distribuição empírica completa de **3367 valores como referência de verificação**. Ela elimina exatamente o erro de representação frente ao comparador finito, independentemente de uma tolerância arbitrária. São 26936 bytes de valores float64 por calibração, contra 1600 bytes em m200, antes de metadados/compressão. Os vetores completos já estão preservados nesta evidência.

Há base técnica para propor **3367 valores para produção neste caso**, se armazenamento, desempenho e contrato de persistência forem aprovados. **Não foi medido o custo de produção nem aprovada a alteração de formato/arquitetura.** Essa resolução tampouco elimina a variabilidade da calibração ou garante aproximação suficiente da lei ideal infinita.

Se for necessário manter compressão, **800 é a menor resolução testada da regra atual que reproduziu todas as ordenações completas deste painel**. Isso não certifica sua precisão para produção: ainda houve erro máximo 0,0003556744 e faltam um requisito de precisão ligado ao uso e cobertura de outros casos. m1600 reduz o máximo para 0,0001123443, mas sua suficiência também não foi cientificamente definida. Médias por intervalo e quantis centrais merecem avaliação como alternativas; escolher entre fidelidade de Q observada e objetivo W1 da referência é uma decisão explícita, não algo a aprovar implicitamente pelo painel de MOEAs.

Permanecem para apreciação científica: precisão necessária, tratamento prático de quase empates, método/resolução normativos, agregação entre calibrações e eventual artefato versionado. Não foram criados critérios estatísticos de aceitação. Nenhum resultado declara diferenças pequenas automaticamente relevantes.

## 8. Reprodução e proveniência

```bash
MPLCONFIGDIR=/tmp/issue81-mpl .venv/bin/python \
  calibration/scripts/study_dpf5_ecdf_compression_issue81.py \
  --output-dir /tmp/issue81-compression-new
MPLCONFIGDIR=/tmp/issue81-mpl .venv/bin/python -m pytest \
  tests/unit/test_issue81_ecdf_compression.py \
  tests/unit/test_issue81_full_calibration_q.py \
  tests/unit/test_dpf5_closeness_protocol_issue81.py \
  tests/unit/test_outward_half_normal_blur_issue83.py \
  tests/unit/test_qscore_ecdf.py tests/unit/test_fair_conformance_issue75.py -q
```

Use diretório novo: o script recusa sobrescrita e locais oficiais protegidos. `--evidence` permite indicar a pasta de entrada preservada. Ambiente executado: Python 3.14.4, NumPy 2.5.4, SciPy 1.18.1, MoeaBench 0.16.1. Os hashes do script, do Q-score, do gerador geométrico e do manifest de entrada estão no sumário. A execução de reprodução independente confirmou igualdade byte a byte dos cinco artefatos de dados. A reconstrução em outro ambiente deve conferir os hashes e falhar diante de discrepâncias, sem continuar silenciosamente com referências diferentes.

Erros/limitações registrados: nenhum erro numérico ou falha de teste nesta bateria; a ausência das amostras negativas completas no artefato original exigiu replay do blur final, validado pelos hashes originais. Os dados completos foram agora persistidos. O metadado legado de algoritmo/orçamento do arquivo antigo foi explicitamente desconsiderado em favor das linhas individuais corretas. As 200 amostras isoladamente não permitem recuperar a distribuição completa.

**Preservado:** runtime FAIR, baselines oficiais, GTs congeladas, normalização e `audit_report`. PR #86 permanece Draft; nenhuma calibração oficial foi alterada.

Correção de apresentação antes do commit: o verificador do Git apontou finais CRLF no CSV como whitespace; o exportador passou a usar LF. Os artefatos foram regenerados e sua reprodução byte a byte foi novamente verificada, sem alteração dos resultados numéricos.

Conferência suplementar: uma primeira invocação do verificador de integridade usou o Python do sistema e falhou no import por ausência de `pymoo`, antes de qualquer cálculo. A repetição com `.venv/bin/python` passou, conferindo manifest, fontes, vetores NPZ e todos os 7.680 valores de epsilon. O erro operacional e sua correção estão em `integrity-check.txt`; os experimentos e os 74 testes já haviam sido executados no ambiente correto.
