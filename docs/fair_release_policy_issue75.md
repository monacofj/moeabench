# FAIR calibration release contract — issue #75

**Status: proposed, not an approved normative change.** This document
explains the release gates exposed by PR #76 and ADR 0051. It does **not**
declare the experimental candidate scientifically certified.

## 1. Distinguish library releases from calibration datasets and method revisions

1. **MoeaBench library version** identifies code that consumes and produces
   diagnostics (at the time of the pilot, `0.16.1`).
2. **Source calibration provenance** identifies the baseline dataset and
   environment in which it was originally produced (frozen `0.14.0`).
   This source metadata must **never** be rewritten to match a later consumer.
   The canonical `baselines.json` can intentionally point to a frozen older
   calibration; different version strings alone are not a numerical defect.
3. **Scientific calibration protocol version** identifies the mathematical
   definition of a metric's anchors, including coordinate normalization,
   reference probability measure and estimation method. The proposed
   `uniform_normalized_objective_pf_area_v1` is a **new, provisional method**.
   Its use would require an explicit **new baseline artifact identity** rather
   than silently altering `baselines_v0.14.0.json` or relabeling that file.

**Important:** neither a mismatch between archived producer and consumer
library versions nor a difference in developer-certified dependency versions
automatically invalidates the historical baseline or mandates a new one.
The baseline can be reused when its mathematical definitions, coordinates,
reference GT and numerical behavior remain compatible with the consumer.
For a *new scientific calibration claim*, evidence should separately record
the source and consumer versions, GT and baseline SHA256, normalization,
K-specific ECDF, sampling protocol, dependency environment and actual
diagnostic behavior. These version strings are provenance, not equality gates.

## 2. Evidence currently available

PR #76 has verified: fixed-GT paired real-NSGA-II Q comparisons, a
non-destructive 88-record DTLZ1/DTLZ2 M=3 area-law candidate; unchanged
unrelated FAIR/Q families, and a versioned noncertified report with a true P0
for its new runs. The legacy archive has 985 final-only records and no
genuine saved P0, so historic Headway and Q_HEADWAY cannot be reconstructed.

A numerical stability smoke check of the current environment is useful but
**cannot** alone establish numerical equivalence between original calibration
code/dependencies and the current clinical scoring protocol.

## 3. Scientific acceptance before an installed-baseline release

1. **Normative choice:** decide if the finite-GT ideal residue is defined
   with respect to uniform intrinsic surface measure on the unique
   *normalized-objective* Pareto front. Specify intrinsic dimension,
   affine metric and conditions under which this measure is well-defined.
   The current evidence supports proposing this measure for DTLZ1/DTLZ2
   M=3, not silently generalizing to every problem.
2. **Unsupported cases:** the 12 other MOPs remain explicitly **unsupported
   for this particular uniform-area candidate**, not generally unsupported
   for all FAIR measures. Preserve individually justified older methods
   and report unsupported corrections with clear status, never as a certified
   `ideal_res=0` inferred from the GT itself.
3. **Independent numeric checks:** separately validate analytical sampling,
   nonoverlap, scaling, K-wise corrected ECDF/anchor identities and the
   unchanged non-Closeness measurements on real optimization fronts.
   Explicitly report Q Closeness drift and the number of clipped observations.
4. **Review changed Q semantics:** the pilot observed a max absolute
   DTLZ1 Q shift around 0.02062. That is a *finding to interpret*, neither
   an acceptance bound nor an automatic rejection. State an acceptance
   criterion independently of the confirmatory sample before claiming
   tolerance/robustness. If the change is a deliberate normative revision,
   its nonzero effect belongs in the compatibility notes.
5. **Environment check:** run both `mb.system.verify_version()` and
   `mb.system.verify_stability()` in the documented execution environment.
   A dependency-version mismatch is a **provenance note**, not an
   automatic release blocker or reason to recalibrate. If numerical stability
   tests *fail*, investigate and block an unsupported numerical certification.
   Do not hide warnings or rewrite the archived baseline version.
6. **Only if numeric calibration changes:** retain the source baseline
   untouched and identify the revised records/method through immutable hashes,
   reproducible generator seed, sampler identity, GT and K coverage.
   Reuse unaffected metric baselines. An actual change to the canonical
   baseline alias requires scientific review; normal library version
   increments do **not** trigger baseline regeneration.

## 4. `audit_report` replacement policy

The existing `calibration/audit_report.html` and the historical report
script are not safe canonical sources for longitudinal Headway: archived
final-only CSVs do not contain P0; a synthetic random initial population
would be fabricated scientific evidence.

A replacement report must:

- keep archived **physical, uncalibrated** FAIR evidence visibly separate
  from real newly generated longitudinal experiments;
- show `HEADWAY` and unavailable Q values as **Unavailable/NA**, not 0,
  when required input or compatible reference is missing;
- show each new real run's P0/Pf hashes and genuine longitudinal Headway,
  full GT/ECDF, calibration/protocol and library/dependency provenance;
- report per-K and per-MOP sample-law support or abstention and explicit
  version-compatibility status; and
- have an immutable versioned filename and checksums. Only an explicit
  release decision may make it the canonical `audit_report`.

Do **not** rerun the obsolete `audit_calibration.py` with fake initial
populations as a shortcut. The provisional `fair-75-candidate-audit-v1.html`
is a noncertified evidence preview, **not** a replacement for the canonical
report.

## 5. Machine-readable release gate

`calibration/scripts/assess_fair_release_issue75.py` produces a read-only
`BLOCKED_PENDING_SCIENTIFIC_RELEASE_REVIEW` report binding hashes,
paired-Q evidence, version and numerical-stability results, and the
historical-data limitations. The `release_blockers` list is a scientific
review checklist, while `compatibility_notes` records descriptive producer/
consumer and certified-dependency differences **without** blocking publication
on version inequality alone. A green CI run is required for engineering confidence but
**never** automatically converts the candidate to a certified artifact.

The current PR remains Draft. No merge or publication is authorized by this
document.
