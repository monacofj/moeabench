def test_audit():
    # Scenario: Test the new Cascade Audit logic
    metrics = {
        "q_denoise": 0.35,
        "q_closeness": 0.70,
        "q_coverage": 0.45,
        "q_gap": 0.80,
        "q_regularity": 0.90,
        "q_balance": 0.90,
        "baseline_ok": True,
        "input_ok": True
    }
    
    print("Testing Cascade Diagnostic Biopsy...")
    # PerformanceAuditor.audit_synthesis needs AuditResult objects, 
    # but for this test we can mock a simple Metrics dict if we refactor PerformanceAuditor.audit back to accept it
    # OR we use the high-level audit() with a mock MOP.
    
    # Since PerformanceAuditor.audit was removed, let's use the actual synthesis logic
    from MoeaBench.diagnostics.auditor import PerformanceAuditor
    from MoeaBench.diagnostics.qscore import QResult
    from MoeaBench.diagnostics.fair import FairResult
    
    q_scores = {k: QResult(value=v, name=k.upper()) for k, v in metrics.items() if k.startswith("q_")}
    f_metrics = {k: FairResult(value=v, name=k.upper(), description="Mock") for k, v in metrics.items() if k.startswith("q_")}
    
    q_res = PerformanceAuditor.audit_quality(q_scores, mop="DTLZ2", k=50)
    f_res = PerformanceAuditor.audit_fair(f_metrics)
    
    diag = PerformanceAuditor.audit_synthesis(q_res, f_res)
    print(diag.report())

if __name__ == "__main__":
    test_audit()
