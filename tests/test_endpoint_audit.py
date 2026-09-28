import pandas as pd

from src.endpoint_audit import audit_curves


def test_endpoint_rule_identifies_only_two_severe_study9_curves():
    table = pd.read_csv("data/processed/phbv_weight_loss_decimal.csv")
    audit = audit_curves(table)
    severe = audit[audit["endpoint_integrity_class"].eq("severe_orientation_conflict")]
    assert set(severe["curve_id"]) == {
        "study_9__instance_4",
        "study_9__instance_8",
    }
    assert len(audit) == 129
