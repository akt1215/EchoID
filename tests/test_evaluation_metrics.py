from evaluation import metrics

GT = {("m1", "S0"): "Jordan Lee", ("m1", "S1"): "Avery Chen",
      ("m2", "S0"): "Jordan Lee"}
SEC = {("m1", "S0"): 100.0, ("m1", "S1"): 100.0, ("m2", "S0"): 200.0}


def test_two_people_under_one_identity_is_a_conflation():
    a = {("m1", "S0"): "Jordan Lee", ("m1", "S1"): "Jordan Lee"}
    found = metrics.conflations(a, GT)
    assert len(found) == 1
    assert found[0][0] == "Jordan Lee"
    assert found[0][1] == {"jordan lee", "avery chen"}


def test_one_person_across_meetings_is_not_a_conflation():
    a = {("m1", "S0"): "Jordan Lee", ("m2", "S0"): "Jordan Lee"}
    assert metrics.conflations(a, GT) == []


def test_unassigned_clusters_never_conflate():
    a = {("m1", "S0"): None, ("m1", "S1"): None}
    assert metrics.conflations(a, GT) == []


def test_wrong_name_is_a_false_accept():
    a = {("m1", "S0"): "Avery Chen"}
    assert metrics.false_accepts(a, GT) == [(("m1", "S0"), "Avery Chen", "Jordan Lee")]


def test_unknown_is_not_a_false_accept():
    assert metrics.false_accepts({("m1", "S0"): None}, GT) == []


def test_coverage_is_share_of_correctly_named_speech_time():
    a = {("m1", "S0"): "Jordan Lee", ("m1", "S1"): None, ("m2", "S0"): "Jordan Lee"}
    assert metrics.coverage(a, GT, SEC) == 300.0 / 400.0


def test_coverage_excludes_wrongly_named_time():
    a = {("m1", "S0"): "Avery Chen", ("m1", "S1"): None, ("m2", "S0"): None}
    assert metrics.coverage(a, GT, SEC) == 0.0


def test_summarize_reports_all_three_metrics():
    a = {("m1", "S0"): "Jordan Lee", ("m1", "S1"): "Jordan Lee", ("m2", "S0"): None}
    s = metrics.summarize(a, GT, SEC)
    assert s["conflations"] == 1
    assert s["false_accepts"] == 1      # m1/S1 is Avery Chen, named Jordan Lee
    assert s["coverage"] == 100.0 / 400.0
    assert s["assigned"] == 2 and s["total"] == 3


def test_spelling_variants_are_the_same_person():
    gt = {("m1", "S0"): "Riley Stone", ("m2", "S0"): "Riley Stone"}
    a = {("m1", "S0"): "Riley Stone", ("m2", "S0"): "Riley Stone"}
    assert metrics.conflations(a, gt) == []
    assert metrics.false_accepts(a, gt) == []
