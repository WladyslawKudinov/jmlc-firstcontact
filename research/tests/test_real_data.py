import numpy as np
from ml.real_data import to_rows, infer_scenario_id, verdict_scores

SITE_OPEN = "Здравствуйте! Это Софья из школы «Общее Дело» по вашей заявке с сайта."
TG_OPEN = "Здравствуйте! Вы оставили номер в нашем боте «Общее Дело»."


def _rec(verdict="горячий", sale="Sold", open_text=SITE_OPEN):
    return {
        "transcript": [
            {"role": "assistant", "text": open_text, "t_sec": None},
            {"role": "user", "text": "да", "t_sec": 1.0},
        ],
        "verdict": verdict,
        "red_flag": None,
        "open_question": None,
        "sale_result": sale,
    }


def test_to_rows_sold_is_label_1():
    assert to_rows([_rec(sale="Sold")])[0]["label"] == 1


def test_to_rows_failed_is_label_0():
    assert to_rows([_rec(sale="Failed")])[0]["label"] == 0


def test_to_rows_builds_transcript_text_in_training_format():
    row = to_rows([_rec(open_text="Привет")])[0]
    assert row["transcript_text"] == "Софья: Привет\nЛид: да"


def test_to_rows_preserves_verdict_and_sale_result():
    row = to_rows([_rec(verdict="холодный", sale="Failed")])[0]
    assert row["verdict"] == "холодный"
    assert row["sale_result"] == "Failed"


def test_infer_scenario_id_site_from_opener():
    assert infer_scenario_id(_rec(open_text=SITE_OPEN)["transcript"]) == "obscheedelo-site"


def test_infer_scenario_id_tg_from_opener():
    assert infer_scenario_id(_rec(open_text=TG_OPEN)["transcript"]) == "obscheedelo-tg"


def test_verdict_scores_are_ordinal_hot_gt_cold_gt_unfit():
    s = verdict_scores([{"verdict": "горячий"}, {"verdict": "холодный"}, {"verdict": "не подходящий"}])
    assert list(s) == [2.0, 1.0, 0.0]


def test_to_rows_merges_external_fields_by_index():
    raw = [_rec(), _rec()]
    fields = [{"grade": "10 класс", "subject": "история"}, {"grade": "9 класс"}]
    rows = to_rows(raw, fields)
    assert rows[0]["fields"]["grade"] == "10 класс"
    assert rows[0]["fields"]["subject"] == "история"
    assert rows[1]["fields"]["grade"] == "9 класс"
