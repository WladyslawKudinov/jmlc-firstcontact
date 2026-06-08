from ml.features import structured_features, make_vectorizer, featurize

REC = {
    "transcript": [{"role": "assistant", "text": "Здравствуйте"},
                   {"role": "user", "text": "Да, мне срочно для работы"}],
    "transcript_text": "Софья: Здравствуйте\nЛид: Да, мне срочно для работы",
    "fields": {"situation": "работа", "level": None, "start_date": "на этой неделе"},
}

def test_structured_features_length_and_completeness():
    f = structured_features(REC)
    assert len(f) == 5
    assert abs(f[0] - 2 / 3) < 1e-9        # 2 of 3 fields filled

def test_featurize_fit_then_transform_same_width():
    vec = make_vectorizer()
    X1 = featurize([REC, REC], vec, fit=True)
    X2 = featurize([REC], vec, fit=False)
    assert X1.shape[1] == X2.shape[1]
    assert X2.shape[0] == 1
