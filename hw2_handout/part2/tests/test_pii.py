from mini_data_pipeline.pipeline import mask_pii


def test_mask_all_supported_pii():
    text = (
        "Email Ada.Lovelace+lab@example.org or admin@test.co.uk. "
        "Call 212-555-0100, 212 555 0101, (212) 555-0102, "
        "+1 212-555-0103, or +1 (212) 555-0104. "
        "Hosts: 192.0.2.1 and 255.255.255.255."
    )
    masked = mask_pii(text)
    assert masked == (
        "Email |||EMAIL_ADDRESS||| or |||EMAIL_ADDRESS|||. "
        "Call |||PHONE_NUMBER|||, |||PHONE_NUMBER|||, |||PHONE_NUMBER|||, "
        "|||PHONE_NUMBER|||, or |||PHONE_NUMBER|||. "
        "Hosts: |||IP_ADDRESS||| and |||IP_ADDRESS|||."
    )


def test_invalid_or_unsupported_forms_are_not_masked():
    text = (
        "alice(at)example(dot)com 212.555.0199 2125550199 "
        "1.2.3 256.1.1.1 01.2.3.4 999.999.999.999 "
        "٢١٢-٥٥٥-٠١٠٠ ١٩٢.٠.٢.١"
    )
    assert mask_pii(text) == text


def test_simplified_email_grammar_is_deliberately_permissive():
    text = "Write to .lead+tag@-lab.example or a@b.co.uk."
    assert mask_pii(text) == (
        "Write to |||EMAIL_ADDRESS||| or |||EMAIL_ADDRESS|||."
    )


def test_empty_text_is_unchanged():
    assert mask_pii("") == ""