from src.functions import redact_pii, redact_pii_rows


def test_redact_pii_masks_personal_and_contact_info():
    text = (
        "Reservation by John Smith, email john.smith@example.com, "
        "phone +48 501 234 567, IP 192.168.1.42."
    )
    redacted = redact_pii(text)

    assert "John Smith" not in redacted
    assert "john.smith@example.com" not in redacted
    assert "501 234 567" not in redacted
    assert "192.168.1.42" not in redacted
    assert "<PERSON>" in redacted
    assert "<EMAIL_ADDRESS>" in redacted
    assert "<PHONE_NUMBER>" in redacted
    assert "<IP_ADDRESS>" in redacted


def test_redact_pii_masks_financial_identifiers():
    text = "Card 4532-0151-1283-0366 and IBAN PL61109010140000071219812874."
    redacted = redact_pii(text)

    assert "4532-0151-1283-0366" not in redacted
    assert "PL61109010140000071219812874" not in redacted
    assert "<CREDIT_CARD>" in redacted
    assert "<IBAN_CODE>" in redacted


def test_redact_pii_preserves_public_parking_data():
    text = (
        "Manufaktura at Łódź, Drewnowska 58 and Galeria Łódzka at "
        "Łódź, al. Piłsudskiego 15/23. Slots A01-A03 and B01-B03 are open "
        "06:00-22:00 on 2026-11-15 for 5.0 PLN/h."
    )
    assert redact_pii(text) == text


def test_redact_pii_handles_empty_strings():
    assert redact_pii("") == ""
    assert redact_pii("   ") == "   "


def test_redact_pii_rows_sanitizes_strings_and_keeps_non_strings():
    rows = [
        {
            "id": 1,
            "slot_code": "A01",
            "parking_name": "Galeria Łódzka",
            "location": "Łódź, al. Piłsudskiego 15/23",
            "spot_details": "Reserved near Maria Kowalska (maria@example.com)",
            "price_per_hour_pln": 5.0,
            "is_available": 1,
            "note": None,
        }
    ]
    sanitized = redact_pii_rows(rows)

    assert sanitized[0]["id"] == 1
    assert sanitized[0]["slot_code"] == "A01"
    assert sanitized[0]["parking_name"] == "Galeria Łódzka"
    assert sanitized[0]["location"] == "Łódź, al. Piłsudskiego 15/23"
    assert sanitized[0]["price_per_hour_pln"] == 5.0
    assert sanitized[0]["is_available"] == 1
    assert sanitized[0]["note"] is None
    assert "Maria Kowalska" not in sanitized[0]["spot_details"]
    assert "maria@example.com" not in sanitized[0]["spot_details"]
    assert "<PERSON>" in sanitized[0]["spot_details"]
    assert "<EMAIL_ADDRESS>" in sanitized[0]["spot_details"]
