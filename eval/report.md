# RAG Evaluation Report

- **Generated:** 2026-11-20 22:31 UTC
- **Pipeline:** Hybrid retriever (Chroma dense `0.60` + BM25 sparse `0.40`) + Presidio PII redaction
- **Dataset:** `dataset.json` (12 queries)

## Retrieval Quality

| Metric | @1 | @4 |
| --- | ---: | ---: |
| Recall | 0.875 | 1.000 |
| Precision | 1.000 | 0.312 |
| MRR | 1.000 | 1.000 |

## Retrieval Latency

| Avg (ms) | p50 (ms) | p95 (ms) | Max (ms) |
| ---: | ---: | ---: | ---: |
| 426.2 | 199.9 | 1339.6 | 2247.3 |

## Per-Query Breakdown

| # | Query | Expected | Top-4 Retrieved | R@4 | P@4 | Latency (ms) |
| ---: | --- | --- | --- | ---: | ---: | ---: |
| 1 | Where is Galeria Łódzka parking located and what is its slot prefix? | `location_a_galeria_lodzka.pdf` | `location_a_galeria_lodzka.pdf, location_c_port_lodz.pdf, shared_parking_rules.pdf, location_d_nowa_sukcesja.pdf` | 1.00 | 0.25 | 2247.3 |
| 2 | What is the address of Manufaktura parking? | `location_b_manufaktura.pdf` | `location_b_manufaktura.pdf, entry_exit_and_safety.pdf, accessible_parking.pdf, shared_parking_rules.pdf` | 1.00 | 0.25 | 596.8 |
| 3 | Where is Port Łódź parking and which slots belong to it? | `location_c_port_lodz.pdf` | `location_c_port_lodz.pdf, shared_parking_rules.pdf, location_a_galeria_lodzka.pdf, location_d_nowa_sukcesja.pdf` | 1.00 | 0.25 | 404.3 |
| 4 | What is the address and slot prefix for Nowa Sukcesja parking? | `location_d_nowa_sukcesja.pdf` | `location_d_nowa_sukcesja.pdf, shared_parking_rules.pdf, location_c_port_lodz.pdf, location_a_galeria_lodzka.pdf` | 1.00 | 0.25 | 356.8 |
| 5 | What information do I need to provide to make a parking reservation? | `booking_and_privacy_policy.pdf, booking_policy.pdf` | `booking_policy.pdf, booking_and_privacy_policy.pdf, terms_of_service.pdf, pricing_and_payments.pdf` | 1.00 | 0.50 | 338.6 |
| 6 | How do I cancel or change an existing parking reservation? | `cancellation_and_refunds.pdf` | `cancellation_and_refunds.pdf, pricing_and_payments.pdf, booking_policy.pdf, entry_exit_and_safety.pdf` | 1.00 | 0.25 | 194.4 |
| 7 | Are there accessible or disabled parking bays available? | `accessible_parking.pdf` | `accessible_parking.pdf, entry_exit_and_safety.pdf, shared_parking_rules.pdf, electric_vehicle_charging.pdf` | 1.00 | 0.25 | 200.0 |
| 8 | Can I park and charge an electric vehicle at these parking lots? | `electric_vehicle_charging.pdf` | `electric_vehicle_charging.pdf, shared_parking_rules.pdf, accessible_parking.pdf, pricing_and_payments.pdf` | 1.00 | 0.25 | 199.8 |
| 9 | What are the maximum vehicle length, width, and weight limits? | `vehicle_size_rules.pdf` | `vehicle_size_rules.pdf, terms_of_service.pdf, accessible_parking.pdf, shared_parking_rules.pdf` | 1.00 | 0.25 | 146.6 |
| 10 | What should I do if my assigned bay is blocked or there is a safety incident? | `entry_exit_and_safety.pdf, shared_parking_rules.pdf` | `entry_exit_and_safety.pdf, shared_parking_rules.pdf, accessible_parking.pdf, booking_policy.pdf` | 1.00 | 0.50 | 150.1 |
| 11 | How are parking prices and payments determined? | `pricing_and_payments.pdf` | `pricing_and_payments.pdf, electric_vehicle_charging.pdf, accessible_parking.pdf, shared_parking_rules.pdf` | 1.00 | 0.25 | 152.7 |
| 12 | What are the terms of service and privacy rules for personal data? | `booking_and_privacy_policy.pdf, terms_of_service.pdf` | `terms_of_service.pdf, booking_and_privacy_policy.pdf, booking_policy.pdf, cancellation_and_refunds.pdf` | 1.00 | 0.50 | 126.5 |
