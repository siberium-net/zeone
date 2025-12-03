# Themis Compliance Layer

## Purpose
Legal-first P2P: nodes must avoid storing/serving prohibited content and protect third-party PII. Themis provides automated triage before persistence or propagation.

## Jurisdiction Awareness
- Geo rules loaded from `legal_rules.json` (DE/RU/US examples) with banned topics.
- GeoIP (optional `geoip2` DB) can infer country; operator can override manually in UI.
- Policies are evaluated per node; no central authority required.

## PII Detection
- **PIIGuard (Presidio):** Detects CREDIT_CARD, CRYPTO, EMAIL, IBAN, PERSON, PHONE_NUMBER, US_SSN.
- Risk scoring (0..1). If critical entities (credit cards/SSN/IBAN) found with high confidence → `BLOCKED_PII`; otherwise `WARNING` or `SAFE`.

## AI Legal Judge
- **ComplianceJudge (OllamaAgent):** System prompt as Compliance Officer; jurisdiction-aware. Flags Hate Speech, Extremism, Illegal porn, Copyright risks. Returns allowed/risk_category/reasoning; conservative defaults.

## Workflow
1. Ingest text → PIIGuard scan → AI judge.
2. compliance_status attached to metadata: SAFE | WARNING | BLOCKED. BLOCKED content is not propagated (or is encrypted with ephemeral key).
3. UI surfaces badges (green/yellow/red shields) per asset.

## Philosophy: Legal-First P2P
- Sovereignty with accountability: jurisdiction-aware filters, opt-out for high-risk topics, and future ZKP proofs for age/identity without leaking PII.
