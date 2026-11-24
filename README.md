# Football_ML

Προσωπικό ερευνητικό project: πιθανότητες για αγώνες Premier League πριν τη σέντρα, με στόχο να βρεθεί
μετρήσιμο edge απέναντι στην αγορά. Δεν είναι σύσταση για στοίχημα.

- **Πλάνο και κατάσταση:** [docs/PLAN.md](docs/PLAN.md)
- **Οδηγός για Claude και κανόνες μεθοδολογίας:** [CLAUDE.md](CLAUDE.md)
- **Παλιά πειράματα (26/10–14/11/2026):** [archive/INDEX.md](archive/INDEX.md), μόνο τοπικά, εκτός git

## Δομή

```
football_ml/   κώδικας: ingest → features → models → evaluate → betting
data/raw/      αμετάβλητα δεδομένα πηγών
data/processed παράγωγα, αναπαράγονται από τον κώδικα
outputs/       αποτελέσματα τρεχόντων runs
tests/         pytest
docs/          πλάνο και έρευνα πηγών
archive/       παγωμένα παλιά πειράματα
```

## Εντολές

```powershell
python -m football_ml prepare
python -m football_ml baseline
python -m pytest -q
```

API keys μπαίνουν μόνο στο τοπικό `.env` (βλ. `.env.example`). Ποτέ σε αρχεία, chat ή git.
