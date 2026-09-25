# Contributing

Please open an issue describing the observable behavior and evidence a change will add. Keep model-specific code behind a runtime adapter and scoring rules inside separate probes. Preserve old experiment schemas or provide a migration. Never add automatic model downloads, cloud credentials, unsupported claims about model intentions, or opaque safety scores.

Run the test suite and linter before submitting changes. For a probe change, include a controlled positive and negative evaluator case plus a complete mock workflow test. For a runtime change, document which local model and hardware were actually exercised.
