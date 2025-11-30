# Contributing to Novartis Datathon 2025 Project

Thank you for your interest in contributing to this repository.  
This project implements an end-to-end pipeline for forecasting generic erosion in pharmaceutical sales.

## How to Contribute

1. **Fork the repository** and create a feature branch from `main`:
   - `feature/...` for new features
   - `fix/...` for bug fixes
2. **Open an issue** before starting major changes (new models, new validation schemes, data pipeline refactors) and briefly describe:
   - What you want to change
   - Why it is useful
   - Any potential impact on existing experiments or results
3. **Write tests** (or update existing ones) for new logic in:
   - Data loading / feature engineering
   - Model training / validation / inference
4. **Run checks locally** before submitting a PR:
   - All tests pass
   - No obvious flake8/black (or equivalent) violations
5. **Submit a Pull Request**:
   - Keep it as focused as possible
   - Clearly describe the motivation, changes, and any breaking behavior
   - If you change metrics, configs, or model selection, include a short summary of before/after results.

## Code Style & Guidelines

- Prefer **clear, modular functions** over large notebooks or ad-hoc scripts.
- Keep all configuration in config files where possible (not hard-coded in code).
- Log important training details (metrics, seeds, config IDs) so experiments are reproducible.
- Do not commit raw competition data or credentials.

## Issues & Bug Reports

When opening an issue, please include:

- OS / environment (local vs Colab / cloud)
- Python version and key libraries (CatBoost, LightGBM, XGBoost, etc.)
- Steps to reproduce (commands, configs)
- Relevant logs or error messages (trimmed to essentials)

## Maintainers

- **Arman Feili**  
  GitHub: https://github.com/armanfeili  
  LinkedIn: https://www.linkedin.com/in/arman-feili/

- **Saeed Zohoorian**  
  GitHub: https://github.com/saeedzns  
  LinkedIn: https://www.linkedin.com/in/saeed-zohoorian-631a80158/

We appreciate every contribution, from fixing typos in the documentation to experimenting with new models and validation strategies.
