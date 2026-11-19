# Quantitative Information Flow in Subliminal Learning

This project investigates information leakage in subliminal learning, where a teacher model transmits preferences to a student through seemingly unrelated data, such as numerical sequences.

We use Quantitative Information Flow (QIF) and Bayes vulnerability to measure how much generated sequences reveal about the teacher's preference. The project compares exact channel computation with Monte Carlo estimation and explores the relationship between leakage and preference transfer.

## Repository

- Python modules for channel modeling, leakage estimation, and student experiments.
- [`experiments/`](experiments/) contains experiment scripts and Slurm launchers.
- [`report/relatorio.tex`](report/relatorio.tex) contains the collaborative report.
- [`runs/`](runs/) contains generated data and results.
