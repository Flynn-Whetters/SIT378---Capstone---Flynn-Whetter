# AFL Heatmap & Model Accuracy Pipeline

This project provides tools to:
1. **Generate AFL player heatmaps** from tracking/event data.
2. **Evaluate model accuracy metrics** from prediction results.

It combines sample datasets, reproducible notebooks, and a script-based pipeline for flexible use.

---

## Project Structure

- **AccuacyMetrics/**
  - `AccuracyPipeline.ipynb` – Notebook for evaluating model confidence, detection counts, and performance stability over time.
  - `accuracyOutputs/` - Accuracy output graphs

- **Data/**
  - `kick.csv`, `mark.csv`, `Tackle.csv` – Example event datasets (player actions).
  - `tracking_csv.csv`, `Video_tracking.csv` – Sample tracking datasets for heatmap generation.

- **Heatmaps/**
  - `afl_heatmap.py` – Command-line script to generate heatmaps from datasets.
  - `Finaloutputs.ipynb` – Notebook showing worked examples and visualizations.
  - `outputs/` – Example generated results:
    - `overall/overall.png` – Heatmap across all players.
    - `per_id/*.png` – Individual player heatmaps (by ID).

---

### Generate Heatmaps
Run the heatmap script with a dataset:

```bash
python afl_heatmap.py Data/Video_tracking.csv:tracking --out-dir outputs --sigma 2.0
