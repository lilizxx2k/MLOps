## Profiling (baseline)

Command:
python -m cProfile -s cumulative -o reports/profiling/model.prof src/mlops/model.py

Visualization:
snakeviz reports/profiling/model.prof

Notes:
This run is dominated by PyTorch import/startup overhead (cold start).
See model_snakeviz.png
