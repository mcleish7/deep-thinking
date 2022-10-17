#code checks if file has been created if not creates it
from pathlib import Path

Path("graph_generation_files").mkdir(parents=True, exist_ok=True)
Path("testing_1").mkdir(parents=True, exist_ok=True)