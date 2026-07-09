from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    settings = yaml.safe_load(file)
