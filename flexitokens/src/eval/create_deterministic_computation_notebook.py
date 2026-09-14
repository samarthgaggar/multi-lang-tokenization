#!/usr/bin/env python3
"""Create a runnable notebook containing the deterministic FLORES computation."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EVALUATOR = ROOT / "flexitokens" / "src" / "eval" / "deterministic_sentence_vocabulary.py"
NOTEBOOK = ROOT / "deterministic_flores_computation.ipynb"


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}


def main():
    evaluator_source = EVALUATOR.read_text().split('if __name__ == "__main__":', 1)[0].rstrip() + "\n"
    notebook = {
        "cells": [
            markdown(
                "# Deterministic FLORES vocabulary computation\n\n"
                "This notebook contains the code used to compute the deterministic full-sentence FlexiTokens vocabularies. "
                "It is separate from `deterministic_flores_vocabulary.ipynb`, which is the shorter results report."
            ),
            markdown(
                "## Protocol\n\n"
                "Each FLORES sentence is processed as one complete input. The model's Gumbel sampling path is disabled for evaluation, and a boundary is placed exactly when `sigmoid(boundary_logit) > 0.5`. "
                "The output vocabulary is the counted set of unique decoded segments across all sentence tokenizations."
            ),
            code(
                "# Make the repository's FlexiTokens package importable when this notebook runs from the repository root.\n"
                "import sys\n"
                "from pathlib import Path\n"
                "REPOSITORY_ROOT = Path.cwd()\n"
                "PACKAGE_ROOT = REPOSITORY_ROOT / 'flexitokens'\n"
                "if not PACKAGE_ROOT.is_dir():\n"
                "    raise FileNotFoundError('Run this notebook from the repository root.')\n"
                "sys.path.insert(0, str(PACKAGE_ROOT))"
            ),
            markdown("## Evaluator implementation\n\nThe following cell is the complete implementation used for the computation."),
            code(evaluator_source),
            markdown("## Run the full six-language computation\n\nSet `RUN_FULL_ANALYSIS` to `True` to run the complete evaluation. It loads the local checkpoint and cached FLORES data, and can take a while."),
            code(
                "import os\n"
                "import subprocess\n"
                "\n"
                "RUN_FULL_ANALYSIS = False\n"
                "if RUN_FULL_ANALYSIS:\n"
                "    environment = os.environ.copy()\n"
                "    environment['HF_HUB_OFFLINE'] = '1'\n"
                "    environment['PYTHONPATH'] = 'flexitokens'\n"
                "    command = [\n"
                "        sys.executable, 'flexitokens/src/eval/deterministic_sentence_vocabulary.py',\n"
                "        '--model_path', 'model_ckpts/_2025-05-03_14-43-31',\n"
                "        '--output_dir', 'results/deterministic_sentence_segment_vocabulary_full_v2',\n"
                "        '--languages', 'en,es,ru,uk,hi,te', '--batch_size', '4', '--device', 'cpu',\n"
                "    ]\n"
                "    subprocess.run(command, check=True, env=environment)\n"
                "else:\n"
                "    print('Set RUN_FULL_ANALYSIS = True to run the full evaluation.')"
            ),
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK.write_text(json.dumps(notebook, indent=1) + "\n")


if __name__ == "__main__":
    main()
