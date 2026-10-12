import json
import os

os.makedirs("colab", exist_ok=True)

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# GUARD-TRAJECTORY: Training-Free Real-Time Monitoring and Safety Interception in Discrete Diffusion Language Models\n",
                "**Conference Paper Experimental Notebook (Google Colab / CPU and GPU Compatible)**\n\n",
                "This notebook implements and executes Guard-Trajectory on the pretrained `kuleshov-group/mdlm-owt` discrete diffusion model with candidate safety scorers on RealToxicityPrompts."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Install Dependencies\n",
                "!pip install -q torch transformers datasets accelerate scikit-learn matplotlib pandas\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. Check Device\n",
                "import torch\n",
                "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
                "print(f'Running on device: {device}')\n",
                "if device == 'cuda':\n",
                "    print(f'GPU: {torch.cuda.get_device_name(0)}')\n",
                "else:\n",
                "    print('Running on CPU mode (Supported and verified without GPU)')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 3. Run Standalone Guard-Trajectory Script\n",
                "# You can adjust --steps (16, 32) and --device ('cpu' or 'cuda')\n",
                "!python run_colab_standalone.py --device {device} --steps 16\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 4. Display Results Table\n",
                "import pandas as pd\n",
                "try:\n",
                "    df = pd.read_csv('colab_results_summary.csv')\n",
                "    display(df)\n",
                "except Exception as e:\n",
                "    print('Results file not found:', e)\n"
            ]
        }
    ],
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "provenance": []
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

with open("colab/guard_trajectory_colab.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Generated colab/guard_trajectory_colab.ipynb successfully.")
