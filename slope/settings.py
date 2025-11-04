import os
import re
import time
import json
from ast import literal_eval
from codetiming import Timer
from tqdm import tqdm, trange
from termcolor import colored, cprint
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

HUIYAN_URL = "https://api.huiyan-ai.com/v1"
DEFAULT_KEY = ""
DIRECT_KEY = ""


GOOGLE_KEY = ""

import subprocess
result = subprocess.run(['whoami'], capture_output=True, text=True)
WIN = True if os.name == 'nt' else False

PUBLIC_MODEL = ""
DATASET_PATH = ""

PRIVATE_MODEL = ""

MODEL_PATH = {"llama3": f"{PUBLIC_MODEL}/Meta-Llama-3-8B-Instruct",
              "llama3-70B": f"{PUBLIC_MODEL}/Meta-Llama-3-70B-Instruct",
              "llama3.1": f"{PUBLIC_MODEL}/Meta-Llama-3.1-8B-Instruct",
              "llama3.1-base": f"{PUBLIC_MODEL}/Meta-Llama-3.1-8B",
              "llama3.1-70B": f"{PUBLIC_MODEL}/Meta-Llama-3.1-70B-Instruct",
              "llama3.1_unc": f"{PRIVATE_MODEL}/Llama-3.1-8B-Unc",
              "llama3.2": f"{PUBLIC_MODEL}/Llama-3.2-3B-Instruct",
              "llama3.2-1B": f"{PUBLIC_MODEL}/Llama-3.2-1B-Instruct",
              "llama3.3-70B": f"{PUBLIC_MODEL}/Llama-3.3-70B-Instruct",
              "llama3.3": f"{PUBLIC_MODEL}/Llama-3.3-70B-Instruct",
              "qwen2.5": f"{PUBLIC_MODEL}/Qwen2.5-7B-Instruct",
              "qwen3": f"{PUBLIC_MODEL}/Qwen3-8B",
              "ds-r1": f"{PUBLIC_MODEL}/DeepSeek-R1-Distill-Qwen-7B",
              "jail_judge": f"{PUBLIC_MODEL}/JailJudge-guard",
              "harmbench": f"{PUBLIC_MODEL}/HarmBench-Llama-2-13b-cls",
              }

TRANS_KEY = ""
TRANS_LOC = ""

# PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
PROJECT_DIR = str(Path(__file__).parent.parent.resolve())
DATA_DIR = f"{PROJECT_DIR}/data"
OUT_DIR = f"{PROJECT_DIR}/output"


def data_path(*file_name):
    return os.path.join(DATA_DIR, *file_name)


def out_path(*file_name):
    return os.path.join(OUT_DIR, *file_name)


def convs_path(file_name: str):
    return os.path.join(DATA_DIR, "conversation", file_name)


def question_path(file_name: str):
    return os.path.join(DATA_DIR, "questions", file_name)


def conf_path(file_name: str):
    return os.path.join(DATA_DIR, "config", file_name)


def pics_path(file_name: str):
    return os.path.join(PROJECT_DIR, "pics", file_name)


cate_abbr = {
    'Malware/Hacking': 'Malware',
    'Privacy': 'Priv',
    'Fraud/Deception': 'Fraud',
    'Economic harm': 'EconHarm',
    'Government decision-making': 'GovDec',
    'Expert advice': 'ExpAdv',
    'Disinformation': 'Disinfo',
    'Harassment/Discrimination': 'Harrass',
    'Physical harm': 'PhysHarm',
    'Sexual/Adult content': 'Sexual',
}
