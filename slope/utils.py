import sys
import uuid
import math
import yaml
import copy
import random
import requests
import datetime
import argparse
import subprocess
import ujson as json
from langid import langid
from pprint import pprint
from collections import Counter
from addict import Dict as aDict

from .settings import *


def save_with_indent(obj: Union[dict, list], filename: str, indent=True, append=False, slient=False):
    """Dump json object in indent or not"""
    if os.path.dirname(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    indent_file = filename.replace(".json", ".indent.json")
    fp_mode = "a" if append else "w"
    print(f"[*]File operation mode is \"{fp_mode}\"")

    if filename.endswith(".json"):
        indent_file = filename  # json file dumps in indent
        print(f"[*]Dumping json in indent mode (Default).")
    elif filename.endswith(".jsonl") and indent != False:
        assert type(obj) is list
        with open(filename, fp_mode, encoding="utf-8") as f:
            [f.write(f"{json.dumps(res, ensure_ascii=False)}\n") for res in obj]
    elif filename.endswith(".txt"):
        assert type(obj) is list
        with open(filename, fp_mode, encoding="utf-8") as f:
            [f.write(f"{res}\n") for res in obj]

    if indent:
        if indent_file.endswith(".json"):
            with open(indent_file, fp_mode, encoding="utf-8") as f:
                f.write(f"{json.dumps(obj, indent=4, ensure_ascii=False)}\n")
        elif indent_file.endswith(".jsonl"):
            with open(indent_file, fp_mode, encoding="utf-8") as f:
                [f.write(f"{json.dumps(res, indent=4, ensure_ascii=False)}\n") for res in obj]

        if indent == "only" and indent_file != filename:
            os.remove(filename)
            cprint(f"[+] Saved in {indent_file}", "green") if not slient else None
        else:
            cprint(f"[+] Saved in {filename} and {indent_file}", "green") if not slient else None
    else:
        cprint(f"[+] Saved in {filename}", "green") if not slient else None


def jsonl_2_list(file_name, key: str | list | None = None) -> list:
    if not os.path.exists(file_name):
        return []
    if file_name.endswith(".jsonl"):
        if key:
            if isinstance(key, str):
                return [json.loads(line).get(key) for line in open(file_name, encoding="utf-8").readlines()]
            elif isinstance(key, list):
                results = []
                for line in open(file_name, encoding="utf-8").readlines():
                    node = json.loads(line)
                    node_dict = {}
                    for k in key:
                        node_dict[k] = node.get(k, None)
                    results.append(node_dict)
                return results
        else:
            return [json.loads(line) for line in open(file_name, encoding="utf-8").readlines()]
    elif file_name.endswith(".txt"):
        return [line.strip() for line in open(file_name, encoding="utf-8").readlines()]
    else:
        print(f"[!]File extension not supported.")
        return []


def current_time():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))


def time_cost(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        dur_time = end_time - start_time
        if dur_time < 60:
            strf_time = f"{dur_time:.2f} s"
        elif dur_time < 3600:
            strf_time = f"{int(dur_time) // 60} m {int(dur_time) % 60} s"
        else:
            strf_time = f"{int(dur_time) // 3600} h {int(dur_time) % 3600 // 60} m {int(dur_time) % 3600 % 60} s"
        print(f"[*]{func.__name__} cost {strf_time}")
        return result

    return wrapper


def show_args(args, pause=True):
    cprint(f"[*]Show the arguments ({datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')})", "blue", attrs=['bold'])
    for arg in vars(args):
        print(f"{arg:<15}: {getattr(args, arg)}")
    cprint("-" * 99, "blue", attrs=['bold'])
    if pause:
        try:
            input(">> Press enter to continue...")
        except KeyboardInterrupt:
            exit()


def print_freedom():
    print(r'''
___________                         .___              
\_   _____/______   ____   ____   __| _/____   _____  
 |    __) \_  __ \_/ __ \_/ __ \ / __ |/  _ \ /     \ 
 |     \   |  | \/\  ___/\  ___// /_/ (  <_> )  Y Y  \
 \___  /   |__|    \___  >\___  >____ |\____/|__|_|  /
     \/                \/     \/     \/            \/  freedom!
    ''')


def print_over():
    print(r"""                              
   ___   __   __   ___   _ __ 
  / _ \  \ \ / /  / _ \ | '__|
 | (_) |  \ V /  |  __/ | |   
  \___/    \_/    \___| |_|   
                               Over!
    """)



if __name__ == "__main__":
    pass
