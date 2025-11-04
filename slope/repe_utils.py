#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/7/14 21:32
# @Author  : Huasir
# @File    : repe_utils.py
# @Description: 表征相关的代码操作
import os.path
import spacy
from datasets import load_dataset
from nltk import sent_tokenize
from sentence_transformers import SentenceTransformer, util
import numpy as np
from sklearn.decomposition import PCA
from .llms import *


class Encoder:
    def __init__(self, model="base", device="cpu"):
        if model == "mini":
            cprint(f"[*]Load embedding model: all-MiniLM-L6-v2 @ {device}\n", "blue")  # 384 dimension
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
            self.dim = 384
        elif model == "base":
            cprint(f"[*]Load embedding model: all-mpnet-base-v2 @ {device}\n", "blue")  # 768 dimension
            self.encoder = SentenceTransformer('all-mpnet-base-v2', device=device)
            self.dim = 768
        else:
            # text-embedding-ada-002  text-embedding-3-large
            self.encoder = llm_adapter(model, device=device)
            self.dim = 999

    def encode(self, text):
        embeddings = self.encoder.encode(text)
        # if isinstance(embeddings, np.ndarray):
        #     embeddings = embeddings.tolist()
        return embeddings


def get_hidden_states(model, tokenizer, prompts: list, layer=-1, token_pos=-1, verbose=False):
    """
    默认获取最后一层隐藏层，最后一个token处的状态
    prompts: 提示词 List[Str]列表，或者是 List[Dict]列表
    """
    if not prompts:
        cprint("Empty prompts list!", "red")
        return []

    if isinstance(prompts[0], str):  # LLM.sys_prompt("You are a helpful assistant.") +
        prompt_list = [LLM.user_prompt(prompt) for prompt in prompts]
    else:
        prompt_list = prompts

    results = None
    batch = 1
    range_bar = trange(0, len(prompt_list), batch, desc="Getting hidden states") if verbose else range(len(prompt_list))

    for i in range_bar:
        inputs = tokenizer.apply_chat_template(prompt_list[i:i+batch],
                                               add_generation_prompt=True,
                                               return_tensors="pt",
                                               padding=True,
                                               return_dict=True).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hidden_state_list = outputs.hidden_states[layer][:, token_pos, :].cpu().numpy()
        if results is None:
            results = hidden_state_list
        else:
            results = np.concatenate([results, hidden_state_list], axis=0)

    return results


def calc_refusal_vector(model: LocalLLM, n_components: int | float, verbose=False):
    """聚类计算拒绝表征向量"""
    print(f"[*]Calculating the Refusal Vector")
    if not isinstance(model, LocalLLM):
        print(f"[*]Local LLM model should be loaded...")
        return None

    # 加载善恶对立问题

    benign_inst = json.load(open(data_path("probe_inst/benign_inst.json")))
    harmful_inst = json.load(open(data_path("probe_inst/harmful_inst.json")))

    benign_repr = get_hidden_states(model.model, model.tokenizer, benign_inst)  # 善意问题
    evil_repr = get_hidden_states(model.model, model.tokenizer, harmful_inst)  # 恶意问题

    # 计算接受、拒绝意识中心点
    pca_model = PCA(n_components=n_components).fit(np.concatenate([benign_repr, evil_repr], axis=0))
    if verbose:
        print("[*]pca_model.explained_variance_ratio_.cumsum()\n", pca_model.explained_variance_ratio_.cumsum())

    helpful_center = pca_model.transform(benign_repr).mean(axis=0)
    refusal_center = pca_model.transform(evil_repr).mean(axis=0)

    # 计算拒绝向量
    refusal_vector = refusal_center - helpful_center
    refusal_vector /= np.linalg.norm(refusal_vector)

    return pca_model, refusal_vector


def calc_refusal_vector2(model: LocalLLM, verbose=False):
    """直接计算表征向量差值"""
    print(f"[*]Calculating the Refusal Vector")
    if not isinstance(model, LocalLLM):
        print(f"[*]Local LLM model should be loaded...")
        return None

    # 加载善恶对立问题
    benign_inst = open(data_path("probe_inst/bh_ques/benign_test.txt")).readlines()
    harmful_inst = open(data_path("probe_inst/bh_ques/harmful_test.txt")).readlines()

    benign_repr = get_hidden_states(model.model, model.tokenizer, benign_inst)  # 善意问题表征
    evil_repr = get_hidden_states(model.model, model.tokenizer, harmful_inst)  # 恶意问题表征

    diff_repr = evil_repr - benign_repr
    refusal_vector = diff_repr.mean(axis=0)

    return refusal_vector


def phrase_chunks(prompt):
    """以名词短语（noun phrases）为单位拆分文本。"""
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(prompt)

    last_end = 0
    chunks = []

    for chunk in doc.noun_chunks:
        # 添加短语前的文本
        if chunk.start_char > last_end:
            chunks.append(prompt[last_end:chunk.start_char])

        # 添加短语
        chunks.append(prompt[chunk.start_char:chunk.end_char])
        last_end = chunk.end_char

    # 添加剩余部分
    if last_end < len(prompt):
        chunks.append(prompt[last_end:])

    # 去除所有chunk的左右空格
    processed_chunks = [chunk.strip().lstrip() for chunk in chunks]

    return processed_chunks


def phrase_sentences(prompt):
    """以句子为单位拆分文本"""
    chunks = []
    sentences = sent_tokenize(prompt)
    # 处理句子，保留句子间的空格和换行
    last_end = 0

    for sent in sentences:
        # 找到句子在原文中的位置
        start = prompt.find(sent, last_end)

        # 添加句子前的空白字符
        if start > last_end:
            chunks.append(prompt[last_end:start])

        # 添加句子本身
        chunks.append(sent)
        last_end = start + len(sent)

    # 添加最后剩余的部分
    if last_end < len(prompt):
        chunks.append(prompt[last_end:])
    chunks = [c for c in chunks if c]
    return chunks
