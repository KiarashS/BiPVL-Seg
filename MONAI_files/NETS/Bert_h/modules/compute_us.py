from typing import Dict, List, Union

import numpy as np
import torch
from ..BERT import BertForMaskedLM
from transformers import BertTokenizer
from .repr_tools import get_module_input_output_at_words

from math import ceil

def compute_us(
    model: BertForMaskedLM,
    tok: BertTokenizer,
    requests: List,
    hparams,
    layer: int,
    device='cuda'
):
    requests_num = sum(len(request_batch) for request_batch in requests)

    contexts = [request["text"] for request_batch in requests for request in request_batch]
    words = [request["subject"] for request_batch in requests for request in request_batch]

    layer_us_batches = []
    batch_size = hparams.batch_size  # Adjust batch size based on your device's capability

    for index in range(ceil(requests_num/batch_size)):
        contexts_batch = contexts[batch_size*index:batch_size*(index+1)]
        words_batch = words[batch_size*index:batch_size*(index+1)]

        # contexts_batch = contexts_batch.tolist()
        # words_batch = words_batch.tolist()

        layer_us_batches.append(get_module_input_output_at_words(model, tok, contexts_batch, words_batch,
                                                    layer, hparams.rewrite_module_tmp, hparams.fact_token)[0].detach().to(device))
        # print(f"layer_us_baches: {layer_us_batches}")
    layer_us = torch.cat(layer_us_batches, dim=0)
    print(f"layer_us: {layer_us.shape}")
    batch_lens = [len(request_batch) for request_batch in requests]
    batch_csum = np.cumsum([0] + batch_lens).tolist()

    u_list = []
    for i in range(len(batch_lens)):
        start = batch_csum[i]
        end = batch_csum[i + 1]
        u_list.append(layer_us[start:end].mean(0))

    # return torch.stack(u_list, dim=0)
    return layer_us

