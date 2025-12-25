import torch
import argparse
import glob
import math
import random
import json
import numpy as np
import pandas as pd
import tqdm as tqdm
from scipy.special import softmax
import scipy.stats as stats
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display, HTML
# from captum.attr import visualization
from typing import List, Dict, Tuple

from transformers import AutoTokenizer

import datasets
from datasets import load_dataset, load_metric 
from datasets import list_datasets, list_metrics

# from BERT_explainability.modules.BERT.BertForSequenceClassification import BertForSequenceClassification
from ..BERT import BertForMaskedLM
from transformers import BertTokenizer
from . import nethook

from .repr_tools import get_module_input_output_at_words


# BERT_PRONOUNS = {"pos": "he",
#                  "neg": "she"}

special_tokens = {"[CLS]", "[SEP]"}
special_idxs = {101,102}    
mask = "[PAD]"
mask_id = 0

def preprocess_sample(tokenizer,text,device):
    tokenized_input  = tokenizer(text, add_special_tokens=True, truncation=True)
    input_ids = tokenized_input['input_ids']
    text_ids = (torch.tensor([input_ids])).to(device)
    text_words = tokenizer.convert_ids_to_tokens(text_ids[0])
    mask_positions = (text_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)
    mask_ids = mask_positions[1].item()

    # mask special tokens
    att_mask = tokenized_input['attention_mask']
    spe_idxs = [x for x, y in list(enumerate(input_ids)) if y in special_idxs]
    att_mask = [0 if index in spe_idxs else 1 for index, item in enumerate(att_mask)]
    att_mask = [0 if index in spe_idxs else 1 for index, item in enumerate(att_mask)]
    att_mask = (torch.tensor([att_mask])).to(device)
    
    return text_ids, att_mask, text_words, mask_ids

def compute_v(
    model: BertForMaskedLM,
    tokenizer: BertTokenizer,
    hparams,
    layer: int,
    context_templates: List[dict[str, str]],
    batch_id,
    gender_values: List[str],
    value_at_mlp: bool = False,
    past_deltas: torch.Tensor = None,
) -> Tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    

    templates = []
    for i in range(len(context_templates)):
        templates.append(context_templates[i]['text'])

    input_tokens = tokenizer(templates,add_special_tokens=True,truncation=True, padding=True, max_length=100,return_tensors='pt').to(model.device)
    indices = (input_tokens.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)
    # Extracting row indices and column indices where 103 is found
    row_indices = indices[0]
    column_indices = indices[1]

    # If you want the result as a list of indices for each row:
    lookup_idxs = [column_indices[row_indices == i].item() for i in range(input_tokens.input_ids.size(0))]
    # print(lookup_idxs)
    
    
    # input_tok = []
    # lookup_idxs = []
    # attention_masks = []
    # for i in range(len(context_templates)):
    #     text_ids, att_mask, text_words, mask_ids = preprocess_sample(tokenizer,context_templates[i]['text'],model.device)
    #     input_tok.append(text_ids)
    #     lookup_idxs.append(mask_ids)
    #     attention_masks.append(att_mask)

    # print(input_tok[0])

    loss_layer = max(hparams.v_loss_layer, layer)
    # print(f"Trying optimization objective to {loss_layer}...")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    deltas = [torch.zeros((input_tokens.input_ids.size(0),model.config.hidden_size), requires_grad=True,
                          device=model.device) for _ in range(len(gender_values))]

    delta_shared = torch.zeros((input_tokens.input_ids.size(0),model.config.hidden_size), requires_grad=True,
                               device=model.device)


    target_init, kl_distr_init = None, None
    output_index = None

    # Insert deltas for computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init, output_index
        
        if cur_layer == hparams.mlp_module_tmp.format(layer):
            # print(cur_out.shape)
            cur_out = cur_out.to(model.device)
            # print(cur_out.shape)
            # Store initial value of the vector of interest
            if target_init is None:
                # print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[torch.arange(cur_out.size(0)),lookup_idxs].detach().clone()

            # print(cur_out[torch.arange(cur_out.size(0)),lookup_idxs].shape)
            # print((delta+delta_shared).shape)

            cur_out[torch.arange(cur_out.size(0)),lookup_idxs] += (delta+delta_shared)
            # for i, idx in enumerate(lookup_idxs):
            #     cur_out[i, idx, :] += (delta + delta_shared)
            # cur_out[i, idx, :] += delta

        elif cur_layer == hparams.layer_module_tmp.format(layer) and not value_at_mlp:
            # print(cur_out.shape)
            # cur_out = (cur_out[0].to(device), tuple(co.to(device) for co  in cur_out[1]))
            cur_out = cur_out.to(model.device)
            
            # Store initial value of the vector of interest
            if target_init is None:
                # print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[torch.arange(cur_out.size(0)),lookup_idxs].detach().clone()

            cur_out[torch.arange(cur_out.size(0)),lookup_idxs] += (delta+delta_shared)

            # for i, idx in enumerate(lookup_idxs):
            #     cur_out[i, idx, :] += (delta + delta_shared)

        return cur_out

    # Optimizer
    opt = torch.optim.Adam(deltas + [delta_shared], lr=hparams.v_lr)
    nethook.set_requires_grad(True, model)

    # Optimmize
    # print("Optimizing...")
    # print("Loss structure: NLL + KL + WEIGHT DECAY + ORTHOGONALITY")

    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # iterate over positive and negative examples
        for index, (delta, target) in enumerate(zip(deltas, gender_values)):

            past_delta = past_deltas[target] if past_deltas else None
            # Forward propagation
            with nethook.TraceDict(
                module=model,
                layers=[
                    hparams.mlp_module_tmp.format(layer) if value_at_mlp else hparams.layer_module_tmp.format(layer),
                    hparams.layer_module_tmp.format(loss_layer)
                ],
                retain_input=False,
                retain_output=True,
                edit_output=edit_output_fn,
            ) as tr:
                logits = model(**input_tokens).logits

                # Compute distribution for KL divergence
                # kl_logits = torch.stack(
                #     [
                #         logits[i - len(kl_prompts), idx, :]
                #         for i, idx in enumerate(lookup_idxs[-len(kl_prompts):])
                #     ],
                #     dim=0,
                # )

                kl_log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                
                if kl_distr_init is None:
                    kl_distr_init = kl_log_probs.detach().clone()

            # Compute loss on rewriting targets
            log_probs = torch.log_softmax(logits, dim=-1)
            # probs = torch.softmax(logits, dim=-1)
            # print(logits.shape)
            # print(target)
             
            token_id = tokenizer.convert_tokens_to_ids(target)

            if(target == 'he'):
                reversed_token_id = tokenizer.convert_tokens_to_ids('she')
            else:
                reversed_token_id = tokenizer.convert_tokens_to_ids('he')

            # mask = torch.zeros(size=log_probs.shape,device=model.device)

            # for i in range(len(lookup_idxs)):   
            #     mask[i,lookup_idxs[i],token_id] = 1

            # # Aggregate total losses
            # nll_loss_each = -(log_probs * mask).sum(-1)

            # nll_loss = nll_loss_each.sum()


            nll_loss = -log_probs[torch.arange(log_probs.size(0)),lookup_idxs,token_id].sum()
            nll_loss_reversed = log_probs[torch.arange(log_probs.size(0)),lookup_idxs,reversed_token_id].sum()

            # hinge_loss = 10 * torch.clamp(1.0 + probs[torch.arange(log_probs.size(0)),lookup_idxs,reversed_token_id] - probs[torch.arange(log_probs.size(0)),lookup_idxs,token_id], min=0.0).sum()

            # kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            #     kl_distr_init[index1], kl_log_probs, log_target=True, reduction=None
            # ) * (1-mask.sum(-1))
            mask = torch.ones(size=log_probs.shape[:-1],device=model.device)
            mask[torch.arange(log_probs.size(0)),lookup_idxs] = 0

            kl_loss = hparams.kl_factor * ((kl_distr_init.exp() * (kl_distr_init-kl_log_probs)).sum(-1) * mask).mean(-1).sum()
            # kl_loss = hparams.kl_factor * ((kl_log_probs.exp() * (kl_log_probs-kl_distr_init)).sum(-1) * (1-mask.sum(-1))).sum()

            weight_decay =  hparams.v_weight_decay * (torch.norm(delta + delta_shared,dim=-1) ** 2 / torch.norm(target_init,dim=-1)).sum()

            orthogonal_loss = torch.tensor(0.0)
            
            if hasattr(hparams, 'orthogonal_constraint') and hparams.orthogonal_constraint and past_delta is not None and batch_id > 0:
                batch_id = torch.tensor(batch_id)
                delta_normed = delta / (torch.norm(delta) + 1e-8)
                orthogonal_loss = hparams.orthogonal_constraint * torch.norm(past_delta[:batch_id,:] @ delta_normed) / torch.sqrt(batch_id)

            loss = nll_loss + nll_loss_reversed + kl_loss + weight_decay + orthogonal_loss
            # loss = hinge_loss + kl_loss + weight_decay + orthogonal_loss
            # loss = nll_loss + weight_decay + orthogonal_loss

            # print(
            #     f"loss ({target}) {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} + {np.round(orthogonal_loss.item(), 3)} "
            #     # f"avg prob of [{targets_dict[g_val]}] "
            #     f"{torch.exp(-nll_loss_each).mean().item()}"
            # )

            # if loss < 5e-2:
            #     he_token_id = tokenizer.convert_tokens_to_ids('he')
            #     she_token_id = tokenizer.convert_tokens_to_ids('she')
            #     nll_loss_he = log_probs[torch.arange(log_probs.size(0)),lookup_idxs,he_token_id]
            #     nll_loss_she = log_probs[torch.arange(log_probs.size(0)),lookup_idxs,she_token_id]

            #     print(target.capitalize())
            #     print("he probs average: ",torch.exp(nll_loss_he).mean().item())
            #     print("she probs average: ",torch.exp(nll_loss_she).mean().item())

            #     print("he probs multiple: ",torch.exp(nll_loss_he.mean()).item())
            #     print("she probs multiple: ",torch.exp(nll_loss_she.mean()).item())
            #     print("=====================================")
            #     continue

            if it == hparams.v_num_grad_steps - 1:
                he_token_id = tokenizer.convert_tokens_to_ids('he')
                she_token_id = tokenizer.convert_tokens_to_ids('she')
                nll_loss_he = log_probs[torch.arange(log_probs.size(0)),lookup_idxs,he_token_id]
                nll_loss_she = log_probs[torch.arange(log_probs.size(0)),lookup_idxs,she_token_id]

                print(target.capitalize())
                print("he probs average: ",torch.exp(nll_loss_he).mean().item())
                print("she probs average: ",torch.exp(nll_loss_she).mean().item())

                print("he probs multiple: ",torch.exp(nll_loss_he.mean()).item())
                print("she probs multiple: ",torch.exp(nll_loss_she.mean()).item())
                print("=====================================")

            # Backpropagate
            # loss.requires_grad = True # just for debugging
            loss.backward()
            opt.step()
            
            # print('='*100)

            # target_mean = torch.tensor(target_init).mean()
            # target_mean = torch.zeros(target_init[0].shape,device=model.device)
            # for i in range(len(target_init)):
            #     target_mean += target_init[i]
            # target_mean /= len(target_init)
            # Project within L2 ball
            max_norm = (hparams.clamp_norm_factor * target_init.norm(dim=-1))
            normalized_deltas = (delta + delta_shared).norm(dim=-1)
            normalized_indexes = (normalized_deltas > max_norm)
            max_norm = max_norm.unsqueeze(-1)
            normalized_deltas = normalized_deltas.unsqueeze(-1)
            # print(delta.shape)
            # print(delta_shared.shape)
            # print(target_init.shape)
            # print(delta[normalized_indexes].shape)
            # print(max_norm[normalized_indexes].shape)
            # print(normalized_deltas[normalized_indexes].shape)
            if normalized_indexes.any():
                with torch.no_grad():
                    delta[normalized_indexes] = delta[normalized_indexes] * max_norm[normalized_indexes] / normalized_deltas[normalized_indexes]
                    delta_shared[normalized_indexes] = delta_shared[normalized_indexes] * max_norm[normalized_indexes] / normalized_deltas[normalized_indexes]

    targets = [target_init + delta + delta_shared for delta in deltas]
    # targets = [target_init + delta for delta in deltas]
    # print("Optimization done")

    (_, cur_outputs) = get_module_input_output_at_words(
        model,
        tokenizer,
        contexts=[request['text'] for request in context_templates],
        words=[request["subject"] for request in context_templates],
        layer=layer,
        module_template=hparams.mlp_module_tmp if value_at_mlp else hparams.layer_module_tmp,
        fact_token_strategy=hparams.fact_token
    )

    # print(cur_outputs)
    # print(cur_outputs.shape)
    cur_output = cur_outputs.mean(0)
    # print(cur_output)
    # print(cur_output.shape)
    
    if torch.cuda.is_available():
        targets = [target.to(model.device) for target in targets]
        delta_shared = delta_shared.to(model.device)
        cur_output = cur_output.to(model.device)
        cur_outputs = cur_outputs.to(model.device)

    rel_targets = [target - cur_output - delta_shared for target in targets]
    if torch.cuda.is_available():
        rel_targets = [rt.to(model.device) for rt in rel_targets]
    
    # print(deltas)
    # print(f"here is the rel shape {rel_targets[0].shape}")

    del cur_outputs, cur_output
    torch.cuda.empty_cache()

    return dict(zip(gender_values, targets)),dict(zip(gender_values,rel_targets))