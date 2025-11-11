#!/usr/bin/env python3

import csv,torch,random,os, numpy as np, pandas as pd
from tokenizers import Tokenizer
from transformers import RobertaConfig, RobertaForMaskedLM, PreTrainedTokenizerFast
from tqdm import tqdm
import matplotlib.pyplot as plt

# Files
input_file = "../../../data/MIBiG_domain_list.txt"
fin = '../../../models/vocab_pfam_product_hmmer33.json'
os.makedirs("output",exist_ok=True)

# tokenizer setting
tokenizer_obj = Tokenizer.from_file(fin)
tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_obj)
tokenizer.add_special_tokens({"mask_token":"[MASK]","pad_token":"[PAD]","sep_token":"[SEP]"})

# model setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RobertaForMaskedLM.from_pretrained("../../../models/model_IV_roberta_bacfun/out.bacfun")
model.to(device).eval()

# Get vocabs
vocab = list(tokenizer.get_vocab().keys())

# Set domain number range
token_count_ranges = list(range(6,15))

# Process per line
with open(input_file,'r',encoding='utf-8') as f:
    lines = f.readlines()

for token_count in token_count_ranges:
    all_results = []
    for line in tqdm(lines,desc=f"Processing lines with {token_count} tokens"):
        line = line.strip()
        if not line: continue

        tokens = line.split()[2:]
        bgc_id = line.split()[0]

        candidates = [token for token in tokens if token != "[SEP]"]

        if not (token_count == len(candidates)): continue

        max_probability = -1
        target_token = None

        for idx in range(len(tokens)):
            token = tokens[idx]
            masked_tokens = tokens.copy()
            if masked_tokens[idx] == "[SEP]": continue
            #print(masked_tokens[idx])
            masked_tokens[idx] = "[MASK]"

            tokens_id = tokenizer.convert_tokens_to_ids(masked_tokens)
            tokens_tensor = torch.tensor([tokens_id]).to(device)

            with torch.no_grad():
                outputs = model(tokens_tensor)
                predictions = outputs[0]

            # Get prediction probability at MASK
            # Set the domain with the highest probability as target
            softmax_scores = torch.softmax(predictions[0,idx],dim=-1)
            token_id = tokenizer.convert_tokens_to_ids([token])[0]

            if softmax_scores[token_id].item() > max_probability:
                max_probability = softmax_scores[token_id].item()
                target_token = token
                target_idx = idx

        # Skip if no target token
        if target_token is None: continue

        # Change substitution number from 0 to token_count
        line_results = [bgc_id, target_token]
        for num_replacements in range(0,token_count):
            avg_probabilities = []

            # Repeat 10 times
            for _ in range(10):
                # Copy tokens for process
                modified_tokens = tokens.copy()
                if num_replacements > 0:
                    available_indices = [i for i in range(len(modified_tokens)) if i != target_idx and modified_tokens[i] != "[SEP]"]  # Omit target_idx and [SEP] indices
                    replace_indices = random.sample(available_indices,num_replacements)

                    # Random sampling from tokenizer
                    for idx in replace_indices:
                        sampled_token = random.choice(vocab)
                        modified_tokens[idx] = sampled_token

                # MASK the target token
                masked_tokens = modified_tokens.copy()
                masked_tokens[target_idx] = "[MASK]"

                # Convert tokens to tensor
                tokens_id = tokenizer.convert_tokens_to_ids(masked_tokens)
                tokens_tensor = torch.tensor([tokens_id]).to(device)

                # Predict by the model
                with torch.no_grad():
                    outputs = model(tokens_tensor)
                    predictions = outputs[0]

                # Get prediction score at MASK
                softmax_scores = torch.softmax(predictions[0,target_idx],dim=-1)
                target_token_id = tokenizer.convert_tokens_to_ids([target_token])[0]
                avg_probabilities.append(softmax_scores[target_token_id].item())

            # Calculate average probability
            mean_probability = np.mean(avg_probabilities)
            line_results.append(mean_probability)

        # Append per line
        all_results.append(line_results)

    # Output
    output_file = os.path.join(output_dir, f"prompt_analysis_{token_count}_tokens.csv")
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        headers = ["BGC_ID", "masked_token"] + [f"Num_Replacements={i}" for i in range(token_count)]
        writer.writerow(headers)
        writer.writerows(all_results)

print("Done")

#---end---

