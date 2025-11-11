Transformer-based Platform for Biosynthetic Gene Cluster Prediction and Design
------------

This repository hosts the research code accompanying our study on transformer-based modeling of biosynthetic gene clusters (BGCs).
Our framework treats protein functional domains as linguistic tokens, allowing transformer language models to learn and predict their positional and contextual relationships within genomes.

Using a RoBERTa architecture, we trained domain-level language models on four progressively broader datasets—ranging from curated bacterial BGCs to complete bacterial and fungal genomes.
Model performance was evaluated by the accuracy of domain prediction and other reference metrics to confirm representation quality.

Beyond prediction, the platform also supports model-guided BGC design, as demonstrated in a case study on the bacterial diterpenoid cyclooctatin.
Future extensions will integrate transcriptomic, structural, and phylogenetic information to further enhance the models' generative capabilities for natural product discovery and engineering.


Key Features
-------------
Domain-as-token modeling: Representing Pfam functional domains for transformer-based contextual learning.

Leakage-safe data splitting: Prevents overlap between training and evaluation sets using MinHash–Jaccard grouping of similar BGCs.

Generative application: Facilitates exploration of model-guided BGC design and modification based on learned domain relationships.


How to Run
-----------
## Check the codes and modify file paths if required ##
1. Prepare input
# Provide, for each genome, a genomic GFF and a domain search output file (HMMer) placed together in a subdirectory under target_dir (an example directory: data/dir_raw_sample/).

python 01_prepare_input.py -i _target_dir_


2. Split data safely
# Provide a list of paths to the domain token files generated in Step 1 (absolute paths are recommended).
# Execute Steps 2/3/4 in the same directory.

python 02_leakage_safe_split_bgc.py --file-list _corpus_filelist.out_ --target _name_ --train-ratio 0.8 --block-size 512 --ngram 6 --num-perm 128 --bands 32 --jaccard-thresh 0.80 --max-bucket 200 --seed 42


3. Pre-tokenize to numeric arrays

python 03_pretokenize_indices.py --target _name_ --vocab-json _vocab.json_


4. Train a model
# Require pretoken_memmap_dataset.py (provided with codes).

python 04_pretraining.py --target _name_ --vocab-json _vocab.json_ --epochs 100 --save-every 10

5. Predict every domain in _input-file_ using a trained model
# Output: every_domain_prediction._name_.csv

python 05_every_domain_prediction.py --vocab-json _vocab.json_ --ckpt-dir _checkpoint-dir_ --title _name_ --input-file _file_to_predict_ --max-len 512

6. Generate a domain in _input-file_ using a trained model

python 06_mask_domain_prediction.py --vocab-json _vocab.json_ --ckpt-dir _checkpoint-dir_ --title _name_ --input-file _file_to_generate_ --max-len 512

# All data including processed genome data, models, and other additional materials is available on Zenodo (https://doi.org/10.5281/zenodo.17577731) with the same directory structure.


Requirements
-------------
This repository was developed and tested with Python 3.8 – 3.10.
GPU acceleration (CUDA) is recommended for model training but not required for tokenization or evaluation.


Repository Structure
---------------------
bgc-transformer/
├── README.md                  # Overview and usage instructions
├── codes/
│   ├── 01_prepare_input.py        # Generate input files
│   ├── 02_leakage_safe_split.py   # Leakage-safe MinHash–Jaccard grouping and dataset split
│   ├── 03_pretokenize_indices.py  # Convert indexed tokens into numerical arrays (memmap)
│   ├── 04_pretraining.py          # Train RoBERTa-based domain language model
│   ├── pretoken_memmap_dataset.py # Functions used in 04
│   ├── 05_every_domain_prediction.py # Predict every domain by MLM task
│   ├── 06_mask_domain_prediction.py  # Predict masked domain by MLM task
│   ├── run.02                     # Shell script example for 02
│   ├── run.03                     # Shell script example for 03
│   ├── run.04                     # Shell script example for 04
│   ├── run.05                     # Shell script example for 05
│   ├── run.06                     # Shell script example for 06
│   └── analyses
│       ├── A_class_prediction/    # Codes and outputs of BGC class prediction
│       ├── B_domain_prediction/   # Codes and outputs of domain prediction
│       ├── C_model_evaluation/    # Codes and outputs of model evaluation (comparison with DeepBGC)
│       └── D_prompt_analysis/     # Code and outputs of prompt analysis
├── models/
│   ├── vocab_pfam_product_hmmer33.json  # Tokenizer json
│   ├── model_I_roberta_bgc/             # Model I dir
│   ├── model_II_roberta_acti/           # Model II dir
│   ├── model_III_roberta_bac/           # Model III dir
│   └── model_IV_roberta_bacfun/         # Model IV dir
└── data/
    ├── genome_corpus_filelist.out  # Input example for 02
    ├── MIBiG_domain_list.txt       # MIBiG BGC list for 05
    ├── MIBiG_domain_list_acti.txt  # MIBiG BGC list including only Actinomycetes for 05
    ├── cyclooctatin.txt            # Input example for 06
    ├── pfam2vec.csv                # DeepBGC embeddings of Pfam domains (obtained from https://github.com/Merck/deepbgc/tree/master/notebooks/mibig_v3_retraining/data)
    └── dir_raw_sample/             # Input directory example for 01


#---end---

