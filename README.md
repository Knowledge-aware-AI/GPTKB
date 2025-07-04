# Welcome to GPTKB

## Overview

GPTKB is a large general-domain knowledge base (KB) entirely from a large
language model (LLM). It demonstrates the feasibility of large-scale KB
construction from LLMs, while highlighting specific challenges arising around
entity recognition, entity and property canonicalization, and taxonomy
construction.

Based on GPT-4.1, GPTKB contains 100 million triples for more than 6.1 million
entities, at a cost 10x less than previous KBC projects.
GPTKB is a landmark for two fields:

- For NLP, for the first time, it provides constructive insights into the
  knowledge (or beliefs) of LLMs.
- For the Semantic Web, it shows novel ways forward for the long-standing
  challenge of general-domain KB construction.

This repository contains the code for KB construction from LLMs to materialize
entity-centric parametric knowledge of LLMs.

Check out our project page for more details and
demonstrations: [https://gptkb.org/](https://gptkb.org/).

## Getting Started

### Install

Please use **Python 3.11+**. Install the required packages by running:

```bash
pip install -r requirements.txt
```

Don't forget to have your OpenAI API key in the environment variable
`OPENAI_API_KEY`.

### Run

## Usage

The executable script is `main.py`. You can run it with the following arguments:

```bash
python main.py \
    --db_path <path_to_db> \
    --template_path_elicitation <path_to_template_for_elicitation> \
    --template_path_ner <path_to_template_for_ner> \
    --max_batch_size <max_batch_size> \
    --max_queue_size <max_queue_size> \
    --max_subjects_processed <max_subjects_processed> \
    --gpt_model_elicitation <gpt_model_for_elicitation> \
    --gpt_model_ner <gpt_model_for_ner>
```

For example:

```bash
mkdir output
cd output
python main.py \
    --db_path output/gpt_kbc.db \
    --template_path_elicitation templates/prompts/prompt_elicitation.json.jinja \
    --template_path_ner templates/prompts/prompt_ner.json.jinja \
    --max_batch_size 5000 \
    --max_queue_size 20 \
    --max_subjects_processed 1000 \
    --gpt_model_elicitation "gpt-4o-mini" \
    --gpt_model_ner "gpt-4o-mini"
```

## Citation

If you use this work please cite our paper:

```bibtex
@InProceedings{GPTKB,
  title={Enabling LLM Knowledge Analysis via Extensive Materialization},
  author={Hu, Yujia and Nguyen, Tuan-Phong and Ghosh, Shrestha and Razniewski, Simon},
  year={2025},
  booktitle={ACL},
}
```
