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

+ See the [knowledge elicitation](GPTKB/knowledge_elicitation/README.md) folder
  for instructions on how to run the knowledge elicitation phase.
+ See the [knowledge consolidation](GPTKB/knowledge_consolidation/README.md)
  folder for instructions on how to run the knowledge consolidation phase.

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
