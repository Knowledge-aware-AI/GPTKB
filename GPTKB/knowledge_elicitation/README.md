# Knowledge Elicitation

This folder contains the code for the knowledge elicitation pipeline of GPTKB.
It is responsible for extracting knowledge from a large language model (LLM) and
storing it in a relational database, i.e., SQLite.

## Setup

Set your OpenAI API key in the system's environment variable `OPENAI_API_KEY`.

## Usage

The executable script is `main.py`.

You can run it with the following arguments:

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
python main.py \
    --db_path output/gpt_kbc.db \
    --template_path_elicitation templates/prompts/prompt_elicitation.json.jinja \
    --template_path_ner templates/prompts/prompt_ner.json.jinja \
    --max_batch_size 500 \
    --max_queue_size 20 \
    --max_subjects_processed 10000 \
    --gpt_model_elicitation "gpt-4o-mini" \
    --gpt_model_ner "gpt-4o-mini"
```

Explanation of the arguments:

- `db_path`: Path to the SQLite database where the intermediate results will be
  stored.
- `template_path_elicitation`: Path to the Jinja template for the knowledge
  elicitation prompt.
- `template_path_ner`: Path to the Jinja template for the named entity
  recognition (NER) prompt.
- `max_batch_size`: Maximum number of subjects to process in a single API call.
- `max_queue_size`: Maximum number of parallel API calls to keep in the queue
  for processing.
- `max_subjects_processed`: Maximum number of subjects to process in this run.
- `gpt_model_elicitation`: The GPT model to use for the knowledge elicitation
  prompt.
- `gpt_model_ner`: The GPT model to use for the named entity recognition (NER)
  prompt.
