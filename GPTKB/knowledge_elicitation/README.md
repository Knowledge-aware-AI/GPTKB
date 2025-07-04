# KBC via OpenAI Batch API

This implementation using SQLite database to store intermediate results.

## Setup

Set your OpenAI API key in the environment variable `OPENAI_API_KEY`.

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
