import fire

from gpt_kbc_runner import GPTKBCRunner
from prompter_parser import PromptJSONSchema


def main(
        gpt_model_elicitation: str,
        gpt_model_ner: str,
        db_path: str,
        template_path_elicitation: str,
        template_path_ner: str,
        seed_subject: str = "Vannevar Bush",
        max_batch_size: int = 5000,
        max_queue_size: int = 5,
        max_subjects_processed: int = 12000,
):
    prompter_parser_module = PromptJSONSchema(
        template_path_elicitation=template_path_elicitation,
        template_path_ner=template_path_ner,
        gpt_model_elicitation=gpt_model_elicitation,
        gpt_model_ner=gpt_model_ner,
    )

    gpt_runner = GPTKBCRunner(
        db_path=db_path,
        seed_subject=seed_subject,
        prompter_parser_module=prompter_parser_module,
    )

    gpt_runner.loop(
        max_batch_size=max_batch_size,
        max_queue_size=max_queue_size,
        max_subjects_processed=max_subjects_processed,
    )


if __name__ == "__main__":
    fire.Fire(main)
