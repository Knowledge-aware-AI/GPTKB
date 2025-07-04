import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from loguru import logger

from db.models import NodeType
from prompter_parser.abstract_prompter_parser import AbstractPrompterParser
from prompter_parser.exceptions import ParsingException


class PromptJSONSchema(AbstractPrompterParser):
    def __init__(
            self,
            template_path_elicitation: str,
            template_path_ner: str,
            gpt_model_elicitation: str,
            gpt_model_ner: str,
    ):
        path_elicitation = Path(template_path_elicitation)
        assert path_elicitation.exists(), f"Template file not found: {template_path_elicitation}"
        assert path_elicitation.is_file(), f"Template path is not a file: {template_path_elicitation}"

        path_ner = Path(template_path_ner)
        assert path_ner.exists(), f"Template file not found: {template_path_ner}"
        assert path_ner.is_file(), f"Template path is not a file: {template_path_ner}"

        folder = path_elicitation.parent
        env = Environment(loader=FileSystemLoader(folder))
        self.template_elicitation = env.get_template(path_elicitation.name)

        folder = path_ner.parent
        env = Environment(loader=FileSystemLoader(folder))
        self.template_ner = env.get_template(path_ner.name)

        self.gpt_model_elicitation = gpt_model_elicitation
        self.gpt_model_ner = gpt_model_ner

    def get_elicitation_prompt(self, subject_name: str) -> dict:
        return json.loads(
            self.template_elicitation.render(
                subject_name=subject_name,
                model=self.gpt_model_elicitation,
            )
        )

    def get_ner_prompt(self, entities: str) -> dict:
        phrases = "\n".join(entities)
        return json.loads(self.template_ner.render(
            phrases=phrases,
            custom_id=f"ner_{entities[0]}___{entities[-1]}"[:512],
            model=self.gpt_model_ner,
        ))

    def parse_elicitation_response(self, response: str) -> list[dict]:
        response_object = json.loads(response.strip())

        subject_name = response_object["custom_id"]
        choice = response_object["response"]["body"]["choices"][0]

        # check if the request was stopped correctly
        finish_reason = choice["finish_reason"]
        if finish_reason != "stop":
            raise ParsingException(f"finish_reason={finish_reason}")

        message = choice["message"]
        # check if the request was refused
        refusal = message["refusal"]
        if refusal:
            raise ParsingException(f"refusal={refusal}")

        output_string = message["content"]
        generated_json_object = json.loads(output_string)

        # check if the response object contains the key "facts"
        key = "facts"
        if (type(generated_json_object) != dict
                or key not in generated_json_object):
            raise ParsingException(f"Key '{key}' not found in response")

        # get the triples from the response object
        # ignore if:
        #   - the triple is not in the correct format (no error)
        #   - the subject of the generated triple is not the same as the prompted subject (i.e., subject_name)
        raw_triples = []
        for line_triple in generated_json_object[key]:
            if ("subject" in line_triple
                    and "predicate" in line_triple
                    and "object" in line_triple):
                if line_triple["subject"] == subject_name:
                    line_triple["subject_name"] = subject_name
                    raw_triples.append(line_triple)
                else:
                    logger.warning(
                        f"Triple subject mismatch: "
                        f"Prompted subject: '{subject_name}' -- "
                        f"Generated subject: '{line_triple['subject']}'"
                    )
            else:
                logger.warning(
                    f"Subject: {subject_name}. "
                    f"Invalid triple format: {line_triple}")

        return raw_triples

    def parse_ner_response(self, response: str) -> dict[str, NodeType]:
        response_object = json.loads(response.strip())

        choice = response_object["response"]["body"]["choices"][0]
        message = choice["message"]
        output_string = message["content"]
        generated_json_object = json.loads(output_string)

        results = {}
        for line_result in generated_json_object["phrases"]:
            if "phrase" in line_result and "is_ne" in line_result:
                phrase = line_result["phrase"]
                is_ne = line_result["is_ne"]
                results[
                    phrase] = NodeType.INSTANCE if is_ne else NodeType.LITERAL

        return results
