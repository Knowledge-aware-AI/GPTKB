from db.models import NodeType


class AbstractPrompterParser:
    def get_elicitation_prompt(self, subject_name: str) -> dict:
        """
        Get a JSON object to be used as API request to OpenAI.
        :param subject_name: The target subject.
        :return: A JSON object.
        """
        raise NotImplementedError

    def parse_elicitation_response(self, response: str) -> list[dict]:
        """
        Parse the API response from OpenAI to extract triples. The result is a list of dictionaries, each dictionary
        containing a generated triple, with the keys "subject", "predicate" and "object", and an additional key "subject_name",
        which is the original subject name sent to the API. For example:
        {"subject_name": "Vannevar Bush", "subject": "Vannevar Bush", "predicate": "bornIn", "object": "1890"}
        :param response: The API response.
        :return: A list of dictionaries.
        """
        raise NotImplementedError

    def get_ner_prompt(self, entities: list[str]) -> dict:
        """
        Get a JSON object to be used as API request to OpenAI for named entity classification.
        :param entities: The entities to classify.
        :return: A JSON object.
        """
        raise NotImplementedError

    def parse_ner_response(self, response: str) -> dict[str, NodeType]:
        """
        Parse the API response from OpenAI to get node types. The result is a dictionary where the keys are the entities
        and the values are the node types. For example:
        {"Vannevar Bush": NodeType.INSTANCE, "1890": NodeType.LITERAL}
        :param response: The API response.
        :return: A dictionary.
        """
        raise NotImplementedError
