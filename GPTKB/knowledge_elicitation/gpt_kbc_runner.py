import json
import time
from pathlib import Path

import openai
from loguru import logger
from openai import OpenAI
from openai.types import Batch as OpenAIBatch
from sqlalchemy.dialects.sqlite import insert
from sqlmodel import Session, select, create_engine, col
from tqdm import tqdm

from db.models import SQLModel, Node, Batch, NodeType, Predicate, Triple, \
    FailedSubject, JobType
from prompter_parser import AbstractPrompterParser


class GPTKBCRunner:
    def __init__(
            self,
            db_path: str,
            seed_subject: str = "Vannevar Bush",
            job_description: str = "Knowledge Elicitation",
            prompter_parser_module: AbstractPrompterParser = None,
    ):
        logger.info("Initialize the GPT-KBC runner")

        if prompter_parser_module is None:
            raise ValueError("Prompter Parser module is not provided.")

        # DB setup
        self.db_path = Path(db_path).resolve()
        self.sqlite_url = f"sqlite:///{self.db_path}"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Create DB engine with SQLite url: `{self.sqlite_url}`")
        self.db_engine = create_engine(self.sqlite_url, echo=False)

        logger.info("Create DB tables if none exists")
        SQLModel.metadata.create_all(self.db_engine)

        if self._is_db_empty():
            logger.info("The database is empty - It's a fresh start!")
            logger.info(
                f"Seed the database with the seed subject: '{seed_subject}'")
            self._seed_db(seed_subject)

        # tmp folder - to store temporary batch requests and results
        self.tmp_folder = self.db_path.parent / f"tmp_{self.db_path.stem}"
        self.tmp_folder.mkdir(exist_ok=True)

        # OpenAI setup
        self.openai_client = OpenAI()
        self.job_description = job_description

        # Prompter Parser
        self.prompter_parser_module = prompter_parser_module

    def loop(
            self,
            max_batch_size: int,
            max_queue_size: int,
            max_subjects_processed: int,
            max_ner_per_prompt: int = 20,
            retry_failed_subjects: bool = True,
    ):
        """Run the GPT-KBC runner loop."""

        logger.info(
            f"Start the GPT-KBC runner loop")

        if retry_failed_subjects:
            logger.info("Processing previously failed subjects.")
            with Session(self.db_engine) as session:
                failed_subjects = session.exec(
                    select(FailedSubject)
                ).all()

                logger.info(
                    f"Found {len(failed_subjects):,} failed subjects to retry.")

                nodes = session.exec(
                    select(Node)  # noqa
                    .where(
                        col(Node.name).in_([fs.name for fs in failed_subjects]))
                ).all()

                for node in nodes:
                    node.batch_id = None
                    session.add(node)

                session.commit()

                logger.info(
                    f"Reset {len(nodes):,} failed subjects.")

                # remove failed subjects from the database
                for fs in failed_subjects:
                    session.delete(fs)

                session.commit()

                logger.info(
                    f"Removed {len(failed_subjects):,} failed subjects from the database.")

        num_elicitation_batches_created = 0
        num_ner_batches_created = 0
        num_subjects_processed = 0
        num_triples = 0
        num_classified_entities = 0

        while num_subjects_processed < max_subjects_processed:
            # Checking the queue for outstanding batches
            logger.info("Checking the batch queue status.")
            outstanding_batch_ids, elicitation_batches, ner_batches = self._check_batch_queue()

            # process the elicitation batches
            if len(elicitation_batches) > 0:
                logger.info(
                    f"Found {len(elicitation_batches)} newly completed elicitation batches.")

                num_triples += self._process_newly_completed_batches(
                    elicitation_batches, job_type=JobType.ELICITATION)

                logger.info(
                    f"Total number of triples committed to the database: {num_triples:,}."
                )

            # process the named entity recognition batches
            if len(ner_batches) > 0:
                logger.info(
                    f"Found {len(ner_batches)} newly completed named entity recognition batches.")

                num_classified_entities += self._process_newly_completed_batches(
                    ner_batches, job_type=JobType.NAMED_ENTITY_RECOGNITION)

                logger.info(
                    f"Total number of named entity classifications: {num_classified_entities:,}."
                )

            # Check if the batch queue is full
            logger.info(
                f"Found {len(outstanding_batch_ids)} outstanding batches.")
            if len(outstanding_batch_ids) >= max_queue_size:
                logger.info(
                    f"Queue is full. Waiting for 30 seconds before checking again.")
                time.sleep(30)
                continue

            # If batch queue is not full, get the next batch of subjects from db
            batches_of_subjects_to_expand = self._get_next_batches_of_subjects(
                max_batch_size=max_batch_size,
                number_of_batches=max_queue_size - len(outstanding_batch_ids),
            )

            if batches_of_subjects_to_expand:
                for subjects_to_expand in batches_of_subjects_to_expand:
                    # Create a new batch to submit to OpenAI
                    logger.info(
                        f"Creating a new batch with {len(subjects_to_expand):,} subjects.")
                    self._create_batch(subjects_to_expand)
                    num_elicitation_batches_created += 1
                    num_subjects_processed += len(subjects_to_expand)

                    logger.info(
                        f"Total number of elicitation batches created: {num_elicitation_batches_created:,}. "
                        f"Total number of subjects queued: {num_subjects_processed:,}.")

                    if num_subjects_processed >= max_subjects_processed:
                        logger.info(
                            f"Maximum number of queued subjects reached. Exiting loop.")
                        break
            else:
                logger.info("Looking for UNDEFINED nodes to classify.")
                batches_of_nodes_to_classify = self._get_next_batches_of_nodes_to_classify(
                    max_ner_per_prompt=max_ner_per_prompt,
                    max_batch_size=max_batch_size,
                    number_of_batches=max_queue_size - len(
                        outstanding_batch_ids)
                )

                if not batches_of_nodes_to_classify:
                    if len(outstanding_batch_ids) == 0:
                        logger.info("Exiting loop.")
                        break
                    else:
                        logger.info(
                            "Waiting for 30 seconds before checking again.")
                        time.sleep(30)
                        continue

                for nodes_to_classify in batches_of_nodes_to_classify:
                    logger.info(
                        f"Creating a new NER batch with {sum([len(b) for b in nodes_to_classify]):,} phrases.")
                    self._create_ner_batch(nodes_to_classify)
                    num_ner_batches_created += 1

                    logger.info(
                        f"Total number of NER batches created: {num_ner_batches_created:,}.")

        # End of loop, wait for the queue to clear
        outstanding_batch_ids, elicitation_batches, ner_batches = self._check_batch_queue()
        while len(outstanding_batch_ids) > 0 or len(
                elicitation_batches) > 0 or len(ner_batches) > 0:
            # Process the newly completed batches
            if len(elicitation_batches) > 0:
                # process the elicitation batches
                logger.info(
                    f"Found {len(elicitation_batches)} newly completed elicitation batches.")
                num_triples += self._process_newly_completed_batches(
                    elicitation_batches, job_type=JobType.ELICITATION)
                logger.info(
                    f"Total number of triples committed to the database: {num_triples:,}."
                )

            if len(ner_batches) > 0:
                # process the named entity recognition batches
                logger.info(
                    f"Found {len(ner_batches)} newly completed named entity recognition batches.")
                num_classified_entities += self._process_newly_completed_batches(
                    ner_batches, job_type=JobType.NAMED_ENTITY_RECOGNITION)
                logger.info(
                    f"Total number of named entity classifications: {num_classified_entities:,}."
                )

            if len(outstanding_batch_ids) > 0:
                logger.info(
                    f"Found {len(outstanding_batch_ids)} outstanding batches. Rechecking in 30 seconds.")
                time.sleep(30)

            outstanding_batch_ids, elicitation_batches, ner_batches = self._check_batch_queue()

        logger.info(
            f"The GPT-KBC runner loop has ended! "
            f"Total number of batches created: {num_elicitation_batches_created:,}. "
            f"Total number of subjects processed: {num_subjects_processed:,}. "
            f"Total number of triples committed to the database: {num_triples:,}."
        )

    def _create_batch(self, subjects_to_expand: list[Node],
                      max_tries: int = 5) -> Batch:
        batch_requests = []
        for subject in subjects_to_expand:
            req = self.prompter_parser_module.get_elicitation_prompt(
                subject_name=subject.name,
            )
            batch_requests.append(req)

        with open(self.tmp_folder / "batch_requests.jsonl", "w") as f:
            for obj in batch_requests:
                f.write(json.dumps(obj) + "\n")

        # upload file
        batch_input_file = self.openai_client.files.create(
            file=open(self.tmp_folder / "batch_requests.jsonl", "rb"),
            purpose="batch"
        )

        batch_input_file_id = batch_input_file.id
        openai_batch = None
        for num_tries in range(max_tries):
            try:
                # create batch
                openai_batch = self.openai_client.batches.create(
                    input_file_id=batch_input_file_id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                    metadata={
                        "description": self.job_description
                    }
                )
                logger.info(
                    f"Batch file created successfully. Batch ID: `{openai_batch.id}`.")
                break
            except openai.RateLimitError as e:
                logger.error(f"Rate limit error: {e}")
                logger.info("Waiting for 60 seconds before retrying.")
                time.sleep(60)
                continue

        if openai_batch is None:
            raise Exception(
                f"Failed to create batch file after {max_tries} attempts.")

        # update the database
        with Session(self.db_engine) as session:
            batch = Batch(
                id=openai_batch.id,
                input_file_id=batch_input_file_id,
                status="created",
                job_type=JobType.ELICITATION.value,
            )
            session.add(batch)

            for subject in subjects_to_expand:
                subject.batch_id = openai_batch.id
                session.add(subject)

            session.commit()

        return batch

    def _get_next_batches_of_subjects(
            self, max_batch_size: int,
            number_of_batches: int) -> list[list[Node]]:
        """
        Get the next batches of subjects to process.

        - Get the named entities that are not yet processed (batch_id is None).
        - Limit the number of subjects to the max batch size.
        """

        if number_of_batches <= 0 or max_batch_size <= 0:
            return []

        max_number_of_subjects = max_batch_size * number_of_batches

        with Session(self.db_engine) as session:
            all_available_subjects = session.exec(
                select(Node)  # noqa
                .where(
                    col(Node.batch_id).is_(None),
                    col(Node.type) == NodeType.INSTANCE.value,
                )
                .order_by(Node.bfs_level)
                .limit(max_number_of_subjects)
            ).all()

            all_available_subjects = list(all_available_subjects)

            if len(all_available_subjects) == 0:
                logger.info("Found no named entities to process.")
                return []

            logger.info(
                f"Found {len(all_available_subjects):,} subjects to process. "
                f"Available slots in queue: {number_of_batches}.")

            batch_size = min(
                max_batch_size,
                int(len(all_available_subjects) / number_of_batches) + 1
            )

            batches = [all_available_subjects[i:i + batch_size] for i in
                       range(0, len(all_available_subjects), batch_size)]

            logger.info(
                f"Prepared {len(batches)} batches of size {batch_size:,}.")

            return batches

    def _check_batch_queue(self) -> tuple[
        list[str], list[OpenAIBatch], list[OpenAIBatch]]:
        """
        Check the batch queue for outstanding batches.

        - First, get all batches from the database that are in progress.
        - Then, retrieve the real-time status of these batches from OpenAI.
        - If the status of the batch is still in progress, add it to the queue.
        - If the status of the batch is completed, add it to the list of batches to be parsed.

        :return: A tuple of three lists: the list of outstanding batch IDs, the list of just completed elicitation/ner batches.
        """

        status_in_progress = [
            "created", "validating", "in_progress", "finalizing", "parsing"
        ]

        outstanding_batch_ids = []
        just_completed_elicitation_batches = []
        just_completed_ner_batches = []

        # get batches from the database
        with Session(self.db_engine) as session:
            batches = session.exec(
                select(Batch)  # noqa
                .where(col(Batch.status).in_(status_in_progress))
            ).all()

            # retrieve status of these batches from OpenAI
            # and update the database
            for batch in batches:
                openai_batch = self.openai_client.batches.retrieve(batch.id)
                batch.status = openai_batch.status
                batch.input_file_id = openai_batch.input_file_id
                batch.output_file_id = openai_batch.output_file_id

                if openai_batch.status in status_in_progress:
                    print(openai_batch.status)
                    outstanding_batch_ids.append(openai_batch.id)
                elif openai_batch.status == "completed":
                    batch.status = "parsing"  # override status
                    if batch.job_type == JobType.ELICITATION.value:
                        just_completed_elicitation_batches.append(openai_batch)
                    elif batch.job_type == JobType.NAMED_ENTITY_RECOGNITION.value:
                        just_completed_ner_batches.append(openai_batch)

                session.add(batch)

            session.commit()

        return outstanding_batch_ids, just_completed_elicitation_batches, just_completed_ner_batches

    def _process_newly_completed_batches(
            self,
            just_completed_batches: list[OpenAIBatch],
            job_type: JobType
    ) -> int:
        """
        Process the newly completed batches returned by the batch queue checking method.
        :param just_completed_batches:
        :return: The number of new triples committed to the database.
        """

        result_count = 0

        batch_ok = []
        batch_failed = []

        for openai_batch in just_completed_batches:
            try:
                if job_type == JobType.ELICITATION:
                    raw_triples = self._process_one_completed_batch(
                        openai_batch)
                    result_count += self._commit_new_triples(
                        raw_triples=raw_triples,
                        batch_id=openai_batch.id)
                elif job_type == JobType.NAMED_ENTITY_RECOGNITION:
                    node_name_type_map = self._process_one_completed_ner_batch(
                        openai_batch)
                    result_count += self._update_nodes_with_new_types(
                        node_name_type_map=node_name_type_map,
                    )
                batch_ok.append(openai_batch.id)
            except Exception as e:
                logger.warning(
                    f"Error processing the completed batch `{openai_batch.id}`: {e}")
                batch_failed.append(openai_batch.id)

        with Session(self.db_engine) as session:
            for batch_id in batch_ok:
                batch = session.get(Batch, batch_id)
                batch.status = "completed"
                session.add(batch)
            for batch_id in batch_failed:
                batch = session.get(Batch, batch_id)
                batch.status = "parsing_failed"
                session.add(batch)
            session.commit()

        return result_count

    def _seed_db(self, seed_subject: str):
        """Seed the database with the seed subject."""
        with Session(self.db_engine) as session:
            node = Node(
                name=seed_subject,
                type=NodeType.INSTANCE.value,
                bfs_level=0,
            )
            session.add(node)
            session.commit()

    def _is_db_empty(self):
        """Check if there are no nodes in the database."""
        with Session(self.db_engine) as session:
            node = session.exec(
                select(Node)
            ).first()
            return node is None

    def _process_one_completed_batch(self, openai_batch) -> list[dict]:
        """
        Process the completed batch:

        - Download the output file.
        - Parse the output into triples.
        """
        # download the result file
        logger.info(
            f"Processing a newly completed batch: `{openai_batch.id}`. Downloading results.")
        result_file_id = openai_batch.output_file_id
        batch_result = self.openai_client.files.content(result_file_id).content

        result_file_path = self.tmp_folder / "batch_results.jsonl"
        with open(result_file_path, "wb") as f:
            f.write(batch_result)

        # parse the result file
        raw_triples = []
        failed_subjects = []
        with open(result_file_path, "r") as f:
            for line in f:
                raw_triples_from_line = []
                try:
                    raw_triples_from_line = self.prompter_parser_module.parse_elicitation_response(
                        line)
                except Exception as e:
                    try:
                        obj = json.loads(line)
                        subject_name = obj["custom_id"]
                        failed_subjects.append(
                            FailedSubject(
                                name=subject_name,
                                error=str(e),
                                batch_id=openai_batch.id,
                            )
                        )
                        logger.warning(
                            f"Failed to parse the response for subject `{subject_name}`: {e}")
                    except json.JSONDecodeError:
                        logger.error(
                            f"Failed to parse OpenAI response: {line.strip()}")

                raw_triples.extend(raw_triples_from_line)

        logger.info(
            f"Found {len(raw_triples):,} raw triples in the batch results.")

        # update the database with failed subjects
        if failed_subjects:
            logger.info(
                f"Found {len(failed_subjects):,} failed subjects in the batch results.")
            with Session(self.db_engine) as session:
                session.add_all(failed_subjects)
                session.commit()

        return raw_triples

    def _process_one_completed_ner_batch(
            self,
            openai_batch: OpenAIBatch
    ) -> dict[str, NodeType]:
        """
        Process the completed named entity recognition batch:

        - Download the output file.
        - Parse the output into named entities.
        """
        # download the result file
        logger.info(
            f"Processing a newly completed NER batch: `{openai_batch.id}`. Downloading results.")
        result_file_id = openai_batch.output_file_id
        batch_result = self.openai_client.files.content(result_file_id).content

        result_file_path = self.tmp_folder / "ner_batch_results.jsonl"
        with open(result_file_path, "wb") as f:
            f.write(batch_result)

        entity_type_map = {}
        with open(result_file_path, "r") as f:
            for line in f:
                try:
                    entity_type_map.update(
                        self.prompter_parser_module.parse_ner_response(line))
                except Exception as e:
                    logger.error(f"Error parsing NER response: {e}")

        logger.info(
            f"Found {len(entity_type_map):,} named entity classifications in the batch results.")

        return entity_type_map

    def _commit_new_triples(self, raw_triples: list[dict], batch_id: str,
                            processing_batch_size: int = 5000) -> int:
        """
        Commit the new triples to the database.
        :param raw_triples: A list of raw triples, each containing the keys "subject_name", "subject", "predicate" and "object".
        :param batch_id: The ID of the batch that generated these triples.
        :return: The number of new triples committed.
        """

        # check for triple duplicate
        checked_triples = set()
        distinct_raw_triples = []
        for triple in raw_triples:
            if (triple["subject"], triple["predicate"],
                triple["object"]) in checked_triples:
                continue

            checked_triples.add(
                (triple["subject"], triple["predicate"], triple["object"]))
            distinct_raw_triples.append(triple)

        logger.info(
            f"Found {len(distinct_raw_triples):,} "
            f"distinct raw triples after deduplication.")

        num_new_nodes = 0
        num_new_predicates = 0
        num_new_triples = 0

        for i in tqdm(
                range(0, len(distinct_raw_triples), processing_batch_size),
                desc=f"Commiting results of Batch `{batch_id}`"):
            raw_triple_batch = distinct_raw_triples[i:i + processing_batch_size]

            # commit new nodes
            with Session(self.db_engine) as session:
                # get existing nodes
                subject_and_object_set = set()
                for t in raw_triple_batch:
                    subject_and_object_set.add(t["subject_name"])
                    subject_and_object_set.add(t["subject"])
                    subject_and_object_set.add(t["object"])

                existing_nodes = session.exec(
                    select(Node)  # noqa
                    .where(col(Node.name).in_(list(subject_and_object_set)))
                ).all()

                node_name_2_bfs_level = {n.name: n.bfs_level for n in
                                         existing_nodes}
                # get new nodes
                new_nodes = []
                newly_added_node_names = set()
                for triple in raw_triple_batch:
                    # OLD VERSION: a mismatched generated subject is automatically considered as new (named entity) node
                    # NEW VERSION: triples with mismatched generated subject are already discarded during output parsing - but we still keep the old logic for now
                    has_new_subject = False
                    if triple["subject"] not in node_name_2_bfs_level and \
                            triple[
                                "subject"] not in newly_added_node_names:
                        new_nodes.append(
                            Node(
                                name=triple["subject"],
                                type=NodeType.INSTANCE.value,
                                creating_batch_id=batch_id,
                                batch_id=batch_id,
                                first_parent=triple["subject_name"],
                                bfs_level=node_name_2_bfs_level[
                                              triple["subject_name"]] + 1
                            )
                        )
                        newly_added_node_names.add(triple["subject"])
                        has_new_subject = True

                    # a new object is always considered as new node (with type UNDEFINED)
                    if triple["object"] not in node_name_2_bfs_level and triple[
                        "object"] not in newly_added_node_names:
                        bfs_level = node_name_2_bfs_level[
                                        triple["subject_name"]] + 2
                        if not has_new_subject:
                            if triple["subject"] in node_name_2_bfs_level:
                                bfs_level = node_name_2_bfs_level[
                                                triple["subject"]] + 1
                        new_nodes.append(
                            Node(
                                name=triple["object"],
                                type=NodeType.UNDEFINED.value,
                                creating_batch_id=batch_id,
                                first_parent=triple["subject"],
                                bfs_level=bfs_level
                            )
                        )
                        newly_added_node_names.add(triple["object"])

                values = [
                    {
                        "name": node.name,
                        "type": node.type,
                        "creating_batch_id": node.creating_batch_id,
                        "first_parent": node.first_parent,
                        "bfs_level": node.bfs_level,
                    } for node in new_nodes
                ]

                sql_statement = (
                    insert(Node)
                    .values(values)
                    .on_conflict_do_nothing(
                        index_elements=["name"]
                    )
                )
                session.exec(sql_statement)  # noqa
                session.commit()

                num_new_nodes += len(new_nodes)

            # commit new predicates
            with Session(self.db_engine) as session:
                # get existing predicates
                existing_predicates = session.exec(
                    select(Predicate)  # noqa
                    .where(col(Predicate.name).in_(
                        list(set([t["predicate"] for t in raw_triple_batch]))))
                ).all()

                existing_predicate_names = set(
                    [p.name for p in existing_predicates])
                new_predicates = []
                for triple in raw_triple_batch:
                    if triple["predicate"] not in existing_predicate_names:
                        new_predicates.append(
                            Predicate(
                                name=triple["predicate"],
                                creating_batch_id=batch_id,
                            )
                        )
                        existing_predicate_names.add(triple["predicate"])

                session.add_all(new_predicates)
                session.commit()
                num_new_predicates += len(new_predicates)

            # commit new triples
            with Session(self.db_engine) as session:
                values = [
                    {
                        "subject": triple["subject"],
                        "predicate": triple["predicate"],
                        "object": triple["object"],
                        "creating_batch_id": batch_id,
                    } for triple in raw_triple_batch
                ]
                sql_statement = (
                    insert(Triple)
                    .values(values)
                    .on_conflict_do_nothing(
                        index_elements=["subject", "predicate", "object"]
                    )
                )
                session.exec(sql_statement)  # noqa
                session.commit()
                num_new_triples += len(values)

        logger.info(
            f"Committed {num_new_nodes:,} new nodes, "
            f"{num_new_predicates:,} new predicates, "
            f"and {num_new_triples:,} new triples to the database.")

        return num_new_triples

    def _update_nodes_with_new_types(
            self, node_name_type_map: dict[str, NodeType],
            processing_batch_size: int = 5000
    ) -> int:
        """
        Update entities with new types to the database.
        :param node_name_type_map: A dictionary of entities and their types.
        :return: The number of updated entities.
        """
        node_names = list(node_name_type_map.keys())
        num_updated_entities = 0
        for i in tqdm(
                range(0, len(node_names), processing_batch_size),
                desc="Commiting new node types"):
            name_batch = node_names[i:i + processing_batch_size]
            with Session(self.db_engine) as session:
                # get nodes
                nodes = session.exec(
                    select(Node)  # noqa
                    .where(col(Node.name).in_(name_batch))
                ).all()

                # update node types
                for node in nodes:
                    if node.name in node_name_type_map:
                        node.type = node_name_type_map[node.name].value
                        session.add(node)
                        num_updated_entities += 1

                session.commit()

        logger.info(
            f"Updated {num_updated_entities:,} entities with new types to the database.")

        return num_updated_entities

    def _get_next_batches_of_nodes_to_classify(
            self,
            max_ner_per_prompt: int,
            max_batch_size: int,
            number_of_batches: int
    ) -> list[list[list[str]]]:
        """
        Get the next batches of nodes to classify.

        :param max_ner_per_prompt: The maximum number of entities to classify per prompt.
        :param max_batch_size: The maximum number of prompts per batch.
        :param number_of_batches: The number of batches to create.
        :return:
        """
        max_num_entities = max_ner_per_prompt * max_batch_size * number_of_batches

        with Session(self.db_engine) as session:
            undefined_nodes = session.exec(
                select(Node)  # noqa
                .where(
                    col(Node.batch_id).is_(None),
                    col(Node.type) == NodeType.UNDEFINED.value,
                )
                .order_by(Node.bfs_level)
                .limit(max_num_entities)
            ).all()

            undefined_names: list[str] = [node.name for node in undefined_nodes]

            if len(undefined_names) == 0:
                logger.info("Found no UNDEFINED nodes to classify.")
                return []

            logger.info(
                f"Found {len(undefined_names):,} UNDEFINED nodes to classify.")

            prompts = []
            for i in range(0, len(undefined_names), max_ner_per_prompt):
                prompts.append(undefined_names[i:i + max_ner_per_prompt])

            batch_size = min(
                max_batch_size,
                int(len(prompts) / number_of_batches) + 1
            )

            batches_of_prompts = [prompts[i:i + batch_size] for i in
                                  range(0, len(prompts), batch_size)]

            logger.info(
                f"Prepared {len(batches_of_prompts)} batches of size {batch_size:,} prompts, each with {max_ner_per_prompt:,} entities.")

            return batches_of_prompts

    def _create_ner_batch(self, nodes_to_classify: list[list[str]],
                          max_tries: int = 5) -> Batch:
        """
        Create a named entity recognition batch.

        :param nodes_to_classify: A list of lists of entities to classify.
        :return: The created batch.
        """
        batch_requests = []
        for entities in nodes_to_classify:
            req = self.prompter_parser_module.get_ner_prompt(
                entities=entities
            )
            batch_requests.append(req)

        with open(self.tmp_folder / "ner_batch_requests.jsonl", "w") as f:
            for obj in batch_requests:
                f.write(json.dumps(obj) + "\n")

        # upload file
        batch_input_file = self.openai_client.files.create(
            file=open(self.tmp_folder / "ner_batch_requests.jsonl", "rb"),
            purpose="batch"
        )

        batch_input_file_id = batch_input_file.id
        openai_batch = None
        for num_tries in range(max_tries):
            try:
                # create batch
                openai_batch = self.openai_client.batches.create(
                    input_file_id=batch_input_file_id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                    metadata={
                        "description": self.job_description
                    }
                )
                logger.info(
                    f"Batch file created successfully. Batch ID: `{openai_batch.id}`.")
                break
            except openai.RateLimitError as e:
                logger.error(f"Rate limit error: {e}")
                logger.info("Waiting for 60 seconds before retrying.")
                time.sleep(60)
                continue

        if openai_batch is None:
            raise Exception(
                f"Failed to create batch file after {max_tries} attempts.")

        # update the database
        with Session(self.db_engine) as session:
            batch = Batch(
                id=openai_batch.id,
                input_file_id=batch_input_file_id,
                status="created",
                job_type=JobType.NAMED_ENTITY_RECOGNITION.value,
            )
            session.add(batch)
            session.commit()

        return batch
