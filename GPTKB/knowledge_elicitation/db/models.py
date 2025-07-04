from datetime import datetime
from enum import Enum

import sqlalchemy as sa
from sqlalchemy import Index
from sqlmodel import Field, SQLModel


class NodeType(Enum):
    UNDEFINED = "undefined"
    LITERAL = "literal"
    INSTANCE = "instance"


class JobType(Enum):
    ELICITATION = "elicitation"
    NAMED_ENTITY_RECOGNITION = "ner"


class Node(SQLModel, table=True):
    name: str = Field(primary_key=True)
    type: str = Field(
        default=NodeType.UNDEFINED.value,
        index=True
    )

    batch_id: str | None = Field(default=None, foreign_key="batch.id",
                                 index=True)

    creating_batch_id: str | None = Field(default=None,  # seed subject
                                          foreign_key="batch.id",
                                          index=True)

    first_parent: str | None = Field(default=None,  # seed subject
                                     index=True)

    bfs_level: int = Field(nullable=False, index=True)

    created_at: datetime | None = Field(
        default=None,
        sa_type=sa.DateTime(timezone=True),
        sa_column_kwargs={"server_default": sa.func.now()},
        nullable=False,
    )

    def __repr__(self):
        return f"< Node : {self.name} >"


class Batch(SQLModel, table=True):
    id: str = Field(primary_key=True)
    input_file_id: str
    status: str = Field(index=True)
    output_file_id: str | None = Field(default=None)

    job_type: str = Field(index=True, nullable=False)

    created_at: datetime | None = Field(
        default=None,
        sa_type=sa.DateTime(timezone=True),
        sa_column_kwargs={"server_default": sa.func.now()},
        nullable=False,
    )

    def __repr__(self):
        return f"Batch {self.id} ({self.status})"


class Predicate(SQLModel, table=True):
    name: str = Field(primary_key=True)

    creating_batch_id: str = Field(nullable=False,
                                   foreign_key="batch.id",
                                   index=True)

    created_at: datetime | None = Field(
        default=None,
        sa_type=sa.DateTime(timezone=True),
        sa_column_kwargs={"server_default": sa.func.now()},
        nullable=False,
    )

    def __repr__(self):
        return f"< Predicate : {self.name} >"


class Triple(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)

    subject: str = Field(index=True, nullable=False)
    predicate: str = Field(index=True, nullable=False)
    object: str = Field(index=True, nullable=False)

    creating_batch_id: str = Field(nullable=False,
                                   foreign_key="batch.id",
                                   index=True)

    created_at: datetime | None = Field(
        default=None,
        sa_type=sa.DateTime(timezone=True),
        sa_column_kwargs={"server_default": sa.func.now()},
        nullable=False,
    )

    __table_args__ = (
        Index(
            "ix_triple_subject_predicate_object",
            "subject", "predicate", "object",
            unique=True
        ),
    )


class FailedSubject(SQLModel, table=True):
    name: str = Field(primary_key=True)
    error: str = Field(index=True, nullable=False)
    batch_id: str = Field(index=True, nullable=False, foreign_key="batch.id")

    created_at: datetime | None = Field(
        default=None,
        sa_type=sa.DateTime(timezone=True),
        sa_column_kwargs={"server_default": sa.func.now()},
        nullable=False,
    )
