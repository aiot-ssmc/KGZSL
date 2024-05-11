import os
import openai
from openai.embeddings_utils import get_embedding

openai.organization = os.getenv("OPENAI_ORG")
openai.api_key = os.getenv("OPENAI_API_KEY")

# see more: https://openai.com/blog/new-and-improved-embedding-model
EMBEDDING_ENGINE = "text-embedding-ada-002"


def get(text: str) -> list[float]:
    """
    Get the embedding of a text.
    """
    return openai.embeddings_utils.get_embedding(text, engine=EMBEDDING_ENGINE)
