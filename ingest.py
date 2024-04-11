import os
import nest_asyncio
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import StorageContext
import chainlit as cl
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.settings import Settings
from llama_index.core.callbacks import CallbackManager
from llama_index.llms.cohere import Cohere
from llama_index.postprocessor.colbert_rerank import ColbertRerank

# NOTE: we add an extra tone_name variable here

from llama_index.core.service_context import ServiceContext

# imports
from llama_index.core.schema import MetadataMode

from llama_index.embeddings.voyageai import VoyageEmbedding
import nest_asyncio
nest_asyncio.apply()
from llama_index.vector_stores.qdrant import QdrantVectorStore

import qdrant_client

nest_asyncio.apply()
qdrant_url = os.environ.get("QDRANT_URL", "default-url-if-not-set")
qdrant_api_key = os.environ.get("QDRANT_API_KEY", "default-api-key-if-not-set")
voyage_api_key = os.environ.get("VOYAGE_API_KEY", "default-voyage-api-key-if-not-set")
cohere_api_key = os.environ.get("CO_API_KEY", "default-cohere-api-key-if-not-set")
qdrant_collection_name = os.environ.get("QDRANT_COLLECTION_NAME", "default-collection-name-if-not-set")
llama_parse_api_key = os.environ.get("LLAMA_PARSE_API_KEY")
parser = LlamaParse(
    api_key=llama_parse_api_key,  # can also be set in your env as LLAMA_CLOUD_API_KEY
    result_type="markdown",  # "markdown" and "text" are available
    verbose=True,
)

# sync
file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader(
    "./data", file_extractor=file_extractor
).load_data()


client = qdrant_client.QdrantClient(
       url= qdrant_url,
        api_key=qdrant_api_key, # For Qdrant Cloud, None for local instance
    )
model_name = "voyage-02"
voyage_api_key = voyage_api_key
embed_model = VoyageEmbedding(
model_name=model_name, voyage_api_key=voyage_api_key
    )
vector_store = QdrantVectorStore(client=client, collection_name=qdrant_collection_name)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents,vector_store=vector_store,embed_model=embed_model)
