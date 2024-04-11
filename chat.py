
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

import os
from dotenv import load_dotenv

load_dotenv()

# Now you can access the variables
qdrant_url = os.environ.get("QDRANT_URL", "default-url-if-not-set")
qdrant_api_key = os.environ.get("QDRANT_API_KEY", "default-api-key-if-not-set")
voyage_api_key = os.environ.get("VOYAGE_API_KEY", "default-voyage-api-key-if-not-set")
cohere_api_key = os.environ.get("CO_API_KEY", "default-cohere-api-key-if-not-set")
qdrant_collection_name = os.environ.get("QDRANT_COLLECTION_NAME", "default-collection-name-if-not-set")


nest_asyncio.apply()
@cl.on_chat_start
async def start():
    llm = Cohere(model="command-r-plus"
                 
                 )

    Settings.llm = llm
      
    
    colbert_reranker = ColbertRerank(
        top_n=5,
        model="colbert-ir/colbertv2.0",
        tokenizer="colbert-ir/colbertv2.0",
        keep_retrieval_score=True,
    )
    
  
    service_context = ServiceContext.from_defaults(llm=llm,embed_model=embed_model,callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]))

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
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store,embed_model=embed_model,service_context=service_context,storage_context=storage_context,batch_size=10,node_postprocessors=[colbert_reranker],)
    query_engine=index.as_chat_engine(
    similarity_top_k=2,
    chat_mode='context')

    cl.user_session.set("query_engine", query_engine )# I

@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine")  
    response = await cl.make_async(query_engine.chat)(message.content)
    
    # Create and send the text response immediately
    response_message = cl.Message(content=response.response)
    await response_message.send()  # Send the text response first

    # Then, generate and play the audio stream

    
    elements = []  # Initialize an empty list to collect elements
    label_list = []
    count = 1

    for sr in response.source_nodes:
# Assuming sr.node.get_metadata() returns a dictionary
            metadata =sr.node.get_content(metadata_mode=MetadataMode.LLM)

            # Using a more structured format for readability
            elements = [
                cl.Text(
                    name="S" + str(count),
                    content=f"Content: {sr.node.text}",                   
                    display="inline", # Changed from 'side' to 'block'
                    size='medium', # Increased size
                    color='black', # Changed color to black
                    font_style='times new roman' )
            ]
            elements2= [
                cl.Text(
                    name="S" + str(count),
                    content=f"LLM Sees:\n {metadata}",
                    display="side", # Changed from 'side' to 'block'
                    size='small', # Increased size
                    color='black', # Changed color to black
                    font_style='times new roman' )
            ]
            response_message.elements = elements + elements2
            label_list.append("S"+str(count))
            await response_message.update()
            count += 1
    response_message.content += "\n\nSources: " + ", ".join(label_list)

    await response_message.update()    

