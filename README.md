Setup Instructions
Step 1: Install Dependencies
To set up your environment, start by installing the necessary Python packages. Run the following commands in your terminal:
```
pip install -U llama-index
pip install llama-index-vector-stores-qdrant llama-index-readers-file
pip install llama-parse
pip install llama-index-embeddings-voyageai
pip install llama-index-llms-cohere
pip install llama-index-embeddings-mistralai
pip install llama-index-llms-mistralai
pip install llama-index-postprocessor-colbert-rerank # Install this after ingesting data
pip install --quiet transformers torch # Necessary for model operations
pip install llama-index-core
pip install -U qdrant_client
pip install chainlit
```
For UI 
```
pip install -U llama-index
pip install llama-index-embeddings-voyageai
pip install llama-index-llms-cohere
pip install llama-index-core
pip install llama-index-postprocessor-colbert-rerank # Install this after ingesting data
pip install -U qdrant_client
pip install llama-index-vector-stores-qdrant 
pip install chainlit
```
for ingest
```
pip install -U llama-index
pip install llama-index-vector-stores-qdrant llama-index-readers-file
pip install llama-parse
pip install llama-index-core
pip install -U qdrant_client
```

Step 2: Configure Environment Variables
1. Copy the env.example file to a new file named .env in the root directory of your project.
2. Open the .env file in a text editor and fill in the required environment variables. This file will typically include API keys, database URLs, and other sensitive configuration options that should not be hard-coded into your application.
Step 3: Finalize Setup
After configuring your environment variables, ensure all dependencies are installed and environment variables are set correctly by running a simple test script or starting your application.
Note:
The env.example file serves as a template for the .env file. It should include all the necessary environment variables with placeholder values or instructions for obtaining those values. Do not include sensitive information in env.example.
cp env.example .env
After copying env.example to .env and configuring your environment variables, you can delete the env.example file if you prefer, but it's generally a good idea to keep it in the repository as a reference for required environment variables.
super ingest extract metadat but takes longer
