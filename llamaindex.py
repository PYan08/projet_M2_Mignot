import yaml

from datetime import datetime

from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.readers.file import PyMuPDFReader


CHUNK_SIZE = 1_000
CHUNK_OVERLAP = 200


def read_config(file_path):
    with open(file_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")
            return None

config = read_config("secrets/config.yaml")


llm = AzureOpenAI(
    model=config["chat"]["azure_deployment"],
    deployment_name=config["chat"]["azure_deployment"],
    api_key=config["chat"]["azure_api_key"],
    azure_endpoint=config["chat"]["azure_endpoint"],
    api_version=config["chat"]["azure_api_version"],
)

# You need to deploy your own embedding model as well as your own chat completion model
embedder = AzureOpenAIEmbedding(
    model=config["embedding"]["azure_deployment"],              # for the moment, same as deployment
    deployment_name=config["embedding"]["azure_deployment"],
    api_key=config["embedding"]["azure_api_key"],
    azure_endpoint=config["embedding"]["azure_endpoint"],
    api_version=config["embedding"]["azure_api_version"],
)

Settings.llm = llm
Settings.embed_model = embedder

vector_store = SimpleVectorStore()


def store_pdf_file(file_path: str, doc_name: str):
    """Store a pdf file in the vector store."""
    loader = PyMuPDFReader()
    documents = loader.load(file_path)

    text_parser = SentenceSplitter(chunk_size=CHUNK_SIZE)
    text_chunks = []
    doc_idxs = []
    for doc_idx, doc in enumerate(documents):
        cur_text_chunks = text_parser.split_text(doc.text)
        text_chunks.extend(cur_text_chunks)
        doc_idxs.extend([doc_idx] * len(cur_text_chunks))

    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        node = TextNode(text=text_chunk)
        src_doc = documents[doc_idxs[idx]]
        node.metadata = src_doc.metadata
        node.metadata['document_name'] = doc_name  # Ajout du nom du document dans les métadonnées
        node.metadata['insert_date'] = datetime.now()  # Ajout de la date d'insertion
        nodes.append(node)

    for node in nodes:
        node_embedding = embedder.get_text_embedding(node.get_content(metadata_mode="all"))
        node.embedding = node_embedding

    vector_store.add(nodes)

def delete_file_from_store(name: str) -> int:
    """Delete documents from the vector store by document name."""
    nodes_to_remove = [node_id for node_id, node in vector_store._node_map.items()
                       if node.metadata.get('document_name') == name]

    for node_id in nodes_to_remove:
        vector_store.delete(node_id)

    return len(nodes_to_remove)

def inspect_vector_store(top_n: int = 10) -> list:
    """Inspect the contents of the vector store."""
    docs = []
    for idx, (node_id, node) in enumerate(vector_store._node_map.items()):
        if idx < top_n:
            docs.append({
                'id': node_id,
                'document_name': node.metadata.get('document_name', 'unknown'),
                'insert_date': node.metadata.get('insert_date', 'unknown'),
                'text': node.text
            })
        else:
            break
    return docs

def get_vector_store_info():
    """Get information about the contents of the vector store."""
    nb_docs = len(vector_store._node_map)
    documents = set(node.metadata.get('document_name', 'unknown') for node in vector_store._node_map.values())

    dates = [node.metadata.get('insert_date') for node in vector_store._node_map.values() if 'insert_date' in node.metadata]
    min_date = min(dates) if dates else None
    max_date = max(dates) if dates else None

    return {
        'nb_chunks': nb_docs,
        'min_insert_date': min_date,
        'max_insert_date': max_date,
        'nb_documents': len(documents)
    }


def retrieve(question: str):
    """Retrieve documents similar to a question.

    Args:
        question (str): text of the question

    Returns:
        list[TODO]: list of similar documents retrieved from the vector store
    """
    query_embedding = embedder.get_query_embedding(question)

    query_mode = "default"
    # query_mode = "sparse"
    # query_mode = "hybrid"

    vector_store_query = VectorStoreQuery(
        query_embedding=query_embedding, similarity_top_k=5, mode=query_mode
    )

    # returns a VectorStoreQueryResult
    query_result = vector_store.query(vector_store_query)
    return query_result.nodes

    # if query_result.nodes:
    #     print(query_result.nodes[0].get_content())
    # else:
    #     print('No results')


def build_qa_messages(question: str, context: str) -> list[str]:
    messages = [
    (
        "system",
        "You are an assistant for question-answering tasks.",
    ),
    (
        "system",
        """Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.
        Use three sentences maximum and keep the answer concise.
        {}""".format(context),
    ),
    (  
        "user",
        question
    ),]
    return messages


def answer_question(question: str) -> str:
    """Answer a question by retrieving similar documents in the store.

    Args:
        question (str): text of the question

    Returns:
        str: text of the answer
    """
    docs = retrieve(question)
    docs_content = "\n\n".join(doc.get_content() for doc in docs)
    print("Question:", question)
    print("------")
    for doc in docs:
        print("Chunk:", doc.id)
        print(doc.page_content)
        print("------")
    messages = build_qa_messages(question, docs_content)
    response = llm.invoke(messages)
    return response.content
