import uuid
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_core.documents import Document
import pickle 
import os


def save_in_memory_store(store, path):
    """Save the InMemoryStore to a file."""
    with open(path, 'wb') as f:
        pickle.dump(store, f)

def load_in_memory_store(path):
    """Load the InMemoryStore from a file."""
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def create_or_update_multi_vector_retriever(
    vectorstore, text_summaries, texts, table_summaries, tables, image_summaries, images, meta_node_info, img_nodes_info, DOCSTORE_PATH
):
    """
    Create or update retriever that indexes summaries, but returns raw images or texts
    """

    # Initialize the storage layer
    if os.path.exists(DOCSTORE_PATH):
        store = load_in_memory_store(DOCSTORE_PATH)
    else:
        store = InMemoryStore()
    
    id_key = "doc_id"

    # Create the multi-vector retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    # Helper function to add documents to the vectorstore and docstore
    def add_documents(retriever, doc_summaries, doc_contents, doc_meta):
        doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
        summary_docs = [
            Document(page_content=list(s.values())[0], metadata={id_key: doc_ids[i]})
            for i, s in enumerate(doc_summaries)
        ]
        retriever.vectorstore.add_documents(summary_docs)
        if isinstance(doc_contents[0], dict):
            content_docs = [
                {'content':list(c.values())[0], 'metadata': doc_meta[i]}
                for i, c in enumerate(doc_contents)
            ]
            retriever.docstore.mset(list(zip(doc_ids, content_docs)))
        else:
            content_docs = [
                {'content': doc_contents[i], 'metadata': doc_meta[i]}
                for i in range(len(doc_contents))
            ]
            retriever.docstore.mset(list(zip(doc_ids, content_docs)))
    
    # Add texts, tables, and images
    # Check that text_summaries is not empty before adding
    if text_summaries:
        texts_meta = [meta_node_info.get(list(text.keys())[0], {}) for text in texts]
        add_documents(retriever, text_summaries, texts, texts_meta)
    if table_summaries:
        tables_meta = [meta_node_info.get(list(table.keys())[0], {}) for table in tables]
        add_documents(retriever, table_summaries, tables, tables_meta)
    if image_summaries:
        images_meta = [img_nodes_info.get(img_file, {}) for img_file in images]
        add_documents(retriever, image_summaries, images, images_meta)
    
    # Save the updated docstore to disk
    save_in_memory_store(retriever.docstore, DOCSTORE_PATH)
    
    return retriever
