import asyncio
import time
from src.pdf_processing.pdf_processing import extract_pdf_elements, categorize_elements, generate_meta_info
from src.summarization.text_summary import generate_text_summaries
from src.summarization.image_summary import generate_img_summaries, process_image_summaries
from src.vector_store.create_retriever import create_or_update_multi_vector_retriever, load_in_memory_store
from src.rag.rag_chain import multi_modal_rag_chain_with_reranking
from langchain.retrievers.multi_vector import MultiVectorRetriever
from src.config import Config
import os

async def process_new_pdf(fpath, fname):
    try:
        print("Starting PDF processing...")

        # Get elements
        start_time = time.time()
        print("Extracting PDF elements...")
        raw_pdf_elements = extract_pdf_elements(fpath, fname)
        print(f"PDF elements extracted. Time taken: {time.time() - start_time:.2f} seconds")

        # Get text, tables
        start_time = time.time()
        print("Categorizing elements into text and tables...")
        texts, tables = categorize_elements(raw_pdf_elements)
        print(f"Elements categorized. Time taken: {time.time() - start_time:.2f} seconds")

        # Generate meta info
        start_time = time.time()
        print("Generating meta information...")
        meta_node_info, img_nodes_info = generate_meta_info(raw_pdf_elements, fname)
        print(f"Meta information generated for Composite nodes. Time taken: {time.time() - start_time:.2f} seconds")

        # Generate summaries
        start_time = time.time()
        print("Generating text summaries...")
        text_summaries, table_summaries = await generate_text_summaries(texts, tables)
        print(f"Text summaries generated. Time taken: {time.time() - start_time:.2f} seconds")

        start_time = time.time()
        print("Generating image summaries...")
        img_base64_list, image_summaries, image_info = await generate_img_summaries("figures", img_nodes_info)
        print(f"Image summaries generated. Time taken: {time.time() - start_time:.2f} seconds")

        # delete the figures
        figures_dir = "figures"
        for file_name in os.listdir(figures_dir):
            file_path = os.path.join(figures_dir, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print("All files in the 'figures' directory have been deleted.")

        start_time = time.time()
        img_nodes_info = process_image_summaries(image_summaries, img_base64_list, image_info, meta_node_info, fname)
        print(f"Meta information generated for Image nodes. Time taken: {time.time() - start_time:.2f} seconds")

        # Create or load retriever
        start_time = time.time()
        print("Creating or loading multi-vector retriever...")
        vectorstore = Config.vectorstore
        
        retriever = create_or_update_multi_vector_retriever(
            vectorstore,
            text_summaries,
            texts,
            table_summaries,
            tables,
            image_summaries,
            img_base64_list,
            meta_node_info,
            img_nodes_info,
            DOCSTORE_PATH='./docstore.pkl'
        )
        print(f"Multi-vector retriever created. Time taken: {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"Error processing PDF: {e}")
    
    return retriever

async def query_vectorstore(query, retriever = None):
    
    if retriever is None:
        start_time = time.time()
        print("Loading vector store...")
        vectorstore = Config.vectorstore
        docstore = load_in_memory_store('./docstore.pkl')

        # Create the retriever
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=docstore,
            id_key="doc_id",
        )
        print(f"Vector store loaded. Time taken: {time.time() - start_time:.2f} seconds")

    # Create RAG chain
    start_time = time.time()
    print("Creating RAG chain with reranking...")
    chain = multi_modal_rag_chain_with_reranking(retriever)
    print(f"RAG chain created. Time taken: {time.time() - start_time:.2f} seconds")

    # Run query
    start_time = time.time()
    print(f"Running query: {query}")
    result = chain(query)
    print(f"Query result obtained. Time taken: {time.time() - start_time:.2f} seconds")
    # print(result['result'])

    return result

if __name__ == "__main__":
    fpath = "/Users/ashwithrambasani/GenAI/micro1/data/"
    fname = "llama3.1_blog.pdf"
    query = "How does Llama3.1 compare against gpt-4o and Claude 3.5 Sonnet in human evals?"

    # Process new PDF and save to vector store
    asyncio.run(process_new_pdf(fpath, fname))

    # Query the vector store
    asyncio.run(query_vectorstore(query))