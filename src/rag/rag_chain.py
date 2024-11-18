from typing import List
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from src.utils.image_utils import looks_like_base64, is_image_data, resize_base64_image
from src.rag.rerank import re_rank_sources
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
import os

class LineListOutputParser(BaseOutputParser[List[str]]):
    """Output parser for a list of lines."""

    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))  # Remove empty lines

def create_multi_query_chain():
    output_parser = LineListOutputParser()
    
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five 
        different versions of the given user question to retrieve relevant documents from a vector 
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of distance-based similarity search.

        Generate questions that:
        1. Make the original question more specific
        2. Break down compound questions
        3. Rephrase to focus on different aspects
        4. Add relevant context that might be implicit
        5. Use different but synonymous terms
        For example:

        Original: "Who is the CEO of the company?"
        Generated:
        - "What is the full name of the company, and who is the leading person associated with it as mentioned in the document?"
        - "Who is the founder of the company, and what role do they currently hold?"
        - "Who is the main executive leading the company, and what are their primary responsibilities?"
        - "Name the company and the person mentioned as the CEO in the document."
        - "Who is the head of the company, and what are their key achievements?"
        Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )
    
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    return QUERY_PROMPT | llm | output_parser

def split_image_text_types(docs):
    b64_images = []
    texts = []
    for doc in docs:
        if isinstance(doc, Document):
            doc = doc.page_content
        if isinstance(doc, dict):
            doc = list(doc.values())[0]
        if looks_like_base64(doc) and is_image_data(doc):
            doc = resize_base64_image(doc, size=(1300, 600))
            b64_images.append(doc)
        else:
            texts.append(doc)
    return {"images": b64_images, "texts": texts}

def img_prompt_func(data_dict):
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []

    if data_dict["context"]["images"]:
        for image in data_dict["context"]["images"]:
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
            messages.append(image_message)

    text_message = {
        "type": "text",
        "text": (
            "You are document audit specialist tasking with providing accurate information from the input provided.\n"
            "You will be given a mix of text, tables, and image(s) usually of charts or graphs.\n"
            "Use this information to provide accurate response to the user question. \n"
            f"User-provided question: {data_dict['question']}\n\n"
            "Text and / or tables:\n"
            f"{formatted_texts}"
        ),
    }
    messages.append(text_message)
    return [HumanMessage(content=messages)]

def multi_modal_rag_chain_with_reranking(retriever):
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini", max_tokens=1024)
    query_chain = create_multi_query_chain()
    
    def chain_with_sources(query):
        # Generate alternative questions
        generated_questions = query_chain.invoke({"question": query})
        all_sources = []
        
        # Get documents for each question
        for q in [query] + generated_questions:
            sources = retriever.get_relevant_documents(q)
            all_sources.extend(sources)
        
        # Rank all collected sources
        ranked_sources, ranked_metadata = re_rank_sources(all_sources, query)
        # Remove duplicates from ranked_sources and ranked_metadata
        unique_sources = []
        unique_metadata = []
        seen_sources = []

        for source, metadata in zip(ranked_sources, ranked_metadata):
            if source not in seen_sources:
                unique_sources.append(source)
                unique_metadata.append(metadata)
                seen_sources.append(source)

        ranked_sources = unique_sources
        ranked_metadata = unique_metadata
        # Only send the top ranked source and metadata
        top_ranked_source = ranked_sources[:min(4, len(ranked_sources))]
        top_ranked_metadata = ranked_metadata[:min(4, len(ranked_metadata))]

        chain = (
            {
                "context": RunnableLambda(lambda _: top_ranked_source) | RunnableLambda(split_image_text_types),
                "question": RunnableLambda(lambda _: query),
            }
            | RunnableLambda(img_prompt_func)
            | model
            | StrOutputParser()
        )

        result = chain.invoke({"context": top_ranked_source, "question": query})
        
        return {
            "result": result,
            "generated_questions": generated_questions,
            "metadata": {
                "sources": top_ranked_source,
                "ranked_metadata": top_ranked_metadata
            }
        }

    return chain_with_sources

