from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from src.utils.image_utils import looks_like_base64, is_image_data, resize_base64_image
from src.rag.rerank import re_rank_sources
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
import os

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
            "You will be given a mixed of text, tables, and image(s) usually of charts or graphs.\n"
            "Use this information to provide accurate response to the user question. \n"
            f"User-provided question: {data_dict['question']}\n\n"
            "Text and / or tables:\n"
            f"{formatted_texts}"
        ),
    }
    messages.append(text_message)
    return [HumanMessage(content=messages)]

def multi_modal_rag_chain_with_reranking(retriever):
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini", max_tokens=1024, openai_api_key=os.getenv("OPENAI_API_KEY"))

    def chain_with_sources(query):
        sources = retriever.invoke(query)
        ranked_sources, ranked_metadata = re_rank_sources(sources, query)

        chain = (
            {
                "context": RunnableLambda(lambda _: ranked_sources) | RunnableLambda(split_image_text_types),
                "question": RunnableLambda(lambda _: query),
            }
            | RunnableLambda(img_prompt_func)
            | model
            | StrOutputParser()
        )

        result = chain.invoke({"context": ranked_sources, "question": query})
        return {"result": result, "metadata": ranked_metadata}

    return chain_with_sources