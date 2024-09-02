from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
import asyncio

async def async_invoke(chain, element):
    return await asyncio.to_thread(chain.invoke, element)

# Generate summaries of text elements
async def generate_text_summaries(texts, tables, summarize_texts=False):
    """
    Summarize text elements
    texts: List of str
    tables: List of str
    summarize_texts: Bool to summarize texts
    """

    # Prompt
    prompt_text = """You are an assistant tasked with summarizing tables and text for retrieval. \
    These summaries will be embedded and used to retrieve the raw text or table elements. \
    Give a concise summary of the table or text that is well optimized for retrieval. Table or text: {element} """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Text summary chain
    model = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=os.getenv("OPENAI_API_KEY"))
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    # Initialize empty summaries
    text_summaries = {}
    table_summaries = {}

    # Apply to text if texts are provided and summarization is requested
    if texts and summarize_texts:
        tasks = [async_invoke(summarize_chain, list(text.values())[0]) for text in texts]
        text_summaries = await asyncio.gather(*tasks)
        text_summaries = [{list(text.keys())[0]: summary} for text, summary in zip(texts, text_summaries)]
    elif texts:
        text_summaries = texts

    # Apply to tables if tables are provided
    if tables:
        tasks = [async_invoke(summarize_chain, list(table.values())[0]) for table in tables]
        table_summaries = await asyncio.gather(*tasks)
        table_summaries = [{list(table.keys())[0]: summary} for table, summary in zip(tables, table_summaries)]

    return text_summaries, table_summaries