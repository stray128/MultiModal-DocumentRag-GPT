from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from src.utils.image_utils import looks_like_base64, is_image_data
import os

def re_rank_sources(sources, query):
    def get_image_summary(image_data, query):
        prompt = f"Provide an image summary for the image attached which could answer the query: '{query}'."
        chat = ChatOpenAI(model="gpt-4o-mini", max_tokens=1024, openai_api_key=os.getenv("OPENAI_API_KEY"))
        msg = chat.invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                    ]
                )
            ]
        )
        return msg.content

    source_summary_dict = {}
    processed_source_contents = []
    source_metadata = []
    for source in sources:
        source_content = source['content']
        source_meta = source['metadata']
        if looks_like_base64(source_content) and is_image_data(source_content):
            image_summary = get_image_summary(source_content, query)
            processed_source_contents.append(image_summary)
            source_summary_dict[image_summary] = source_content
            source_metadata.append(source_meta)
        else:
            processed_source_contents.append(source_content)
            source_metadata.append(source_meta)

    vectorizer = TfidfVectorizer()
    all_texts = [query] + processed_source_contents
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    ranked_indices = cosine_similarities.argsort()[::-1]
    ranked_sources = [sources[i] for i in ranked_indices]
    ranked_metadata = [source_metadata[i] for i in ranked_indices]

    return ranked_sources, ranked_metadata