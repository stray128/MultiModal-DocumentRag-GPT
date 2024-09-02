import base64
import os
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import asyncio

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

async def async_image_summarize(img_base64, prompt):
    """Make image summary"""
    chat = ChatOpenAI(model="gpt-4o-mini", max_tokens=1024, openai_api_key=os.getenv("OPENAI_API_KEY"))

    msg = await asyncio.to_thread(chat.invoke, [
        HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                },
            ]
        )
    ])
    return msg.content

async def generate_img_summaries(path, img_nodes_info):
    """
    Generate summaries and base64 encoded strings for images
    path: Path to list of .jpg files extracted by Unstructured
    """

    # Store base64 encoded images
    img_base64_list = []

    # Store image summaries
    image_summaries = []

    # Store img parsed info
    image_info = []

    # Prompt
    prompt = """You are an assistant tasked with summarizing images for retrieval. \
    These summaries will be embedded and used to retrieve the raw image. \
    Give a concise summary of the image that is well optimized for retrieval. \
    For images having charts or tables, the summary should contain the important \
    names, and dates which could help in retrieval. """

    # Apply to images
    tasks = []
    for img_file in sorted(os.listdir(path)):
        if img_file.endswith(".jpg"):
            img_path = os.path.join(path, img_file)
            if os.path.getsize(img_path) > 3 * 1024:  # Filter out files less than 3KB
                base64_image = encode_image(img_path)
                img_base64_list.append(base64_image)
                tasks.append(async_image_summarize(base64_image, prompt))
                image_info.append(img_nodes_info.get(img_file, None))

    image_summaries = await asyncio.gather(*tasks)

    # Combine image filenames with their summaries
    image_summaries = [{img_file: summary} for img_file, summary in zip(sorted(os.listdir(path)), image_summaries) if img_file.endswith(".jpg")]

    return img_base64_list, image_summaries, image_info



def process_image_summaries(image_summaries, img_base64_list, image_info, meta_node_info, fname):
    img_nodes_info = {}
    for _, b64 in enumerate(img_base64_list):
        img_name = list(image_summaries[_].keys())[0]
        page_number = img_name.split('-')[1]

        if image_info[_] is None:
            img_nodes_info[b64] = {
                'unstructured_partition_id': None,
                'coordinates': None,
                'pagenumber': page_number,
                'filename': fname,
                'type': 'Table',
                'element_filename': img_name
            }
        else:
            img_source_filename = image_info[_][1]
            coordinates = image_info[_][0]
            img_id = image_info[_][2]
            node_info = meta_node_info[img_id]
            img_nodes_info[b64] = node_info

    return img_nodes_info