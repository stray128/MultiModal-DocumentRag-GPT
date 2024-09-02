from unstructured.partition.pdf import partition_pdf
import os

def extract_pdf_elements(path, fname):
    return partition_pdf(
        filename=path + fname,
        extract_images_in_pdf=True,
        infer_table_structure=True,
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        image_output_dir_path=path,
        strategy='hi_res',
        chunking_strategy="by_title",
        include_metadata=True,
        additional_partition_args={"coordinates": True},
    )

def categorize_elements(raw_pdf_elements):
    table_texts = []
    composite_texts = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            table_texts.append({element.id: str(element)})
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            composite_texts.append({element.id: str(element)})
    return composite_texts, table_texts

def convert_points_to_bbox(tups):
    x1, x2 = min([tup[0] for tup in tups]), max([tup[0] for tup in tups])
    y1, y2 = min([tup[1] for tup in tups]), max([tup[1] for tup in tups])
    return [x1, y1, x2, y2]

def generate_meta_info(raw_pdf_elements, fname):
    meta_node_info = {}
    img_nodes_info = {}
    for _, element in enumerate(raw_pdf_elements):
        if 'Table' in str(type(element)):
            meta_node_info[element.id] = [
                {
                    'unstructured_partition_id': x.id,
                    'coordinates': convert_points_to_bbox(x.metadata.coordinates.to_dict()['points']),
                    'pagenumber': x.metadata.page_number,
                    'filename': element.metadata.filename,
                    'type': 'Table',
                    'element_filename': None,
                    'element_text':str(x),
                    'layout_width': x.metadata.to_dict()['coordinates']['layout_width'],
                    'layout_height': x.metadata.to_dict()['coordinates']['layout_height']
                } for x in element.metadata.orig_elements
            ]
        if 'CompositeElement' in str(type(element)):
            for sube in element.metadata.orig_elements:
                if 'Image' in str(type(sube)):
                    img_dict = sube.metadata.to_dict()
                    image_name = img_dict['image_path'].rsplit('/', 1)[-1]
                    image_path = img_dict['image_path']
                    file_size_kb = os.path.getsize(image_path) / 1024
                    img_nodes_info[image_name] = img_dict['coordinates']['points'], fname, sube.id, file_size_kb
                    meta_node_info[sube.id] = [
                        {
                            'unstructured_partition_id': sube.id,
                            'coordinates': convert_points_to_bbox(img_dict['coordinates']['points']),
                            'pagenumber': img_dict['page_number'],
                            'filename': fname,
                            'type': 'Image',
                            'element_filename': image_name,
                            'element_text':None,
                            'layout_width': sube.metadata.to_dict()['coordinates']['layout_width'],
                            'layout_height': sube.metadata.to_dict()['coordinates']['layout_height']
                        }
                    ]
            meta_node_info[element.id] = [
                {
                    'unstructured_partition_id': x.id,
                    'coordinates': convert_points_to_bbox(x.metadata.coordinates.to_dict()['points']),
                    'pagenumber': x.metadata.page_number,
                    'filename': element.metadata.filename,
                    'type': str(type(x)).split('.')[-1].strip("'>"),
                    'element_filename': None,
                    'element_text':str(x),
                    'layout_width': x.metadata.to_dict()['coordinates']['layout_width'],
                    'layout_height': x.metadata.to_dict()['coordinates']['layout_height']
                } for x in element.metadata.orig_elements
            ]
    return meta_node_info, img_nodes_info