from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.document_loaders.chromium import AsyncChromiumLoader
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.document_transformers.html2text import Html2TextTransformer
from langchain.vectorstores.faiss import FAISS
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts.prompt import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
import asyncio
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from search_engine.search_engine_retrieve import SearchEngineRetriever


parser = argparse.ArgumentParser()
parser.add_argument('--pretrain_model_path',
                    type=str,
                    default='/private/model/qwen/Qwen1.5-7B-Chat',
                    required=False)
parser.add_argument('--device',
                    type=torch.device,
                    default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                    required=False)
parser.add_argument('--torch_dtype',
                    type=str,
                    default='bf16',
                    required=False,
                    help='[fp16, bf16]')

args = parser.parse_args()

def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        args.pretrain_model_path,
        torch_dtype=torch.bfloat16 if args.torch_dtype == 'bf16' else torch.float16,
        device_map='auto',
        attn_implementation='flash_attention_2'
    )
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model_path)
    model.eval()
    return model, tokenizer

def generate(model, tokenizer, text):
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': text}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    with torch.no_grad():
        model_inputs = tokenizer([text], return_tensors='pt').to(args.device)
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response

def pdf_retriever(file_path, query, k=4):
    loader = PyPDFLoader(file_path)
    all_docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    chunked_documents = text_splitter.split_documents(all_docs)
    vector_db = FAISS.from_documents(chunked_documents, HuggingFaceBgeEmbeddings(model_name='/private/model/BAAI/bge-large-zh-v1.5'))
    docs = vector_db.similarity_search(query, k)
    page = list(set([docs.metadata['page'] for docs in docs]))
    page.sort()
    reference = [docs.page_content for docs in docs]
    return reference

def url_retriever(url_list, query, k=4):
    # url_list = [
    #     'http://paper.people.com.cn/rmrb/html/2024-04/27/nw.D110000renmrb_20240427_1-01.htm',
    #     'http://paper.people.com.cn/rmrb/html/2024-04/27/nw.D110000renmrb_20240427_7-01.htm',
    #     'http://www.news.cn/fortune/2023-12/21/c_1130038559.htm',
    #     'http://www.news.cn/legal/2023-09/11/c_1129856020.htm'
    # ]
    loader = AsyncChromiumLoader(url_list)
    docs = loader.load()
    html2text = Html2TextTransformer()
    doc_transformed = html2text.transform_documents(docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    chunked_documents = text_splitter.split_documents(doc_transformed)

    vector_db = FAISS.from_documents(chunked_documents, HuggingFaceBgeEmbeddings(model_name='/private/model/BAAI/bge-large-zh-v1.5'))
    docs = vector_db.similarity_search(query, k)
    reference = [docs.page_content for docs in docs]
    return reference

def text_retriever(text_list, query, k=4):
    all_docs = []
    for text in text_list:
        doc = Document(page_content=text)
        all_docs.append(doc)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    chunked_documents = text_splitter.split_documents(all_docs)
    vector_db = FAISS.from_documents(chunked_documents, HuggingFaceBgeEmbeddings(model_name='/private/model/BAAI/bge-large-zh-v1.5'))
    docs = vector_db.similarity_search(query, k)
    reference = [docs.page_content for docs in docs]
    return reference
    
async def bing_search(query):
    retriever = SearchEngineRetriever(searchers='bing')
    results = await retriever.retrieve(query=query, searcher='bing', pre_filter=True, filter_min=50, filter_max=1200)
    urls = []
    texts = []
    titles = []
    for result in results:
        urls.append(result['url'])
        texts.append(result['text'])
        titles.append(result['title'])
    return {
        'urls': urls,
        'texts': texts,
        'titles': titles
    }

if __name__ == '__main__':
    PROMPT_TEMPLATE = '参考信息：\n{reference}\n\n请从参考信息中筛选关联内容回答：{query}'
    query = '北京买房什么时候开始限购的？'

    text_list = asyncio.run(bing_search(query))['texts']
    reference = text_retriever(text_list, query, k=4)
    print(reference)
    
    text = PROMPT_TEMPLATE.format(reference=reference, query=query)

    model, tokenizer = load_model()
    ori_response = generate(model, tokenizer, query)
    print('origin response:\n\n{}\n\n'.format(ori_response))
    rag_response = generate(model, tokenizer, text)
    print('rag response:\n\n{}\n\n'.format(rag_response))

    “”“
    origin response:

    北京市于2013年9月30日开始实施购房限购政策，主要针对非北京户籍居民家庭购房，即“认房又认贷”的政策。该政策要求在本市无住房的非本市户籍家庭，需要提供五年内连续在北京缴纳社会保险或者个人所得税的记录，且家庭名下在京无住房才能购买一套住房。此后，北京市政府根据房地产市场情况，适时调整了限购政策。如果您需要最新的购房政策信息，建议您咨询当地的房地产管理部门或关注官方发布的最新通知。


    rag response:
    
    北京限购房子是从2011年开始的。具体来说，只要是2011年2月17日0：00之后进行网签的住房，都执行了新的限购政策。
    ”“”

