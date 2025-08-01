import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

page_url = "https://warhammer40k.fandom.com/ru/wiki/%D0%98%D0%BC%D0%BF%D0%B5%D1%80%D0%B0%D1%82%D0%BE%D1%80_%D0%A7%D0%B5%D0%BB%D0%BE%D0%B2%D0%B5%D1%87%D0%B5%D1%81%D1%82%D0%B2%D0%B0"

def load_and_split_documents():
    loader = WebBaseLoader(web_paths=[page_url])
    web_pages = loader.load()
    plain_text = web_pages[0].page_content

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n\n", "\n"],
    )
    chunks = splitter.split_documents([Document(page_content=plain_text)])

    return chunks