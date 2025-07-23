import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

page_url = "https://se.moevm.info/doku.php/diplomants:start:slides_checklist_etu#%D1%87%D1%82%D0%BE_%D0%BB%D1%8E%D0%B1%D0%B8%D1%82_%D1%81%D0%BF%D1%80%D0%B0%D1%88%D0%B8%D0%B2%D0%B0%D1%82%D1%8C_%D0%BA%D0%BE%D0%BC%D0%B8%D1%81%D1%81%D0%B8%D1%8F_%D0%B8_%D0%BD%D0%B5_%D0%BF%D0%BE%D0%B4%D1%81%D1%82%D0%B0%D0%B2%D0%B8%D1%82%D1%8C_%D1%81%D0%B5%D0%B1%D1%8F_%D0%B2_%D1%81%D0%BB%D0%B0%D0%B9%D0%B4%D0%B0%D1%85"

def load_and_split_documents():
    loader = WebBaseLoader(web_paths=[page_url])
    web_pages = loader.load()
    plain_text = web_pages[0].page_content

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n"],
    )
    chunks = splitter.split_documents([Document(page_content=plain_text)])

    return chunks