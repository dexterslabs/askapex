# Standard library imports
import os
import sys
import json

# External library imports
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv, find_dotenv
from urllib.parse import urljoin

# langchain imports
import langchain
from langchain import OpenAI
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.chains.conversational_retrieval.prompts import (
    CONDENSE_QUESTION_PROMPT,
    QA_PROMPT,
)

# langchain community imports
from langchain_community.document_loaders import DataFrameLoader

# pinecone import
import pinecone


class CustomScraper:
    def __init__(self, base_url, storage_dir="./pages"):
        self.base_url = base_url
        self.storage_dir = storage_dir
        self.ensure_storage_dir_exists()

    def ensure_storage_dir_exists(self):
        """Ensures the storage directory exists."""
        if not os.path.exists(self.storage_dir):
            os.mkdir(self.storage_dir)

    def format_url_for_saving(self, url):
        """Adjust URL to a safe filename format."""
        clean_url = url.replace("https://", "").replace("http://", "")
        forbidden_chars = ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]
        for char in forbidden_chars:
            clean_url = clean_url.replace(char, "_")
        return clean_url

    def fetch_page_and_store(self, url, filename=None):
        """Fetch a page content and store it locally."""
        try:
            result = requests.get(url)
            if filename is None:
                filename = self.format_url_for_saving(url)
            filepath = os.path.join(self.storage_dir, f"{filename}.html")
            with open(filepath, "wb") as file:
                file.write(result.content)
            print(f"Downloaded: {url} -> {filepath}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {url}: {e}")

    def scrape_and_download_links(self):
        """Scrape all links on the base page and download their content."""
        try:
            response = requests.get(self.base_url)
            soup = BeautifulSoup(response.content, "html.parser")
            links = soup.find_all("a")

            for link in links:
                href = link.get("href")
                if href and not href.startswith("#"):
                    full_url = urljoin(self.base_url, href)
                    self.fetch_page_and_store(full_url)
        except requests.exceptions.RequestException as e:
            print(f"Failed to access {self.base_url}: {e}")


# Example usage

base_url1 = "http://support.apexsystemsinc.com/kb/faq.php?cid=1"
scraper = CustomScraper(base_url1)
scraper.scrape_and_download_links()

base_url2 = "http://support.apexsystemsinc.com/kb/faq.php?cid=2"
scraper = CustomScraper(base_url2)
scraper.scrape_and_download_links()


class DataExtractor:
    def __init__(self, storage_dir):
        self.storage_dir = storage_dir

    def _extract_link_from_filename(self, filename):
        # Reconstruct the original link from the saved filename
        base_url = "http://support.apexsystemsinc.com/"
        parts = filename.split("_")
        reconstructed_path = (
            "/".join(parts[1:]).replace(".html", "").replace("_", "=").replace("-", "&")
        )
        return base_url + reconstructed_path

    def _parse_html_file(self, filepath):
        with open(filepath, "r", encoding="utf-8") as file:
            title_text = ""
            info_text = ""
            soup = BeautifulSoup(file, "html.parser")

            # Extracting the article title
            title = soup.find("div", class_="article-title")
            info = soup.find("div", class_="thread-body")
            if title and info:
                title_text = title.get_text(strip=True)
                info_text = " ".join(info.stripped_strings)

            return title_text, info_text

    def extract_data_to_dataframe(self):
        data = []
        for filename in os.listdir(self.storage_dir):
            if not filename.endswith(".html"):
                continue

            filepath = os.path.join(self.storage_dir, filename)
            title, info = self._parse_html_file(filepath)
            if info == "":
                continue
            link = self._extract_link_from_filename(filename)

            data.append({"Article Title": title, "Information": info, "Link": link})

        return pd.DataFrame(data)


# Usage
storage_dir = "./pages"  # Update this path to where your HTML files are stored
extractor = DataExtractor(storage_dir)
df = extractor.extract_data_to_dataframe()
print(df)
# Environment variable management
load_dotenv(find_dotenv())

# Document loading
loader = DataFrameLoader(df, page_content_column="Information")
documents = loader.load()

# Document processing
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Pinecone setup
pc = pinecone.Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV")
)
index_name = "apex"
docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)

# Initialize LLM
llm = OpenAI(model_name="gpt-4", temperature=0)

# Question generator setup
question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)

# Streaming LLM setup
streaming_llm = OpenAI(
    streaming=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True,
    temperature=0,
)
doc_chain = load_qa_chain(streaming_llm, chain_type="stuff", prompt=QA_PROMPT)

# Conversational Retrieval Chain setup
qa = ConversationalRetrievalChain(
    retriever=docsearch.as_retriever(),
    combine_docs_chain=doc_chain,
    question_generator=question_generator,
)

# Interaction loop
chat_history = []
question = input("Hi! Ask me a question about Apex FAQ. ")

while True:
    result = qa({"question": question, "chat_history": chat_history})
    print("\n")
    chat_history.append((result["question"], result["answer"]))
    question = input()
