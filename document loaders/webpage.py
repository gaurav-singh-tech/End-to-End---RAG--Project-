

from langchain_community.document_loaders import WebBaseLoader #It will load the web page and convert it into a list of documents. Each document will be represented as a string.

url="https://www.apple.com/in/macbook-pro/"

data=WebBaseLoader(url) #Here, we are creating an instance of the WebBaseLoader class and passing the URL of the web page that we want to load. The load() method will read the contents of the web page and return a list of documents. Each document will be represented as a string.

docs=data.load() #The load() method will read the contents of the web page and return a list of documents. Each document will be represented as a string.

print(len(docs))

print(docs[0].page_content)