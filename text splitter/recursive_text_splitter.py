from langchain_community.document_loaders import PyPDFLoader #It will load the PDF file and convert it into a list of documents. Each document will be represented as a string.
from langchain_text_splitters import  RecursiveCharacterTextSplitter

data= PyPDFLoader("document loaders/Attention is all you need.pdf") #Here, we are creating an instance of the PyPDFLoader class and passing the path to the PDF file that we want to load. The load() method will read the contents of the file and return a list of documents. Each document will be represented as a string.

docs= data.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000,#The chunk_size parameter specifies the maximum number of characters that each chunk should contain. In this case, we have set it to 10, which means that each chunk will contain at most 10 characters from the original document.
                                chunk_overlap=10#The chunk_overlap parameter specifies the number of characters that should overlap between consecutive chunks. In this case, we have set it to 1, which means that there will be a 1-character overlap between each chunk. This can help to ensure that important context is not lost when splitting the document into smaller pieces.
)


chunks= splitter.split_documents(docs) #The split_documents() method will take the list of documents and split each document into smaller chunks based on the specified chunk size and overlap. The resulting chunks will be returned as a list of strings, where each string represents a chunk of text from the original document.

print(len(chunks))

print(chunks[0].page_content)

