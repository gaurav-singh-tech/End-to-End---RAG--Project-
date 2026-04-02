  #HERE WE WILL DO ESSENTIAL STUFFS FOR OUR PROJECT LIKE CREATING VECTOR DB, CREATING EMBEDDINGS, SPLITTING DOCUMENTS INTO CHUNKS, LOADING DOCUMENTS, ETC.

#Load pdf
#Split into chunks
#Create enbeddings
#Store the embeddings in chroma db



from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader #It will load the text file and convert it into a list of documents. Each document will be represented as a 
from langchain_text_splitters import  RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings


load_dotenv()

loader= PyPDFLoader("document loaders/Think_and_Grow_Rich.pdf") #Here, we are creating an instance of the PyPDFLoader class and passing the path to the PDF file that we want to load. The load() method will read the contents of the file and return a list of documents. Each document will be represented as a 
docs= loader.load() #The load() method will read the contents of the file and return a list of documents. Each document will be represented as a string.

splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
    )


chunks=splitter.split_documents(docs)

embedding_model= MistralAIEmbeddings(model="mistral-embed")

vector_store= Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db" #Here, we are specifying the directory where the vector store will be persisted. This means that the vector store will be saved to disk and can be loaded again later without having to recreate it from scratch.
)