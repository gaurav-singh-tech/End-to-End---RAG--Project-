from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader #It will load the text file and convert it into a list of documents. Each document will be represented as a 
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import  RecursiveCharacterTextSplitter

load_dotenv()

data= PyPDFLoader("document loaders/Think_and_Grow_Rich.pdf") #Here, we are creating an instance of the PyPDFLoader class and passing the path to the PDF file that we want to load. The load() method will read the contents of the file and return a list of documents. Each document will be represented as a 
docs= data.load() #The load() method will read the contents of the file and return a list of documents. Each document will be represented as a string.

splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
    )



chunks=splitter.split_documents(docs)

template=ChatPromptTemplate.from_messages([("system", "You are a helpful assistant AI summarizer."),
                                  ("human", "{data}")])

model=ChatMistralAI(model="mistral-small-2506")

prompt= template.format_messages(data=chunks[0].page_content) #We only want to see page content not the metadeata

result=model.invoke(prompt) #The invoke() method will take the formatted prompt and pass it to the model for processing. The model will generate a response based on the input prompt and return it as a string.
print(result.content)