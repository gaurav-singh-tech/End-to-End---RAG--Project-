from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document #This is simply used to create own documents and pass it to the vector store. It is not necessary to use this class, but it can be helpful for organizing the data and metadata.
import os

load_dotenv()


#Creating the documents by own haveing page content and metadata
docs=[Document(page_content="Numpy is used for numerical operations", metadata={"source": "doc1"}),
      Document(page_content="Pandas is used for data analysis", metadata={"source": "doc2"}),
      Document(page_content="ANNis deep learning algorithm", metadata={"source": "doc3"})
      ]

#Creating embeddings for the documents using MistralAIEmbeddings
embedding_model= MistralAIEmbeddings(model="mistral-embed")

persist_dir = "chroma_db"

#Creating a vectore store using Chroma and passing the documents and embeddings to it
if not os.path.exists(persist_dir):
    vector_store= Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory="chroma_db" #Here, we are specifying the directory where the vector store will be persisted. This means that the vector store will be saved to disk and can be loaded again later without having to recreate it from scratch.
        )
    
    # Save the vector store after first creation
    vector_store.persist()

else:
    # Load existing vector store instead of recreating
    vector_store = Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding_model
    )

#As our documents are samall and are only 3 , we are creating directory in our own system and saving the vector store there. In real world applications, you would typically have a large number of documents and you would want to save the vector store to a more permanent location, such as a database or cloud storage.
#This will create a folder name chroma-db in our venv, which will acts as a vectore store
#Now inside chroma-db, we have chromasqllite, which stre structured data and the index, because our documents also has page content and metadata, so it will be stored in the chromasqllite file. The index file is used to store the index of the vector store, which is used for efficient retrieval of similar documents based on their embeddings.
#And the folder inside chroma-db is the collection folder, which is used to store the collection of documents and their embeddings. Each collection can have its own set of documents and embeddings, and you can create multiple collections within the same vector store if needed.


result= vector_store.similarity_search("what is used for data analysis?", k=2) #k=2 means we want to retrieve the top 2 most similar documents based on the similarity of their embeddings to the query embedding.
#Vectore stores are not responsible for answering your questions, they  are responsible for retrieving your information.
#LLM is responsible for answering your question.
#It will do the similarity search of the question embeddings with the all document embeddings in vector DB


for r in result:
    print(r.page_content) #This will print the page content of the retrieved documents. It will return the most similar document to the question, which is "Pandas is used for data analysis" in this case. It will also return the metadata of the document, which is {"source": "doc2"} in this case.


#Now if you rerun the same document embeddings will be stored again in vector DB because we are creating the vector store from documents again. To avoid this, we can use the persist() method of the vector store to save the vector store to disk after creating it for the first time. Then, we can use the load() method to load the vector store from disk in subsequent runs without having to recreate it from scratch. This way, we can avoid storing duplicate embeddings in the vector DB and improve the efficiency of our application.
#Simply it means “Rerunning will store duplicate embeddings of same document"

#Using retriever
retriever=vector_store.as_retriever() #The as_retriever() method is used to convert the vector store into a retriever object, which can be used to perform similarity search and retrieve relevant documents based on a query. The retriever object provides a convenient interface for interacting with the vector store and retrieving relevant information based on the similarity of document embeddings to the query embedding.

docs=retriever.invoke("Explain deep learning")

for d in docs:
    print(d.page_content) #This will print the page content of the retrieved documents based on the query "Explain deep learning". It will return the most similar document to the query, which is "ANNis deep learning algorithm" in this case. It will also return the metadata of the document, which is {"source": "doc3"} in this case.