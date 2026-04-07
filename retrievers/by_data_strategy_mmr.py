from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

docs=[
    Document(page_content="Gradient descent is an optimization algorithm used to minimize the loss function in machine learning models.", metadata={"source": "doc1"}),
    Document(page_content="Gradient descent minimizes the loss function by iteratively updating the model parameters in the direction of the steepest descent.", metadata={"source": "doc2"}),
    Document(page_content="Gradient descent is commonly used in training neural networks and other machine learning models to find the optimal parameters that minimize the error between predicted and actual values.", metadata={"source": "doc3"}),
    Document(page_content="Support Vector Machines are supervised learning models used for classification and regression analysis.", metadata={"source": "doc4"}),
    Document(page_content="Neural Networks use gradient descent for training", metadata={"source": "doc5"})
    
]  

embedding_model= MistralAIEmbeddings(model="mistral-embed")

vector_store= Chroma.from_documents(
    documents=docs,
    embedding=embedding_model
)#as you have created document (docs), within this file, it will save embeddings locally in RAM(not hard disk), unlike chroma_db which saves in hard disk


#1. Using similarity search strategy
similarity_retriever=vector_store.as_retriever(strategy="similarity", k=3) #Here, we are creating a retriever object using the as_retriever() method of the vector store. We are specifying the strategy as "similarity", which means that the retriever will use cosine similarity to retrieve top 3  relevant documents based on the similarity of their embeddings to the query embedding.

print("\n====Similarity Retriever Results====")

similarity_docs=similarity_retriever.invoke("What is gradient descent?") #The invoke() method is used to perform the retrieval based on the specified strategy. We are passing the query "What is gradient descent?" to the invoke() method, and it will return the top 3 most similar documents based on the cosine similarity of their embeddings to the query embedding.
for doc in similarity_docs:
    print(f"Source: {doc.metadata['source']}, Content: {doc.page_content}")
    
    
#2. Using MMR strategy(Maximal Marginal Relevance)

mmr_retriever=vector_store.as_retriever(strategy="mmr", k=3) #Here, we are creating another retriever object using the as_retriever() method of the vector store. We are specifying the strategy as "mmr", which stands for Maximal Marginal Relevance. This strategy aims to retrieve documents that are not only relevant to the query but also diverse from each other. The k=3 parameter indicates that we want to retrieve the top 3 relevant and diverse documents based on the MMR algorithm.

print("\n====MMR Retriever Results====")

mmr_docs=mmr_retriever.invoke("What is gradient descent?") #The invoke() method is used to perform the retrieval based on the specified strategy. We are passing the query "What is gradient descent?" to the invoke() method, and it will return the top 3 most relevant and diverse documents based on the MMR algorithm.
for doc in mmr_docs:
    print(f"Source: {doc.metadata['source']}, Content: {doc.page_content}")
    
    
    
#OUTPUT

#Using similarity search strategy, it will print doc 1, 2, 3

#Using similarity search strategy, it will print doc 1, 2, 4