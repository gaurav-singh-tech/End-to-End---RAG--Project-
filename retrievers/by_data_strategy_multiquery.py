from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv



load_dotenv()

docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent minimizes the loss function."),
    Document(page_content="Gradient descent is an optimization that minimizes the loss function."),
    Document(page_content="Neural networks use gradient descent for training."),
    Document(page_content="Support Vector Machines are supervised learning algorithms."),
    Document(page_content="Cricket is a popular sport in many countries."),
    Document(page_content="Football is the most popular sport in the world."),
]

#Generating embeddings for the documents using HuggingFaceEmbeddings
embedding_model = MistralAIEmbeddings(model="mistral-embed")


#Creating a vector store using Chroma and passing the documents and embeddings to it
vectorstore = Chroma.from_documents(docs, embedding=embedding_model)

#Creating a retriever from the vector store
retriever = vectorstore.as_retriever()#Using normal retriever without specifying any strategy, it will apply similarity search by default. It will retrieve the most similar documents based on the cosine similarity of their embeddings to the query embedding.

#Creating a MultiQueryRetriever using the retriever and a language model (ChatMistralAI in this case)
#LLM here will be used to generate multiple queries from the original query, which will then be used by the retriever to retrieve relevant documents. The MultiQueryRetriever will combine the results from multiple queries to provide a more comprehensive set of retrieved documents.
llm = ChatMistralAI(model="mistral-small-latest")

#Creating a MultiQueryRetriever using the retriever and a language model (ChatMistralAI in this case)
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm
)

query = "What is gradient descent?"

#Using the multi-query retriever to retrieve documents based on the query
docs = multi_query_retriever.invoke(query, k=3) #k=3 means we want to retrieve the top 3 most relevant documents based on the multiple queries generated from the original query.
#Here even if we have set k=3, it may return less or more than 3 documents based on the multiple queries generated from the original query and their relevance to the documents in the vector store.

print("\nRetrieved Documents:\n")

for doc in docs:
    print(doc.page_content)