from langchain_community.retrievers import ArxivRetriever


#Creating Retriever
retriever= ArxivRetriever(
    load_max_docs=2, #number of documents to load from arxiv website through API
    load_all_available_metadata=True, #Whether to load all available metadata for the documents. If set to True, the retriever will attempt to retrieve and include all metadata associated with the documents, such as title, authors, abstract, publication date, etc. If set to False, only a subset of metadata may be loaded.
    load_all_available_links=True, #Whether to load all available links for the documents. If set to True, the retriever will attempt to retrieve and include all links associated with the documents, such as links to the full text, PDF, supplementary materials, etc. If set to False, only a subset of links may be loaded.
)

#User query arxiv
docs=retriever.invoke("What is LLM?")

#print results

for doc in docs:
    print(doc.page_content) #IT will print the content of the document that we retrieved from arxiv based on the user query. The page_content attribute contains the main text of the document, which can include the abstract, introduction, methodology, results, and conclusion sections, depending on the structure of the document. By printing doc.page_content, we can see the relevant information that was retrieved from arxiv in response to our query about LLM (Language Model).