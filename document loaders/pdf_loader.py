from langchain_community.document_loaders import PyPDFLoader

data=PyPDFLoader("document loaders/Attention is all you need.pdf") #path of the pdf

docs=data.load()

print(docs)

print(len(docs)) #This will print the number of pages in the pdf. Each page will be represented as a document in the list of documents.

print(docs[10].page_content) #This will print the content of the last page i.e, 11th page of the pdf. We only want to see page content not the metadeata