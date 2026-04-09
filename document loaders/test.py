from langchain_community.document_loaders import TextLoader #It will load the text file and convert it into a list of documents. Each document will be represented as a string.

data= TextLoader("document loaders/notes.txt") #Here, we are creating an instance of the TextLoader class and passing the path to the text file that we want to load. The load() method will read the contents of the file and return a list of documents. Each document will be represented as a string.

docs= data.load() #The load() method will read the contents of the file and return a list of documents. Each document will be represented as a string.

#print(docs[0]) #This will print the first document in the list of documents.

print(docs[0].page_content) #We only want to see page content not the metadeata

#print(len(docs)) #This will print the number of documents in the list of documents.