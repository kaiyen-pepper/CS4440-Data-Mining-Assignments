# -------------------------------------------------------------------------
# AUTHOR: Kaitlin Yen
# FILENAME: similarity.py
# SPECIFICATION: Use clustering to find the two most similar documents from cleaned_documents.csv
#                based on cosine similarity
# FOR: CS 4440 (Data Mining) - Assignment #1
# TIME SPENT: 5 hours
# -----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy or pandas.
#You have to work here only with standard dictionaries, lists, and arrays

# Importing some Python libraries
import csv
from sklearn.metrics.pairwise import cosine_similarity

documents = []

#reading the documents in a csv file
with open('cleaned_documents.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         documents.append(row)

#Building the document-term matrix by using binary encoding.
#You must identify each distinct word in the collection using the white space as your character delimiter.
import re
docTermMatrix = []
uniqueWords = []

for row in documents:
   for words in row:
      wordList = re.findall(r"[a-zA-Z]+", words)
      for word in wordList:
         uniqueWords.append(word)
uniqueWords = list(set(uniqueWords))

for row in documents:
   tempRow = []
   for word in uniqueWords:
      if word in row[1].split():
         tempRow.append(1)
      else:
         tempRow.append(0)
   docTermMatrix.append(tempRow)

# Compare the pairwise cosine similarities and store the highest one
# Use cosine_similarity([X], [Y]) to calculate the similarities between 2 vectors
highestSimilarity = -1
documentPair = []
for x in range(len(docTermMatrix)):
   for y in range(x+1, len(docTermMatrix)):
      tempCosine = cosine_similarity([docTermMatrix[x]], [docTermMatrix[y]])[0][0]
      if tempCosine > highestSimilarity:
         highestSimilarity = tempCosine
         documentPair = [x, y]         

# Print the highest cosine similarity following the information below
# The most similar documents are document 10 and document 100 with cosine similarity = x
print(f"The most similar documents are document {documentPair[0]} and document {documentPair[1]} with cosine similarity = {highestSimilarity}.")