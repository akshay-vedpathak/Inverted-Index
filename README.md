# Inverted-Index
Implemented a simple Inverted Index for document retrieval for the cranfield document collection. 

## Overview
The code performs the following functions for document retrieval on the cranField documents collection. 
1. Preprocesses the data in the text collection (stop words removal, stemming, removing punctuations etc.)
2. Creates an inverted index for looking up relevant documents for each word in the vocabulary of the document collection
3. Assigns weights to each word in the documents/queries by using TF-IDF weighting scheme
4. Retrieves a list of relevant documents for the given list of queries specified in queries.txt and ranks the retrieved documents in decreasing order of their cosine similarty with the query
5. Calculates average precision and recall for the top (10/50/100/500) retrieved documents by using relevance.txt 

## Running the code  
*Python 3 and nltk are required to run the code*

Ensure all the files from the repo are present in the same directory. After checking out the project the code can be run in two ways. 

### From the command line/ terminal

python Inverted_index.py <path_to_cranfieldDocs> <path_to_query_file> <path_to_relevance_file>

### From python IDE
Open the project in any preferred IDE of your choice and run the file Inverted_index.py by passing the above arguments at runtime 

## Precision and Recall scores for queries
----------------------------------------------------------------
#### Top 10 documents in rank list<br>
Query: 1 &nbsp;&nbsp; Pr: 0.0 &nbsp;&nbsp; Re:0.0<br>
Query: 2 &nbsp;&nbsp; Pr: 0.2 &nbsp;&nbsp; Re:0.13333333333333333<br>
Query: 3 &nbsp;&nbsp; Pr: 0.2 &nbsp;&nbsp; Re:0.13333333333333333<br>
Query: 4 &nbsp;&nbsp; Pr: 0.1 &nbsp;&nbsp; Re:0.05555555555555555<br>
Query: 5 &nbsp;&nbsp; Pr: 0.1 &nbsp;&nbsp; Re:0.05263157894736842<br>
Query: 6 &nbsp;&nbsp; Pr: 0.4 &nbsp;&nbsp; Re:0.2222222222222222<br>
Query: 7 &nbsp;&nbsp; Pr: 0.6 &nbsp;&nbsp; Re:0.6666666666666666<br>
Query: 8 &nbsp;&nbsp; Pr: 0.2 &nbsp;&nbsp; Re:0.5<br>
Query: 9 &nbsp;&nbsp; Pr: 0.1 &nbsp;&nbsp; Re:0.125<br>
Query: 10 &nbsp;&nbsp; Pr: 0.2 &nbsp;&nbsp; Re:0.08333333333333333<br>
#### Avg precision: 0.21000000000000002
#### Avg recall: 0.19720760233918128
----------------------------------------------------------------
#### Top 50 documents in rank list
Query: 1 &nbsp;&nbsp; Pr: 0.0 &nbsp;&nbsp; Re:0.0<br> 
Query: 2 &nbsp;&nbsp; Pr: 0.12 &nbsp;&nbsp; Re:0.4<br>
Query: 3 &nbsp;&nbsp; Pr: 0.14 &nbsp;&nbsp; Re:0.4666666666666667<br> 
Query: 4 &nbsp;&nbsp; Pr: 0.06 &nbsp;&nbsp; Re:0.16666666666666666<br>
Query: 5 &nbsp;&nbsp; Pr: 0.14 &nbsp;&nbsp; Re:0.3684210526315789<br>
Query: 6 &nbsp;&nbsp; Pr: 0.14 &nbsp;&nbsp; Re:0.3888888888888889<br>
Query: 7 &nbsp;&nbsp; Pr: 0.16 &nbsp;&nbsp; Re:0.8888888888888888<br>
Query: 8 &nbsp;&nbsp; Pr: 0.06 &nbsp;&nbsp; Re:0.75<br>
Query: 9 &nbsp;&nbsp; Pr: 0.12 &nbsp;&nbsp; Re:0.75<br>
Query: 10 &nbsp;&nbsp; Pr: 0.08 &nbsp;&nbsp; Re:0.16666666666666666<br>
#### Avg precision: 0.10200000000000001
#### Avg recall: 0.4346198830409357
----------------------------------------------------------------
#### Top 100 documents in rank list
Query: 1 &nbsp;&nbsp; Pr: 0.0 &nbsp;&nbsp; Re:0.0<br>
Query: 2 &nbsp;&nbsp; Pr: 0.09 &nbsp;&nbsp; Re:0.6<br>
Query: 3 &nbsp;&nbsp; Pr: 0.09 &nbsp;&nbsp; Re:0.6<br>
Query: 4 &nbsp;&nbsp; Pr: 0.06 &nbsp;&nbsp; Re:0.3333333333333333<br>
Query: 5 &nbsp;&nbsp; Pr: 0.13 &nbsp;&nbsp; Re:0.6842105263157895<br>
Query: 6 &nbsp;&nbsp; Pr: 0.09 &nbsp;&nbsp; Re:0.5<br>
Query: 7 &nbsp;&nbsp; Pr: 0.09 &nbsp;&nbsp; Re:1.0<br>
Query: 8 &nbsp;&nbsp; Pr: 0.03 &nbsp;&nbsp; Re:0.75<br>
Query: 9 &nbsp;&nbsp; Pr: 0.06 &nbsp;&nbsp; Re:0.75<br>
Query: 10 &nbsp;&nbsp; Pr: 0.04 &nbsp;&nbsp; Re:0.16666666666666666<br>
#### Avg precision: 0.068
#### Avg recall: 0.5384210526315789
----------------------------------------------------------------
#### Top 500 documents in rank list
Query: 1 &nbsp;&nbsp; Pr: 0.002 &nbsp;&nbsp; Re:1.0<br>
Query: 2 &nbsp;&nbsp; Pr: 0.03 &nbsp;&nbsp; Re:1.0<br>
Query: 3 &nbsp;&nbsp; Pr: 0.03 &nbsp;&nbsp; Re:1.0<br>
Query: 4 &nbsp;&nbsp; Pr: 0.032 &nbsp;&nbsp; Re:0.8888888888888888<br>
Query: 5 &nbsp;&nbsp; Pr: 0.038 &nbsp;&nbsp; Re:1.0<br>
Query: 6 &nbsp;&nbsp; Pr: 0.036 &nbsp;&nbsp; Re:1.0<br>
Query: 7 &nbsp;&nbsp; Pr: 0.018 &nbsp;&nbsp; Re:1.0<br>
Query: 8 &nbsp;&nbsp; Pr: 0.008 &nbsp;&nbsp; Re:1.0<br>
Query: 9 &nbsp;&nbsp; Pr: 0.016 &nbsp;&nbsp; Re:1.0<br>
Query: 10 &nbsp;&nbsp; Pr: 0.026 &nbsp;&nbsp; Re:0.5416666666666666<br>
#### Avg precision: 0.0236
#### Avg recall: 0.9430555555555555

## Detailed List of Functionalities Implemented
#### 1. tokenize_and_remove_punctuations:
The function accepts the text data as a parameter and does the following operations:
a. Removes the punctuations from the text data
b. Removes numbers from the data
c. Converts the data to lower case
d. Tokenizes the words from the text data
#### 2. get_stopwords: 
The function reads the common English stop words from the file stopwords.txt and then returns them as a list
#### 3. parse_data: 
This function takes in the raw data from the SGML files from the cranfieldDocs collection and extracts the data present between the <TITLE> and <TEXT> tags and returns the content
#### 4. remove_stop_words: 
The function takes in tokenized words and removes stop words from it and the words which are smaller than 2 characters in length
#### 5. read_data: 
The function takes the path of the crafieldDocs directory and reads the contents of the files present in the directory calls the parse_data function and returns a list of tuples of the following format – (document_no, text_data)
#### 6. calculate_tf: 
The function takes in a list of tokenized words and returns a dictionary with token as the key and the tf_score for that token as the value
#### 7. get_vocabulary: 
This function parses all the words from the document collection and returns a vocabulary of all the unique words from the collection
#### 8. stem_words: 
The function takes in a list of tokens and returns a list of stemmed words using NLTKs PorterStemmer
#### 9. preprocess_data:
The function takes a list of tuples generated by read_data and then performs the following functions:
a. tokenizes and removes puntuations
b. removes stop words
c. stems words
d. removes stop words after stemming and words shorter than 2 characters
#### 10. calculate_idf: 
The function takes in the entire data dictionary and calculates the idf score for each word in the vocabulary for the document collection and returns a dictionary with word as key and idf score as the value
#### 11. calculate_tfidf: 
The function takes in a the data dictionary of the parsed and preprocessed data and the dictionary for idf_scores and returns a dictionary containing key as the document id and value which is another dictionary with words of that document as keys and their tf-idf scores as values
#### 12. preprocecss_queries: 
This function is similar to that of preprocess data it performs the same operations on the queries
#### 13. calculate_tfidf_queries: 
This function takes the preprocessed queries and the idf_scores dictionary and then performs calculation for calculating the tfidf scores of the query terms and returns a similar dictionary as returned by calculate_tfidf function but for query terms
#### 14. generate_inverted_index: 
This function takes in the data dictionary of preprocessed data, builds and inverted index and then returns a dictionary with key as each unique word from the document vocabulary and value having a list of documents where the word in present
#### 15. get_relevance: 
This function takes in the path of the file containing the query ids and their relevant documents and then returns a dictionary with key as document ids and value as a list of documents relevant for it
#### 16. find_precision_recall: 
This function takes the retrieved document list and relevant document list of a query, calculates and returns the precision and recall
