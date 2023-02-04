import re
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from cleantext import clean
from textblob import TextBlob

text = """I would appreciate any comments please. We have lived in our bungalow for only a couple of years and my husband's health has deteriorated so much. After suffering a brain haemorrhage last September his cognitive decline has been rapid."""

#seperate comments into single words
print("Original Text:")
words = WordPunctTokenizer().tokenize(text)
print(words)

#remove non words (emojis, symbols, numbers)
#words = clean(words, no_emoji=True)
#print(words)
#remove stop words
#will need dictionary of stop words
stop_words = set(stopwords.words('english'))
#print(stop_words)
#add additional stop words if necessary
#stop_words.extend("very")

#remove stop words from text
words = [x for x in words if x not in stop_words ]
print("Text with stop words removed")
print(words)

#keep count of amount of words
word_count = len(words)
print("Word count: ", word_count)

def word_polarity(test_subset):
    pos_word_list=[]
    neu_word_list=[]
    neg_word_list=[]

    for word in test_subset:               
        testimonial = TextBlob(word)
        if testimonial.sentiment.polarity > 0:
            pos_word_list.append(word)
        elif testimonial.sentiment.polarity < 0:
            neg_word_list.append(word)
        else:
            neu_word_list.append(word)

    print('Positive :',len(pos_word_list))        
    print('Neutral :',len(neu_word_list))    
    print('Negative :',len(neg_word_list))      

word_polarity(words)
