

## created by: Nanchun (Aslan) Shi （nanchuns@usc.edu）
## created on: May 3, 2020

# In[1]:


import pandas as pd
import numpy as np
import re
import pickle
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from gensim.models.phrases import Phraser, Phrases
from keras.preprocessing.sequence import pad_sequences

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

tokenizer = load_obj('style_tokenizer')
embedding_bigram = load_obj('style_embedding_bigram')
stopwords = set(stopwords.words('english'))
punc = string.punctuation.replace('-','')
lemmatizer = WordNetLemmatizer()


# In[2]:


class embedding_preprocessing():
    
    def __init__(self):
        
        self.tk = tokenizer
        
    def preprocess(self, df, maxlen = 169):
        
        """
        
        - df must be a dataframe where index is product id + color id, and two columns are description + detials
        
        """
        
        vocabs = self.tk.word_index.keys()
        
        df1 = self.treat_na(df)
        df2 = self.remove_punc_sw(df1)
        df3 = self.remove_numbers(df2)
        df4 = self.lemma_pos(df3)
        df5 = self.bigram(df4)
        df6 = self.combine_bigrams(df5)
        
        new_docs = []
        
        for word_list in df6:
            
            if len(word_list) == 2 and word_list[0].lower() == 'noinfo' and word_list[1].lower() == 'noinfo':
                new_docs.append(list(np.zeros(maxlen)))
            
            else:
                new_word_list = []
                for word in word_list:
                    if word not in vocabs:
                        word = 'UNKNOWN_TOKEN'
                    new_word_list.append(word)
                    
                sequence = " ".join(new_word_list)
                vectors = self.tk.texts_to_sequences([sequence])
                padded_vectors = pad_sequences(vectors, maxlen=maxlen, padding='post', truncating='post')
                    
                new_docs.append(list(padded_vectors[0]))
            
        return new_docs
    
    def treat_na(self, df):
        
        """
        - input must be a dataframe where index is product id + color id, and two columns are description + detials
        - output a pd.series where null values are filled and two fields are concatenated
        """
        
        df.fillna('NOINFO',inplace=True)
        
        df['combined_text'] = df[['description','details']].apply(lambda x: x[0]+' '+x[1],axis=1)
        
        return df.combined_text
    
    def remove_punc_sw(self, docs):
        
        """
        - input should be a iterable of strings
        - output a pd.series of strings with punctuation and stopwords removed and in lower cases
        """
        
        new_docs = []
        
        for text in docs:
    
            for p in punc:
                text = text.replace(p,' ')
            text = text.replace('-', '')
            text = text.replace("’", ' ')
            text = text.lower()
            tokens = word_tokenize(text)
            filtered_tokens = list(filter(lambda token: token not in stopwords, tokens))
            
            new_text =  " ".join(filtered_tokens)
            new_docs.append(new_text)
            
        return pd.Series(new_docs)
    
    def remove_numbers(self, docs):
        
        """
        - input should be a iterable of strings
        - output a pd.series of strings
        """
        
        new_docs = []
        
        for text in docs:
    
            text = re.sub(r'\b\d+\b',' ',text)
            text = re.sub(r'\s+',' ',text)
            
            new_docs.append(text)

        return pd.Series(new_docs)
    
    def lemma_pos(self, docs):
        
        """
        - input should be an iterable of strings, each string is a document
        - output a pd.series of lists of lemmatized tokens, each list is a document
        
        """
        
        new_docs = []
        
        for text in docs:
            
            words = []

            for word, tag in pos_tag(text.split()):
                if tag.startswith("N"):
                    word = lemmatizer.lemmatize(word, wordnet.NOUN)
                elif tag.startswith('V'):
                    word = lemmatizer.lemmatize(word, wordnet.VERB)
                elif tag.startswith('J'):
                    word = lemmatizer.lemmatize(word, wordnet.ADJ)
                elif tag.startswith('R'):
                    word = lemmatizer.lemmatize(word, wordnet.ADV)
                else:
                    word = word
                    
                words.append(word)
                
            new_docs.append(words)
            
        return pd.Series(new_docs)
    
    def bigram(self, docs):
        
        """
        
        - docs should be an iterable of lists of tokens
        - output a pd.series of lists of strings with recognized bi-grams combined
        
        """
        new_docs = list(embedding_bigram[docs])
            
        return pd.Series(new_docs)
    
    def combine_bigrams(self, docs):
    
        new_docs = []
        
        for doc in docs:
            new_doc = []
            for w in doc:
                w = w.replace('_','')
                new_doc.append(w)
            new_docs.append(new_doc)

        return new_docs


# In[3]:


vectorizer = load_obj('style_tfidf_vectorizer')
tfidf_bigram = load_obj('style_tfidf_bigram')
punc2 = string.punctuation.replace('-','')


# In[5]:


class tfidf_preprocessing():
    
    def __init__(self):
        
        """
        -'brand','p_full_name','brand_category','brand_canonical_url'
        """
        
        self.vt = vectorizer
        
    def preprocess(self, df):
        
        df1 = self.na_punc_sw_combine(df)
        df2 = self.remove_numbers(df1)
        df3 = self.lemma(df2)
        df4 = self.bigram(df3)
        df5 = self.combine_bigrams(df4)
        
        joined_docs = []
        for doc in df5:
            joined_docs.append(" ".join(doc))
        
        X = self.vt.transform(joined_docs)
        terms = self.vt.get_feature_names()
        
        return pd.DataFrame(X.toarray(), columns = terms)
        
    
    def na_punc_sw_combine(self, df):
        
        df.fillna('NOINFO',inplace=True)
    
        cols = [df.brand, df.p_full_name, df.brand_category, df.brand_canonical_url]
        cleaned_cols = []

        for col in cols:
            new_col = []
            for text in col:
                text = text.lower()
                text = text.replace('\n',' ')
                for p in punc2:
                    text = text.replace(p,' ')
                text = text.replace('-', '')
                text = text.replace("’", ' ')
                new_col.append(text)
            cleaned_cols.append(new_col)
            
        new_docs = []
        for i in range(len(cleaned_cols[0])):
            new_docs.append(" ".join(list(map(lambda x: x[i], cleaned_cols))))

        return pd.Series(new_docs)
    
    def remove_numbers(self, docs):
        
        new_docs = []
        
        for text in docs:
            
            text = re.sub(r'(\b\d+\b)','', text)
            text = re.sub(r'(\swww\s|\shttps*\s)','', text)
            new_docs.append(text)
        
        return pd.Series(new_docs)
    
    def lemma(self, docs):
        
        """
        - input an iterable of token strings
        - return an pd.series of lists of tokens
        """
        
        new_docs = []
        for text in docs:
            new_doc = []
            for token in word_tokenize(text):
                token = lemmatizer.lemmatize(token)
                new_doc.append(token)
            new_docs.append(new_doc)
            
        return pd.Series(new_docs)
    
    def bigram(self, docs):
        
        """
        
        - docs should be an iterable of lists of tokens
        - output a pd.series of lists of strings with recognized bi-grams combined
        
        """
        new_docs = list(tfidf_bigram[docs])
            
        return pd.Series(new_docs)
    
    def combine_bigrams(self, docs):
    
        new_docs = []
        
        for doc in docs:
            new_doc = []
            for w in doc:
                w = w.replace('_','')
                new_doc.append(w)
            new_docs.append(new_doc)

        return new_docs

