
# Project Simulations for SI650-Information Retrieval: Recommender system based on political bias

from datetime import datetime
import pandas as pd
import numpy as np
import re
import nltk
import os 
import warnings
warnings.filterwarnings("ignore")

# Read in text file of dictionary sources and append to a list
null = None
dicts_from_file = []
with open('docs_new.txt', 'r', encoding = 'utf8') as documents:
    for line in documents:
        dicts_from_file.append(eval(line))

# Initialize political leaning 
political_leaning = 0

# Create initial file where political leanings will be stored
with open('leanings.txt', 'w', encoding = 'utf8') as leanings:
    leanings.write(str(political_leaning))

# Make into pandas dataframe
sources = pd.DataFrame(dicts_from_file)

# Keep only data with text
sources = sources[sources['text'] != ""]

# Extract source
import tldextract
sources = sources.reset_index()
url_name_list = []
for url in sources['url']:
    url_name = tldextract.extract(url)
    url_name_list.append(url_name.domain)
    
# Add index and merge with df
url_df = pd.DataFrame(url_name_list).reset_index()
url_df['idx'] = url_df.index

# Add index to sources
sources['idx'] = sources.index

# Merge
sources1 = pd.merge(sources, url_df, how = "inner", on = "idx")
sources1 = sources1.rename(columns = {0 : 'source' })

# Read in news rank files already created
news_rank = pd.read_csv("SI_650_news_sources.csv")

# Merge on source
sources1 = pd.merge(sources1, news_rank, how = "inner", on = "source")

# Rename
sources1 = sources1.rename(columns = {'template-top' : 'template_top',
                                      "political_leaning_x":"political_leaning",
                                      "score_x":"score"})

# Adjust template_top and section to lists and pull out first list to use for keywords, if applicable
sources1['template_top2'] = sources1['template_top'].str.split(',')
sources1['template_top2'] = sources1.template_top2.fillna("")
sources1['template_top2'] = np.where(sources1['template_top2'] != "", sources1['template_top2'][0][0], "")
sources1['section2'] = sources1['section'].str.split(',')
sources1['section2'] = sources1.section2.fillna("")
sources1['section2'] = np.where(sources1['section2'] != "", sources1['section'][0], "")

# Use first keyword as the main topic of the article
sources1['first_keyword'] = np.where(sources1.template_top2.str.len()>0, sources1.template_top2,
                                    np.where(sources1.section2.str.len() > 0, sources1.section2,
                                    np.where(sources1.keywords.str.len() > 0, sources1.keywords.str[0], "")))
sources1['first_keyword'] = sources1['first_keyword'].str.lower()

# Remove duplicates
sources1 = sources1.drop_duplicates(subset = ['title'])

# Keep unlabeled data separately which will be predicted on 
unlabeled = sources1.copy()
unlabeled = unlabeled[unlabeled['first_keyword'] == '']
unlabeled = unlabeled[unlabeled['title'] != 'Transcripts']

# Make a copy and subset the copy to work with only labeled data
labeled = sources1.copy()
labeled = labeled[labeled['first_keyword'] != '']

# Drop source with data quality issues
labeled = labeled[labeled['first_keyword'] != 'cigars']
labeled = labeled.drop(["index_x", "index_y"], axis = 1)

# Make into 5 broad categories- Business, Tech, Politics/political news, general/other news, sports, other
labeled['politics'] = labeled['first_keyword'].apply(lambda x: re.findall(r'(politics|trump|clinton|g-20|politics|state|election|george)', x))
labeled['business'] = labeled['first_keyword'].apply(lambda x: re.findall(r'(business|economy|national|finance)', x))
labeled['technology'] = labeled['first_keyword'].apply(lambda x: re.findall(r'(tech|technology|privacy|cybersecurity|apps)', x))
labeled['news'] = labeled['first_keyword'].apply(lambda x: re.findall(r'(news|new)', x))
labeled['entertainment'] = labeled['first_keyword'].apply(lambda x: re.findall(r'(entertainment)', x))

# Make into string
cats = ['politics', 'business', 'technology', 'news', 'entertainment']
for cat in cats:
    labeled[cat] = labeled[cat].apply(', '.join)
    
# Combine into 1 variable
labeled['category'] = np.where(labeled['politics'] != "", "politics",
                          np.where(labeled['business'] != "", "business",
                              np.where(labeled['news'] != "", "news",
                                   np.where(labeled['technology'] != "", "technology",
                                                     np.where(labeled['first_keyword'] == "entertainment", "entertainment","other")))))

# Clean title
# Removing stop words
stopwords = nltk.corpus.stopwords.words('english')

# Get rid of extras, split, and join for uncleaned paragraphs
def preprocess_para(x):
    x = re.sub('[^a-z\s]', '',x.lower())                   
    x = [w for w in x.split('\n\n') if w not in stopwords]       
    return x[0]                                   


# Get rid of extras, split, and join for text,title, and cleaned paragraphs
def preprocess(x):
    x = re.sub('[^a-z\s]', '',x.lower())                   
    x = [w for w in x.split() if w not in stopwords]       
    return ' '.join(x)                                     

# Call cleaning for labeled and unlabeled data
dfs = [labeled, unlabeled]

for df in dfs:
    df['para'] = df['text'].apply(preprocess_para)
    df['title_clean'] = df['title'].apply(preprocess)
    df['text_clean'] = df['text'].apply(preprocess)
    df['para_clean'] = df['para'].apply(preprocess)
    # Combine cleaned title and cleaned first paragraph
    df['title_1st_para'] = df['title_clean'] + ' ' + df['para_clean']

# Keep only obs with text
unlabeled = unlabeled[unlabeled['text_clean'] != '']

# Make labels numeric 
labeled.category = pd.Categorical(labeled.category)
labeled['label'] = labeled.category.cat.codes

# Randomly sample 75% into train / 25% into test
train = labeled.sample(frac = .75)

# Create test dataset
test = pd.merge(labeled, train[['idx']], on = 'idx', how = "outer",indicator=True)
test = test[test['_merge']=='left_only']

# Tokenize title and make into a wide dataframe with unigrams of titles
def tokenize(df):  
    # Make into a list of lists
    titles = df['title_clean'].tolist()
    tks = [nltk.word_tokenize(x) for x in titles]

    word_in_title = list()

    # Make into a wide 
    for title in tks:
        # Make a new dictionary and append at the end of each title
        word_cnt = dict()
        for word in title:
            if not word in word_cnt:
                word_cnt[word] = 1
            else:
                word_cnt[word] += 1
        # Append title specific dictionary to list of all dictionaries
        word_in_title.append(word_cnt)
    
    # Wide df with unigrams of title
    wd_type = pd.DataFrame(word_in_title)
    wd_type = wd_type.fillna(0)
    return wd_type

# Train-unigram words
train_frame = tokenize(train)
test_init = tokenize(test)

# Test has to match train (this is with no cleaning)
test_frame = pd.DataFrame(test_init).reindex_like(train_frame)
test_frame = test_frame.fillna(0)

# Train models using a number of different features 
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
# Use Counts and TFIDF, SGD Classifier with hinge loss, l2 penalty
text_class = Pipeline([
    ('vect', CountVectorizer()),
     ('tfidf', TfidfTransformer()),
     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                           alpha=1e-3, random_state=52,
                           max_iter=10, tol=None)),])

# Train this on the our title / 1st paragraph (x_train)
text_class.fit(train['title_1st_para'].values.astype(str), train['label'])
# Test on our subset of data left aside for testing that is labeled (x_test)
predicted = text_class.predict(test['title_1st_para'].values.astype(str))            

# Now tune parameters and use cross validation
from sklearn.model_selection import GridSearchCV
parameters = { 'vect__ngram_range': [(1, 1), (1, 2)],
               'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-1, 1e-3),}
grid_clf = GridSearchCV(text_class, parameters, cv=5, iid=False, n_jobs=-1)

# Do this for x_train
grid_clf = grid_clf.fit(train['title_1st_para'].values.astype(str), train['label']) 
# Predict on x_test
predicted = grid_clf.predict(test['title_1st_para'].values.astype(str))           

# Predict on unlabeled data
predicted_unlabeled = grid_clf.predict(unlabeled['title_1st_para'].values.astype(str))

# Add label
unlabeled['label'] = predicted_unlabeled
# Add corresponding category
unlabeled['category'] = np.where(unlabeled['label'] == 0, "business",
                                np.where(unlabeled['label'] == 1, "entertainment",
                                        np.where(unlabeled['label'] == 2, "news",
                                            np.where(unlabeled['label'] == 3, "other",
                                                    np.where(unlabeled['label'] == 4, "politics", "technology")))))

# Combine labeled and previously unlabeled datasets
all_sources = labeled.append(unlabeled)
# Keep only necessary variables
all_sources = all_sources[['label', 'text_clean', 'text', 'title', 'title_clean',
                           'title_1st_para', 'url', 'category', 'political_leaning',
                           'score', 'source', 'first_keyword']]
all_sources = all_sources.drop_duplicates(subset = ['title'])

# Separate out political leanings
pol_lean_list = list()
political_leanings = all_sources.political_leaning.unique()
for leaning in political_leanings:
    df = all_sources[all_sources['political_leaning'] == leaning]
    pol_lean_list.append(df)
    
### Create Simulated Users ###

# 1. Very conservative user (only very conservative sources)
all_cons = pol_lean_list[4].sample(n = 30, replace = False)
all_cons['user_rank'] = all_cons.score.mean()

# 2. Very liberal user (only very liberal sources)
all_lib_left = pol_lean_list[5].sample(n = 30, replace = True)

# 3. Fairly liberal left user (all left sources- no mixed)
all_lib_mid1 = pol_lean_list[5].sample(n = 10, replace = True)
all_lib_mid2 = pol_lean_list[7].sample(n = 10, replace = True)
all_lib_mid3 = pol_lean_list[3].sample(n = 10, replace = True)
all_lib_mid = all_lib_mid1.append(all_lib_mid2).append(all_lib_mid3)

# 4. Fairly conservative right user (Most cons right and mixed right)
all_cons_mid1 = pol_lean_list[4].sample(n = 10, replace = True)
all_cons_mid2 = pol_lean_list[2].sample(n = 20, replace = True)
all_cons_mid = all_cons_mid1.append(all_cons_mid2)

# 5. Only reads middle ground
mixed_mid = pol_lean_list[6].sample(n = 30, replace = True)

# 6. Left-mid mix 
left_cent1 = pol_lean_list[7].sample(n = 10, replace = True)
left_cent2 = pol_lean_list[3].sample(n = 10, replace = True)
left_cent3 = pol_lean_list[6].sample(n = 10, replace = True)
left_cent = left_cent1.append(left_cent2).append(left_cent3)

# 7. Right-mid mix
right_cent1 = pol_lean_list[2].sample(n = 15, replace = True)
right_cent2 = pol_lean_list[6].sample(n = 15, replace = True)
right_cent = right_cent1.append(right_cent2)

# 8. Left and right mix 
left_right1 = pol_lean_list[2].sample(n = 15, replace = True)
left_right2 = pol_lean_list[1].sample(n = 15, replace = True)
left_right = left_right1.append(left_right2)

# 9. Random (this will oversample on left)
rand= all_sources.sample(n = 30, replace = True)

# 10. Reads everything equally (more mixed middle)
equal_read = pol_lean_list[6].sample(n = 9, replace = True)

for i in range(8):
    if i != 6:
        df = pol_lean_list[i].sample(n = 3, replace = True)
        equal_read = equal_read.append(df)

# Create a dataframe with all users and their group average
# As the initial political leanings score
df_list = [all_lib_left, all_lib_mid, all_cons_mid, mixed_mid,
           left_cent, right_cent, left_right, rand, equal_read]
for df in df_list:
    df['user_rank'] = df.score.mean()
    all_cons = all_cons.append(df)

# Rename
users = all_cons

if os.path.exists('simulated_users_art.csv'):
    users = pd.read_csv('simulated_users_art.csv')
else:
    users.to_csv('simulated_users_art.csv')

### Code below by project partner Teerth Patel ###

# Tokenize cleaned words
word_list = list()
for txt in all_sources['text_clean']:
    tokens = nltk.word_tokenize(txt)
    word_list.append(tokens)

category = {
    'business':0,
    'entertainment':1,
    'news':2,
    'other':3,
    'politics':4,
    'technology':5
}

category_input = input("Which of the following categories would you like to search? \n Business, Entertainment, News, Politics, Other \n")
category_input = category_input.lower()

query = input("What in this category would you like to search? ")

all_sources['relevant'] = (all_sources['label'] == int(category[category_input]))

import gensim.summarization.bm25 as bm25
import warnings

warnings.filterwarnings("ignore")
results = bm25.BM25(word_list)

average_idf = sum(map(lambda k: float(results.idf[k]), results.idf.keys())) / len(results.idf.keys())
scores = results.get_scores(category_input + ' ' + query, average_idf)
all_sources['bm25'] = scores

import random

simulated_leaning = users['political_leaning'].drop_duplicates()
sel_user = simulated_leaning.sample(1).iloc[0]
ind_user = users[users['political_leaning'] == sel_user]

user_political_leaning = ind_user['score'].mean()
political_weight = 1/abs(user_political_leaning - all_sources['score'])
all_sources['bm25weight'] = all_sources['bm25'].multiply(political_weight)
output = all_sources.sort_values(by = ['bm25weight'], ascending = False)
indexed_output = pd.DataFrame({'Index': range(1, 21), 
                               'Title':output.title[:20], 
                               'Text':output['text'][:20], 
                               'Relevant':output['relevant'][:20], 
                               'Source':output['source'][:20],
                               'Keyword':output['first_keyword'][:20]})

pd.options.display.max_colwidth = 100

print('This simulated user is: ', sel_user)

print(indexed_output[['Index', 'Title']])

q_input = input('Would you like to read one of the articles? (Y/N)')

if q_input == 'Y':
    art_input = input("Which article would you like to read? \n")
    text = indexed_output.loc[indexed_output['Index'] == int(art_input)]['Text']
    title = indexed_output.loc[indexed_output['Index'] == int(art_input)]['Title']
    print('\n', title.values[0], ' \n')
    print(text.values[0])
else:
    print('Sorry you didn\'t find anything you wanted to read!')

idx = indexed_output.index[indexed_output['Index'] == int(art_input)]

users = users.append(all_sources.loc[idx])

users.to_csv('simulated_users_art.csv')
