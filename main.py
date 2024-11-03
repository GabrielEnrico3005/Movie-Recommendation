import os
import pickle
import csv
import random
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.probability import FreqDist
from nltk.classify import NaiveBayesClassifier, accuracy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATASET_PATH = 'filtered_data.csv'
SEED = 42
WORD_DICTIONARY = []
labeled_dataset = []


def read_csv():

    global WORD_DICTIONARY, labeled_dataset
    dataset = []
    with open(DATASET_PATH, encoding='UTF-8') as file:
        reader = csv.reader(file)
        for row_data in reader:
            dataset.append(row_data)
    random.seed(SEED)
    random.shuffle(dataset)
    for row in dataset:
        sentence = row[1]
        # clean dataset
        sentence = re.sub('[^A-Za-z ]', '', sentence)
        # tokenize dataset
        word_list = word_tokenize(sentence)
        english_stopword = stopwords.words('english')
        word_list = [word for word in word_list if word not in english_stopword]
        # stemming
        porter_stemmer = PorterStemmer()
        word_list = [porter_stemmer.stem(word) for word in word_list]
        # Lemmatizing
        lemmatizer = WordNetLemmatizer()
        word_list = [lemmatizer.lemmatize(word)for word in word_list]
        
        WORD_DICTIONARY.extend(word_list)
        labeled_dataset.append((row[0], row[1], row[2]))

    fd = FreqDist(WORD_DICTIONARY)
    WORD_DICTIONARY = [word for word, count in fd.most_common(100)]

def classify():
    global labeled_dataset, WORD_DICTIONARY
    dataset = []
    for title, sentence, label in labeled_dataset:
      
        # clean dataset
        sentence = re.sub('[^A-Za-z ]', '', sentence)
        # tokenize dataset
        word_list = word_tokenize(sentence)
        english_stopword = stopwords.words('english')
        word_list = [word for word in word_list if word not in english_stopword]
        # stemming
        porter_stemmer = PorterStemmer()
        word_list = [porter_stemmer.stem(word) for word in word_list]
        # Lemmatizing
        lemmatizer = WordNetLemmatizer()
        word_list = [lemmatizer.lemmatize(word)for word in word_list]
        if label == 'POSITIVE':
            new_label = 1
        elif label == 'NEGATIVE':
            new_label = 0

        dict = {}
        for feature in WORD_DICTIONARY:
            key = feature
            value = feature in word_list
            dict[key] = value

        dataset.append((dict, new_label))
    # split training, testing
    training_ammount = int(len(dataset)*0.75)
    training_data = dataset[:training_ammount]
    testing_data = dataset[training_ammount:]

    classifier = NaiveBayesClassifier.train(training_data)
    print(f'MODEL ACCURACY: {accuracy(classifier, testing_data)*100}%')

    file = open('model.pickle', 'wb')
    pickle.dump(classifier, file)
    file.close()
        
def give_recommendation():
    global WORD_DICTIONARY, labeled_dataset
    text_review = 'i like action movie and comedy movie'
    vectorizer = TfidfVectorizer(WORD_DICTIONARY)
    all_matix = vectorizer.fit_transform([row[1] for row in labeled_dataset])
    user_review_matrix = vectorizer.transform([text_review])

    cosine_sim = cosine_similarity(all_matix, user_review_matrix)
    top_idx = cosine_sim.argsort(axis = 1)[0][-2:][::-1]

    top_movie = [(labeled_dataset[i][0]) for i in top_idx]
    print('TOP 2 MOVIE RECOMENDATION')
    for i in range(2):
        print(f'{i+1}. {top_movie[i]}')


def check_model():
    if os.path.isfile("model.pickle"):
        file = open('model.pickle', 'rb')
        classifier = pickle.load(file)
        file.close()
        print(' [>] LOAD MODEL COMPLETED!')
    else:
        print('[>] TRAIN MODEL . . .')

if __name__ == "__main__":
    read_csv()
    # print(WORD_DICTIONARY)
    # classify()
    give_recommendation()