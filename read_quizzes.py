import pandas as pd
import numpy as np

from scipy.special import softmax

people = ['jack', 'ben', 'will', 'daisy', 'anna', 'matt', 'michael', 'max', 'emily', 'hanna', 'jeremy', 'sabrina']
quizzes = ['gise_pt2.csv', 'gise_pt2.csv']

def get_normalized_counts(dataframe, people):
    '''
    Get the normalized counts for each person for each question
    param dataframe: dataframe of csv from google forms
    param people: list of all people in google form
    return: dataframe where columns are people, rows are questions, and entries are a person's percent of votes for a certain question
    '''

    questions = dataframe.columns[2:-1]
    counts = pd.DataFrame(0, index=questions, columns=people)
    for question in questions:
        for person in people:
            counts.loc[question, person] = np.exp(sum([voted_person == person for voted_person in dataframe[question]]))
    return counts.div(counts.sum(axis=1), axis=0)

def get_full_normalized_counts(people=people, quizzes=quizzes):
    '''
    Get the normalized counts for each person for each question for all quizzes
    param people: list of all people in google form
    param quizzes: list of all quizzes
    return: dataframe where columns are people, rows are questions, and entries are a person's percent of votes for a certain question
    '''
    return pd.concat([get_normalized_counts(pd.read_csv(quiz_name), people) for quiz_name in quizzes]).dropna(axis=0)

if __name__ == '__main__':
    # print(get_full_normalized_counts())
    # Check if there is any NaNs 
    print(get_full_normalized_counts().isnull().values.any())



