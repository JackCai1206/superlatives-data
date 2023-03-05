import pandas as pd
import random

def get_random_percentage_list():
    r = [random.random() for _ in range(len(column_names))]
    r = [x / sum(r) for x in r]
    return r

column_names = ['ben', 'jerry', 'deez', 'aaron']
data = [get_random_percentage_list() for _ in range(100)]
questions = [f'cream {i}?' for i in range(100)]

fake_df = pd.DataFrame(data, index=questions, columns=column_names)
fake_prompts = ['huge', 'not huge', 'daisy']