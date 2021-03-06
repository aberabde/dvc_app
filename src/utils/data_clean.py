from tqdm import tqdm
import joblib
import numpy as np
import pandas as pd
import re
import scipy.sparse as sparse
import datetime
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from num2words import num2words
from nltk.corpus import wordnet
from nltk import tokenize
nltk.download('stopwords', quiet=True)
None
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('omw-1.4', quiet=True)
import json
from time import time
import datetime


###-0

def timer_decorator(func):
    """
    Affiche le temps d'exécution de l'objet fonction passé comme paramètre
    """
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        seconds_input = (t2-t1)
        conversion = datetime.timedelta(seconds=seconds_input)
        converted_time = str(conversion)
        print(f'Function {func.__name__!r} executed in {converted_time} \n')
        return result
    return wrap_func


#######-1


@timer_decorator
def convert_lower_case(source) -> pd.Series:
    print("starting convertig to lower case ...")
    return source.str.lower()

###### -2 
@timer_decorator
def remove_stop_words(source, excluded) -> pd.Series:
    print("Starting removing stopwords ...")
    def helper(line, excluded = excluded):
        return " ".join([word for word in line.split() if word not in excluded])
    return source.apply(helper)

###### -3
@timer_decorator
def keep_alphanumeric_and_dot(source) -> pd.Series:
    print("Starting filtering alphanumeric characters ...")
    return source.apply(lambda x: re.sub('[^A-Za-z0-9.]+', ' ', x))

###### - 4
@timer_decorator
def remove_single_characters(source) -> pd.Series:
    print("starting removing single characters ...")
    
    def helper(line):
        words = line.split()
        text = ""
        for w in words:
            if len(w) > 1 or w == '.':
                text = text + " " + w
        return text
    
    return source.map(helper)

##### - 5 
@timer_decorator
def remove_dot(source) -> pd.Series:
    print("starting removing dots ...")
    
    def helper(line):
        words = line.split()
        text = ""
        for w in words:
            if w != '.':
                text = text + " " + w
        return text
    
    return source.map(helper)

###### - 6 
@timer_decorator
def remove_stop_words(source, excluded) -> pd.Series:
    print("Starting removing stopwords ...")
    def helper(line, excluded = excluded):
        return " ".join([word for word in line.split() if word not in excluded])
    return source.apply(helper)

#### - 7 
@timer_decorator
def convert_numbers_to_text(source) -> pd.Series:
    print("starting converting numbers to text ...")
    
    def helper(line):
        words = line.split()
        text = ""
        for word in words:
            text += " "
            if word.isnumeric():
                text += num2words(word, to = 'cardinal')
            else:
                has_digits = any(n.isdigit() for n in word)
                if not has_digits:
                    text += word
                else:
                    
                    #extraire les chiffres et considérer le reste des caractères comme un bruit.
                    #pas le plus optimisé mais c'est acceptable dans notre cas
                    
                    for m in [num2words(n, to = 'cardinal') for n in re.findall(r'\d+', word)]:
                        text += " " + m 
        return text
    
    return source.apply(helper)

    ##### - 8



def lemmatize_helper(sentence): 
    
      #  mappage entre les tags NLTK et WordNetLemmatizer. WordNetLemmatizer ne suporte pas tous les tags NLTK
    
    pos_tagger = lambda nltk_tag: {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV 
    }.get(nltk_tag[0].upper(), None)
    
    # tokenisez la phrase et trouvez le tag POS pour chaque token
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    
    for word, tag in wordnet_tagged:
        if tag is None:
            # s'il n'y a pas un tag disponible, ajoutez le token tel quel
            lemmatized_sentence.append(word)
        else:       
            # utiliser le tag pour lemmatiser le token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))

    return " ".join(lemmatized_sentence)

@timer_decorator
def lemmatize(source) -> pd.Series:
    print("starting lemmatizing ...")
    return source.apply(lemmatize_helper)



def processed_data(source):


    # print("STARTING PREPROCESSING THE RAW TEXT ...\n")
    
    tqdm.pandas(desc='Processing Dataframe')

    result = keep_alphanumeric_and_dot(source)
    result = convert_lower_case(result)
    result = convert_numbers_to_text(result)
    result = remove_single_characters(result)
    result = lemmatize(result)
    result = remove_dot(result)
    result = remove_stop_words(result, stopwords.words('english'))
    
    # print("FINISHED PREPROCESSING THE RAW TEXT ...")
    return result


def get_df(path_to_data:str, 
            sep = ';', 
            column_names : list=['label','text'],
            encoding = 'utf8') -> pd.DataFrame:
    df = pd.read_csv(
        path_to_data,
        usecols= ["label","text"] ,
        delimiter = sep, 
        encoding =encoding,
        # header = None,
        # names=column_names "Unnamed: 0",'id',
    
    )
    # df.rename(columns={"Unnamed: 0":"id"}, inplace = True)
    return df

def save_matrix(df, text_matrix, out_path):
    # print("summary: >>>>>>>>>>>>>>>\n",df.describe())
    id_matrix = sparse.csr_matrix(df.index.values.astype(np.int64)).T
    label_matrix = sparse.csr_matrix(df.label.astype(np.int64)).T

    result = sparse.hstack([id_matrix, label_matrix, text_matrix], format="csr")
    print(result)
    joblib.dump(result, out_path) 

def save_json(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=4)



