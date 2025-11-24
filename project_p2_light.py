# CS421: Natural Language Processing
# University of Illinois Chicago
# Fall 2025
# Project Part 2
#
# Do not rename/delete any functions or global variables provided in this template and write your solution
# in the specified sections. Use the main function to test your code when running it from a terminal.
# Avoid writing that code in the global scope; however, you should write additional functions/classes
# as needed in the global scope. These templates may also contain important information and/or examples
# in comments so please read them carefully. If you want to use external packages not specified in the
# assignment then you need prior approval from course staff.
# =========================================================================================================

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import string
import re
import csv
import nltk
from time import localtime, strftime
#from nltk.parse.corenlp import CoreNLPDependencyParser
import spacy
nlp = spacy.load("en_core_web_sm")
#-----------------------------------FILE WRITER-------------------------------------------------------

import nltk
from nltk.tokenize import word_tokenize
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")
f = open("{0}.txt".format(strftime("%Y-%m-%d_%H-%M-%S", localtime())), "w")

#-----------------------------------CODE FROM PART 1--------------------------------------------------

# Before running code that makes use of GloVe, you will need to download the provided glove.pkl file
# which contains the pre-trained GloVe representations from Blackboard
#
# If you store the downloaded .pkl file in the same directory as this Python
# file, leave the global EMBEDDING_FILE variable below as is.  If you store the
# file elsewhere, you will need to update the file path accordingly.
EMBEDDING_FILE = "glove.pkl"


# Function: load_glove
# filepath: path of glove.pkl
# Returns: A dictionary containing words as keys and pre-trained GloVe representations as numpy arrays of shape (300,)
def load_glove(filepath):
    with open(filepath, 'rb') as fin:
        return pkl.load(fin)


# Function: load_as_list(fname)
# fname: A string indicating a filename
# Returns: Two lists: one a list of document strings, and the other a list of integers
#
# This helper function reads in the specified, specially-formatted CSV file
# and returns a list of documents (documents) and a list of binary values (label).
def load_as_list(fname):
    df = pd.read_csv(fname)
    documents = df['review'].values.tolist()
    labels = df['label'].values.tolist()
    return documents, labels


# Function: extract_user_info(user_input)
# user_input: A string of arbitrary length
# Returns: name as string
#
# This helper function extracts a user's name from the text.  It's an imperfect function---feel free to customize it!
def extract_user_info(user_input):
    name = ""
    name_match = re.search(r"(^|\s)([A-Z][A-Za-z-&'\.]*(\s|$)){2,4}", user_input)
    if name_match is not None:
        name = name_match.group(0).strip()
    return name


# Function to convert a given string into a list of tokens
# Args:
#   inp_str: input string
# Returns: token list, dtype: list of strings
def get_tokens(inp_str):
    # Initialize NLTK tokenizer
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("NLTK tokenizer not found, downloading...")
        nltk.download('punkt')
    return nltk.tokenize.word_tokenize(inp_str)


# Function: vectorize_train, see project statement for more details
# training_documents: A list of strings
# Returns: An initialized TfidfVectorizer model, and a document-term matrix, dtype: scipy.sparse.csr.csr_matrix
def vectorize_train(training_documents):
    # Initialize the TfidfVectorizer model and document-term matrix
    vectorizer = TfidfVectorizer()
    tfidf_train = None
    # [YOUR CODE HERE FROM PROJECT PART 1]
    tfidf_train = vectorizer.fit_transform(training_documents)

    return vectorizer, tfidf_train


# Function: glove(glove_reps, token)
# glove_reps: The pretrained GloVe representations as dictionary
# token: A string containing a single token
# Returns: The GloVe embedding for that token, as a numpy array of size (300,)
#
# This function provides access to 300-dimensional GloVe representations
# pretrained on Dolma data.  If the specified token does not exist in the
# pretrained model, it should return a zero vector; otherwise, it returns the
# corresponding word vector from the GloVe dictionary.
def glove(glove_reps, token):
    word_vector = np.zeros(300,)

    # [YOUR CODE HERE FROM PROJECT PART 1]
    if token in glove_reps:
        word_vector = glove_reps[token]
    else:
        new_token = token.lower()
        if new_token in glove_reps:
            word_vector = glove_reps[new_token]
        
    return word_vector


# Function: string2vec(glove_reps, user_input)
# glove_reps: The pretrained GloVe representations
# user_input: A string of arbitrary length
# Returns: A 300-dimensional averaged GloVe embedding for that string
#
# This function preprocesses the input string, tokenizes it using get_tokens, extracts a word embedding for
# each token in the string, and averages across those embeddings to produce a single, averaged embedding for the
# entire input.
def string2vec(glove_reps, user_input):
    embedding = np.zeros(300,)

    # [YOUR CODE HERE FROM PROJECT PART 1]
    tokens = get_tokens(user_input)
    if len(tokens) == 0:
        return embedding

    vector_sum = np.zeros(300,)
    count = 0
    for tok in tokens:
        vec = glove(glove_reps, tok)
        vector_sum += vec
        count += 1

    if count > 0:
        embedding = vector_sum / count

    return embedding


# Function: instantiate_models()
# This function does not take any input
# Returns: Four instantiated machine learning models
#
# This function instantiates the four imported machine learning models, and
# returns them for later downstream use.  You do not need to train the models
# in this function.
def instantiate_models():
    nb = None
    logistic = None
    svm = None
    mlp = None

    # [YOUR CODE HERE FROM PROJECT PART 1]
    nb = GaussianNB()
    logistic = LogisticRegression(max_iter=1000)
    svm = LinearSVC()
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)

    return nb, logistic, svm, mlp


# Function: train_model_tfidf(model, glove_reps, training_documents, training_labels)
# model: An instantiated machine learning model
# tfidf_train: A document-term matrix built from the training data
# training_labels: A list of integers (all 0 or 1)
# Returns: A trained version of the input model
#
# This function trains an input machine learning model using averaged GloVe
# embeddings for the training documents.
def train_model_tfidf(model, tfidf_train, training_labels):
    # [YOUR CODE HERE FROM PROJECT PART 1]
    
    if isinstance(model, GaussianNB) or isinstance(model, MLPClassifier):
        X = tfidf_train.toarray()
    else:
        X = tfidf_train

    model.fit(X, training_labels)

    return model


# Function: train_model_glove(model, glove_reps, training_documents, training_labels)
# model: An instantiated machine learning model
# glove_reps: Pretrained GloVe representations
# training_documents: A list of training documents
# training_labels: A list of integers (all 0 or 1)
# Returns: A trained version of the input model
#
# This function trains an input machine learning model using averaged GloVe
# embeddings for the training documents.
def train_model_glove(model, glove_reps, training_documents, training_labels):
    # [YOUR CODE HERE FROM PROJECT PART 1]
    X = []
    for doc in training_documents:
        X.append(string2vec(glove_reps, doc))
    X = np.vstack(X)

    model.fit(X, training_labels)

    return model


# Function: test_model_tfidf(model, glove_reps, training_documents, training_labels)
# model: An instantiated machine learning model
# vectorizer: An initialized TfidfVectorizer model
# test_data: A list of test documents
# test_labels: A list of integers (all 0 or 1)
# Returns: Precision, recall, F1, and accuracy values for the test data
#
# This function tests an input machine learning model by extracting features
# for each preprocessed test document and then predicting an output label for
# that document.  It compares the predicted and actual test labels and returns
# precision, recall, f1, and accuracy scores.
def test_model_tfidf(model, vectorizer, test_documents, test_labels):
    precision = None
    recall = None
    f1 = None
    accuracy = None

    # [YOUR CODE HERE FROM PROJECT PART 1]
    tfidf_test = vectorizer.transform(test_documents)

    if isinstance(model, GaussianNB) or isinstance(model, MLPClassifier):
        Xtest = tfidf_test.toarray()
    else:
        Xtest = tfidf_test

    preds = model.predict(Xtest)

    precision = precision_score(test_labels, preds)
    recall = recall_score(test_labels, preds)
    f1 = f1_score(test_labels, preds)
    accuracy = accuracy_score(test_labels, preds)

    return precision, recall, f1, accuracy


# Function: test_model_glove(model, glove_reps, test_documents, test_labels)
# model: An instantiated machine learning model
# glove_reps: Pretrained GloVe representations
# test_documents: A list of test documents
# test_labels: A list of integers (all 0 or 1)
# Returns: Precision, recall, F1, and accuracy values for the test data
#
# This function tests an input machine learning model by extracting features
# for each preprocessed test document and then predicting an output label for
# that document.  It compares the predicted and actual test labels and returns
# precision, recall, f1, and accuracy scores.
def test_model_glove(model, glove_reps, test_documents, test_labels):
    precision = None
    recall = None
    f1 = None
    accuracy = None

    # [YOUR CODE HERE FROM PROJECT PART 1]
    Xtest = []
    for doc in test_documents:
        Xtest.append(string2vec(glove_reps, doc))
    Xtest = np.vstack(Xtest)

    preds = model.predict(Xtest)

    precision = precision_score(test_labels, preds)
    recall = recall_score(test_labels, preds)
    f1 = f1_score(test_labels, preds)
    accuracy = accuracy_score(test_labels, preds)

    return precision, recall, f1, accuracy


#-----------------------------------NEW CODE--------------------------------------------------


# Function: get_dependency_parse(input)
# This function accepts a raw string input and returns a CoNLL-formatted output
# string with each line indicating a word, its POS tag, the index of its head
# word, and its relation to the head word.
# Parameters:
# input - A string containing a single text input (e.g., a sentence).
# Returns:
# output - A string containing one row per word, with each row containing the
#          word, its POS tag, the index of its head word, and its relation to
#          the head word.
def get_dependency_parse(input: str):
    """
    Return a CoNLL-style dependency parse:
    one line per token:
        WORD \t POS \t HEAD_INDEX \t RELATION
    HEAD_INDEX = 0 means ROOT.
    """
    doc = nlp(input)
    lines = []

    # spaCy tokens are 0-based; we want 1-based like CoreNLP / our code
    for i, token in enumerate(doc):
        idx = i + 1  # this token's index

        # head index: if the head is itself (root), use 0
        if token.head == token:
            head_idx = 0
        else:
            head_idx = token.head.i + 1

        word = token.text
        pos = token.tag_          # detailed POS (similar to CoreNLP's tag)
        rel = token.dep_          # dependency relation

        lines.append(f"{word}\t{pos}\t{head_idx}\t{rel}")

    return "\n".join(lines)



# Function: get_dep_categories(parsed_input)
# parsed_input: A CONLL-formatted string.
# Returns: Five integers, corresponding to the number of nominal subjects (nsubj),
#          direct objects (obj), indirect objects (iobj), nominal modifiers (nmod),
#          and adjectival modifiers (amod) in the input, respectively.
#
# This function counts the number of grammatical relations belonging to each of five
# universal dependency relation categories specified for the provided input.
def get_dep_categories(parsed_input):
    num_nsubj = 0
    num_obj = 0
    num_iobj = 0
    num_nmod = 0
    num_amod = 0

    # Write your code here:
    for i in parsed_input.splitlines():
        i = i.strip()
        if not i:
            continue
        parts = i.split()
        if len(parts) < 4:
            continue
        rel = parts[-1]

        if rel.startswith("nsubj"):
            num_nsubj += 1

        if rel.startswith("iobj"):
            num_iobj += 1

        if (rel == "obj" or rel == "dobj" or rel.startswith("obj:") or rel.startswith("dobj:")):
            num_obj += 1

        if rel.startswith("nmod"):
            num_nmod += 1

        if rel.startswith("amod"):
            num_amod += 1

    return num_nsubj, num_obj, num_iobj, num_nmod, num_amod


# Function: custom_feature(user_input)
# user_input: A string of arbitrary length
# Returns: An output specific to the feature type implemented.
#
# This function implements a custom stylistic feature extractor.
def custom_feature(user_input):
    # Write your code here:
    # Tokenize and normalize
    tokens = get_tokens(user_input)
    new_token = [t.lower() for t in tokens]

    num_tokens = len(tokens)
    unique_tokens = len(set(new_token)) if num_tokens > 0 else 0
    lexical_diversity = round(unique_tokens / num_tokens, 3) if num_tokens > 0 else 0.0
    
    first_person = {"i", "me", "my", "mine", "we", "us", "our", "ours"}
    first_person_count = sum(1 for t in new_token if t in first_person)

    num_questions = user_input.count("?")
    num_exclaims = user_input.count("!")
    all_caps_words = sum(
        1 for t in tokens
        if t.isalpha() and t.isupper() and len(t) > 1
    )

    informal_markers = {
        "lol", "lmao", "idk", "gonna", "wanna", "kinda",
        "sorta", "btw", "u", "ya", "yeah", "nope", "nah"
    }
    num_informal = sum(1 for t in new_token if t in informal_markers)

    summary_parts = []
    summary_parts.append(
        f"Tokens: {num_tokens}, Lexical diversity: {lexical_diversity}"
    )
    summary_parts.append(
        f"First-person pronouns: {first_person_count}"
    )
    summary_parts.append(
        f"Questions: {num_questions}, Exclamations: {num_exclaims}"
    )
    summary_parts.append(
        f"ALL-CAPS words: {all_caps_words}, Informal/slang markers: {num_informal}"
    )

    interpretation = []

    if first_person_count > 0:
        interpretation.append(
            "You focus on yourself, so the message feels personal/self-reflective."
        )
    else:
        interpretation.append(
            "You avoid first-person pronouns, so the message feels less personal."
        )

    if num_questions > 0:
        interpretation.append(
            "You ask questions, which suggests curiosity or uncertainty."
        )

    if num_exclaims > 0 or all_caps_words > 0:
        interpretation.append(
            "There are signs of emotional intensity (exclamations / ALL-CAPS)."
        )

    if num_informal > 0:
        interpretation.append(
            "You use informal/slang expressions, which makes the tone casual."
        )
    else:
        interpretation.append(
            "You don't really use slang, which makes the tone more formal/neutral."
        )

    if num_tokens >= 10:
        if lexical_diversity < 0.4:
            interpretation.append(
                "You repeat words quite a bit, giving a more repetitive style."
            )
        elif lexical_diversity > 0.7:
            interpretation.append(
                "You use a wide range of vocabulary, which feels more varied."
            )

    if not interpretation:
        interpretation.append(
            "Overall, the style is very brief and neutral."
        )

    summary_parts.append("Interpretation: " + " ".join(interpretation))

    return " | ".join(summary_parts) #"CUSTOM FEATURE #1"


# Function: welcome_state()
# This function does not take any input
# Returns: A string indicating the next state
#
# This function implements the chatbot's welcome states.  Feel free to customize
# the welcome message!  In this state, the chatbot greets the user.
def welcome_state():
    # Display a welcome message to the user
    user_input = print("Welcome to the CS 421 chatbot!  ")
    f.write("CHATBOT:\nWelcome to the CS 421 chatbot!  ")

    return "get_user_info"


# Function: get_info_state()
# This function does not take any input
# Returns: A string indicating the next state and a string indicating the
#          user's name
#
# This function implements a state that requests the user's name and then processes
# the user's response to extract that information.  Feel free to customize this!
def get_info_state():
    # Request the user's name, and accept a user response of
    # arbitrary length.  Feel free to customize this!
    user_input = input("What is your name?\n")
    f.write("What is your name?\n")
    f.write("\nUSER:\n{0}\n".format(user_input))

    # Extract the user's name
    name = extract_user_info(user_input)

    return "sentiment_analysis", name


# Function: sentiment_analysis_state(name, model, vectorizer, glove_reps)
# name: A string indicating the user's name
# model: The trained classification model used for predicting sentiment
# vectorizer: OPTIONAL; The trained vectorizer, if using TFIDF (leave empty otherwise)
# glove_reps: OPTIONAL; The pretrained GloVe model, if using GloVe (leave empty otherwise)
# Returns: A string indicating the next state
#
# This function implements a state that asks the user what they want to talk about,
# and then processes their response to predict their current sentiment.  Feel free
# to customize this!
def sentiment_analysis_state(name, model, vectorizer=None, glove_reps=None):
    # Check the user's sentiment
    user_input = input("Thanks {0}!  What do you want to talk about today?\n".format(name))
    f.write("\nCHATBOT:\nThanks {0}!  What do you want to talk about today?\n".format(name))
    f.write("\nUSER:\n{0}\n".format(user_input))

    # Predict the user's sentiment
    # test = vectorizer.transform([user_input])  # Use if you selected a TFIDF model
    test = string2vec(glove_reps, user_input)  # Use if you selected a GloVe model

    label = None
    label = model.predict(test.reshape(1, -1))

    if label == 0:
        print("Hmm, it seems like you're feeling a bit down.")
        f.write("\nCHATBOT:\nHmm, it seems like you're feeling a bit down.\n")
    elif label == 1:
        print("It sounds like you're in a positive mood!")
        f.write("\nCHATBOT:\nIt sounds like you're in a positive mood!\n")
    else:
        print("Hmm, that's weird.  My classifier predicted a value of: {0}".format(label))
        f.write("\nCHATBOT:\nHmm, that's weird.  My classifier predicted a value of: {0}\n".format(label))

    return "stylistic_analysis"


# Function: stylistic_analysis_state()
# This function does not take any input
# Returns: A string indicating the next state
#
# This function implements a state that asks the user what's on their mind, and
# then analyzes their response.  Feel free to customize this!
def stylistic_analysis_state():
    user_input = input("I'd also like to do a quick stylistic analysis. What's on your mind today?\n")
    f.write("\nCHATBOT:\nI'd also like to do a quick stylistic analysis. What's on your mind today?\n")
    f.write("\nUSER:\n{0}\n".format(user_input))
    dep_parse = get_dependency_parse(user_input)
    num_nsubj, num_obj, num_iobj, num_nmod, num_amod = get_dep_categories(dep_parse)
    custom = custom_feature(user_input)

    # Generate a stylistic analysis of the user's input
    print("Thanks!  Here's what I discovered about your writing style.")
    # print("Dependencies:\n{0}".format(dep_parse)) # Uncomment to view the full dependency parse.
    print("# Nominal Subjects: {0}\n# Direct Objects: {1}\n# Indirect Objects: {2}"
          "\n# Nominal Modifiers: {3}\n# Adjectival Modifiers: {4}".format(num_nsubj, num_obj,
                                                                           num_iobj, num_nmod, num_amod))
    print("Custom Feature: {0}".format(custom))

    f.write("\nCHATBOT:\nThanks!  Here's what I discovered about your writing style.\n")
    # f.write("Dependencies:\n{0}\n".format(dep_parse)) # Uncomment to view the full dependency parse.
    f.write("# Nominal Subjects: {0}\n# Direct Objects: {1}\n# Indirect Objects: {2}"
          "\n# Nominal Modifiers: {3}\n# Adjectival Modifiers: {4}\n".format(num_nsubj, num_obj,
                                                                           num_iobj, num_nmod, num_amod))
    f.write("Custom Feature: {0}\n".format(custom))

    return "check_next_action"


# Function: check_next_state()
# This function does not take any input
# Returns: A string indicating the next state
#
# This function implements a state that checks to see what the user would like
# to do next.  The user can indicate that they would like to quit, redo the sentiment
# analysis, or redo the stylistic analysis.  Feel free to customize this!
def check_next_state():
    next_state = ""

    user_input = input("What would you like to do next?  You can quit, redo the "
                       "sentiment analysis, or redo the stylistic analysis.\n")
    f.write("\nCHATBOT:\nWhat would you like to do next?  You can quit, redo the "
                       "sentiment analysis, or redo the stylistic analysis.\n")
    f.write("\nUSER:\n{0}\n".format(user_input))

    match = False # Only becomes true once a match is found
    while not match:
        quit_match = re.search(r"\bquit\b", user_input) # This is *not* a comprehensive regex ...feel free to update!
        sentiment_match = re.search(r"\bsentiment\b", user_input)
        analysis_match = re.search(r"\bstyl", user_input)

        if quit_match is not None:
            next_state = "quit"
            match = True
        elif sentiment_match is not None:
            next_state = "sentiment_analysis"
            match = True
        elif analysis_match is not None:
            next_state = "stylistic_analysis"
            match = True
        else:
            user_input = input("Sorry, I didn't understand that.  Would you like "
                               "to quit, redo the sentiment analysis, or redo the stylistic analysis?\n")
            f.write("\nCHATBOT:\nSorry, I didn't understand that.  Would you like "
                               "to quit, redo the sentiment analysis, or redo the stylistic analysis?\n")
            f.write("\nUSER:\n{0}\n".format(user_input))

    return next_state


# Function: run_chatbot(model, vectorizer=None):
# model: A trained classification model
# vectorizer: OPTIONAL; The trained vectorizer, if using Naive Bayes (leave empty otherwise)
# glove_reps: OPTIONAL; The pretrained GloVe model, if using other classification options (leave empty otherwise)
# Returns: This function does not return any values
#
# This function implements the main chatbot system---it runs different
# dialogue states depending on rules governed by the internal dialogue
# management logic, with each state handling its own input/output and internal
# processing steps.  The dialogue management logic is implemented as follows:
# welcome_state() (IN STATE) -> get_info_state() (OUT STATE)
# get_info_state() (IN STATE) -> sentiment_analysis_state() (OUT STATE)
# sentiment_analysis_state() (IN STATE) -> stylistic_analysis_state() (OUT STATE - First time sentiment_analysis_state() is run)
#                                    check_next_state() (OUT STATE - Subsequent times sentiment_analysis_state() is run)
# stylistic_analysis_state() (IN STATE) -> check_next_state() (OUT STATE)
# check_next_state() (IN STATE) -> sentiment_analysis_state() (OUT STATE option 1) or
#                                  stylistic_analysis_state() (OUT STATE option 2) or
#                                  terminate chatbot
def run_chatbot(model, vectorizer=None, glove_reps=None):
    next_state = welcome_state()  # Initialize the chatbot
    sentiment_analysis_counter = 0

    while next_state != "quit":
        if next_state == "get_user_info":
            next_state, name = get_info_state()
        elif next_state == "sentiment_analysis":
            # sentiment_analysis_state() always returns stylistic_analysis_state()
            # as its next state; this updates the next state to check_next_state()
            # if the sentiment analysis has already been run once (no reason to assume
            # the user would like to redo their stylistic analysis as well)
            next_state = sentiment_analysis_state(name, model, vectorizer, glove_reps)
            if sentiment_analysis_counter > 0:
                next_state = "check_next_action"
            sentiment_analysis_counter += 1
        elif next_state == "stylistic_analysis":
            next_state = stylistic_analysis_state()
        elif next_state == "check_next_action":
            next_state = check_next_state()

    f.close()

    return


# This is your main() function.  Use this space to try out and debug your code
# using your terminal.  The code you include in this space will not be graded.
if __name__ == "__main__":
    # Set things up ahead of time by training the TfidfVectorizer and Naive Bayes model
    documents, labels = load_as_list("dataset.csv")

    # Load the GloVe representations so that you can make use of them later
    glove_reps = load_glove(EMBEDDING_FILE)  # Use if you selected a GloVe model

    # Compute TFIDF representations so that you can make use of them later
    # vectorizer, tfidf_train = vectorize_train(documents)  # Use if you selected a TFIDF model

    # Instantiate and train the machine learning models
    # To save time, only uncomment the lines corresponding to the sentiment
    # analysis model you chose for your chatbot!

    # nb_tfidf, logistic_tfidf, svm_tfidf, mlp_tfidf = instantiate_models() # Uncomment to instantiate a TFIDF model
    nb_glove, logistic_glove, svm_glove, mlp_glove = instantiate_models()  # Uncomment to instantiate a GloVe model
    # nb_tfidf = train_model_tfidf(nb_tfidf, tfidf_train, labels)
    # nb_glove = train_model_glove(nb_glove, glove_reps, documents, labels)
    # logistic_tfidf = train_model_tfidf(logistic_tfidf, tfidf_train, labels)
    # logistic_glove = train_model_glove(logistic_glove, glove_reps, documents, labels)
    # svm_tfidf = train_model_tfidf(svm_tfidf, tfidf_train, labels)
    svm_glove = train_model_glove(svm_glove, glove_reps, documents, labels)
    # mlp_tfidf = train_model_tfidf(mlp_tfidf, tfidf_train, labels)
    # mlp_glove = train_model_glove(mlp_glove, glove_reps, documents, labels)

    # ***** New! *****

    # run_chatbot(mlp, glove_reps=glove_reps) # Example for running the chatbot with
                                        # MLP (make sure to comment/uncomment
                                        # properties of other functions as needed)
    run_chatbot(svm_glove, glove_reps=glove_reps) # Example for running the chatbot with SVM and GloVe---make sure your earlier functions are copied over for this to work correctly!
    f.close()
