import nltk
import spacy
nlp = spacy.load("en_core_web_sm")
from nltk.util import bigrams
from collections import defaultdict
import random
import nltk
from nltk.util import ngrams
import string
from nltk.corpus import gutenberg


#TASK 1


# Access Shakespeare's Macbeth
macbeth_text = gutenberg.raw('shakespeare-macbeth.txt')
#print(macbeth_text[:500])  #Display the first 500 characters

#Accessing Hamlet from Guttenberg Libray 
hamlet_text = gutenberg.raw('shakespeare-hamlet.txt')

#Accessing Ceasar from Guttenberg Libray 
ceasar_text = gutenberg.raw('shakespeare-caesar.txt')

#Joining them all
shakespeare_text = macbeth_text + ' ' + hamlet_text + ' ' + ceasar_text

#Cleaning Text
shakespeare_text = shakespeare_text.lower()  #Make it lower case 
shakespeare_text = shakespeare_text.translate(str.maketrans('', '', string.punctuation))  #Remove Punctuation


#Further Clenaing
shakespeare_text = shakespeare_text.replace('\n', ' ')  #replace newlines with a space

#Tokenize
shakespeare_tokenized = nlp(shakespeare_text)

#Display tokenized words, excluding spaces
tokens = [token.text for token in shakespeare_tokenized if token.text.strip() != '']

# Create list of bigrams
bigram_pairs = list(bigrams(tokens))

# Initialize the counts dictionary
from_bigram_to_next_token_counts = defaultdict(lambda: defaultdict(int))

# Fill the counts dictionary
for i in range(len(tokens)-2):
    current_bigram = (tokens[i], tokens[i+1])
    next_token = tokens[i+2]
    from_bigram_to_next_token_counts[current_bigram][next_token] += 1

# TASK 2
from_bigram_to_next_token_probs = defaultdict(lambda: defaultdict(float))

for bigram, next_tokens in from_bigram_to_next_token_counts.items():
    total_count = sum(next_tokens.values())
    for next_token, count in next_tokens.items():
        from_bigram_to_next_token_probs[bigram][next_token] = count / total_count

# TASK 3
# Samples the next token based on a given bigram using probability distribution
def sample_next_token(bigram):
    if bigram not in from_bigram_to_next_token_probs:
        return None
    
    next_token_probs = from_bigram_to_next_token_probs[bigram]
    tokens = list(next_token_probs.keys())
    probabilities = list(next_token_probs.values())
    
    return random.choices(tokens, weights=probabilities, k=1)[0]

# TASK 4
# Generates text starting from a bigram for a specified number of words
def generate_text_from_bigram(initial_bigram, num_words):
    if initial_bigram not in from_bigram_to_next_token_probs:
        return "Initial bigram not found in training data"
    
    generated_words = list(initial_bigram)
    
    for _ in range(num_words - 2):
        current_bigram = (generated_words[-2], generated_words[-1])
        next_token = sample_next_token(current_bigram)
        
        if next_token is None:
            break
            
        generated_words.append(next_token)
    
    return ' '.join(generated_words)

# TASK 5
# Creates probability and count dictionaries for n-grams of any size
def create_ngram_model(tokens, n):
    counts_dict = defaultdict(lambda: defaultdict(int))
    
    for i in range(len(tokens) - n):
        current_ngram = tuple(tokens[i:i+n-1])
        next_token = tokens[i+n-1]
        counts_dict[current_ngram][next_token] += 1
    
    prob_dict = defaultdict(lambda: defaultdict(float))
    
    for ngram, next_tokens in counts_dict.items():
        total_count = sum(next_tokens.values())
        for next_token, count in next_tokens.items():
            prob_dict[ngram][next_token] = count / total_count
            
    return counts_dict, prob_dict

# Create trigram and quadgram models
from_trigram_to_next_token_counts, from_trigram_to_next_token_probs = create_ngram_model(tokens, 3)
from_quadgram_to_next_token_counts, from_quadgram_to_next_token_probs = create_ngram_model(tokens, 4)

# Test the text generation
print("\nText Generation Examples:")
print("Starting with 'to be':")
print(generate_text_from_bigram(('to', 'be'), 20))

# Samples the next token based on an n-gram sequence using probability distribution
def sample_next_token_ngram(ngram, prob_dict):
    if ngram not in prob_dict:
        return None
    
    next_token_probs = prob_dict[ngram]
    tokens = list(next_token_probs.keys())
    probabilities = list(next_token_probs.values())
    
    return random.choices(tokens, weights=probabilities, k=1)[0]

# Generates text starting from any n-gram sequence for a specified number of words
def generate_text_from_ngram(initial_sequence, num_words, prob_dict):
    if tuple(initial_sequence) not in prob_dict:
        return "Initial sequence not found in training data"
    
    generated_words = list(initial_sequence)
    n = len(initial_sequence) + 1  # n-gram size
    
    for _ in range(num_words - len(initial_sequence)):
        current_ngram = tuple(generated_words[-(n-1):])
        next_token = sample_next_token_ngram(current_ngram, prob_dict)
        
        if next_token is None:
            break
            
        generated_words.append(next_token)
    
    return ' '.join(generated_words)

# Analyzes and displays statistics and sample generations for different n-gram models
def analyze_model_quality():
    print("\nModel Statistics:")
    print(f"Number of unique bigrams: {len(from_bigram_to_next_token_counts)}")
    print(f"Number of unique trigrams: {len(from_trigram_to_next_token_counts)}")
    print(f"Number of unique quadgrams: {len(from_quadgram_to_next_token_counts)}")
    
    print("\nSample generations with different n-grams (20 words each):")
    print("\nBigram starting with 'to be':")
    print(generate_text_from_bigram(('to', 'be'), 20))
    
    print("\nTrigram starting with 'to be or':")
    print(generate_text_from_ngram(('to', 'be', 'or'), 20, from_trigram_to_next_token_probs))
    
    print("\nQuadgram starting with 'to be or not':")
    print(generate_text_from_ngram(('to', 'be', 'or', 'not'), 20, from_quadgram_to_next_token_probs))

analyze_model_quality()