# Module (I) Assignment: Building a Shakespearean Text Generator


# Shakespeare Text Generation using N-grams

This project implements an N-gram based text generation model trained on Shakespeare's works. It demonstrates the use of bigrams, trigrams, and quadgrams for generating Shakespeare-like text.

## Overview

The project uses statistical language modeling techniques to:
1. Process and analyze Shakespeare's texts
2. Build N-gram models (N=2,3,4)
3. Generate new text based on probability distributions

## Dataset

The project uses three Shakespeare plays from the NLTK Gutenberg corpus:
- Macbeth
- Hamlet
- Julius Caesar

## Requirements

The following Python libraries are required:

## Installation

To install dependencies:
```bash
pip install nltk spacy
python -m spacy download en_core_web_sm
```

## Setup

1. Install the required packages
2. Download the necessary NLTK data:
```python
import nltk
nltk.download('gutenberg')
```

## Running the Code

1. Run the main script:
```bash
python homework1.py
```

2. Run the tests:
```bash
python -m unittest test_homework1.py
```

## Implementation Details

The project implements several key components:

1. **Data Preprocessing**
   - Text cleaning (lowercase conversion, punctuation removal)
   - Tokenization using spaCy
   - N-gram creation

2. **Model Building**
   - Creation of count dictionaries for N-grams
   - Calculation of probability distributions
   - Implementation of sampling methods

3. **Text Generation**
   - Bigram-based generation
   - Trigram-based generation
   - Quadgram-based generation

4. **Testing**
   - Unit tests for all major components
   - Validation of Shakespearean style text generation
   - Testing of n-gram model creation and sampling
   - Basic text processing verification

## Output Examples

The program will generate text samples using different N-gram sizes, starting with famous Shakespeare phrases like "to be". The output includes:
- Model statistics (number of unique N-grams)
- Generated text samples
- Comparison between different N-gram models

## Model Comparison

- **Bigrams**: Provide more variety but less coherent text
- **Trigrams**: Balance between variety and coherence
- **Quadgrams**: Most coherent but potentially more repetitive

## Author

Laura Cuellar

## License

MIT License 