import unittest
import string
from homework1 import *

class TestNGramModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Simplified test tokens
        cls.test_tokens = ['to', 'be', 'or', 'not', 'to', 'be']
        
        # Create test models
        cls.test_bigram_counts, cls.test_bigram_probs = create_ngram_model(cls.test_tokens, 2)
        cls.test_trigram_counts, cls.test_trigram_probs = create_ngram_model(cls.test_tokens, 3)

    def test_create_ngram_model(self):
        """Test if create_ngram_model creates non-empty dictionaries"""
        self.assertTrue(len(self.test_bigram_counts) > 0)
        self.assertTrue(len(self.test_bigram_probs) > 0)

    def test_shakespeare_style_generation(self):
        """Test if bigram text generation produces Shakespearean-style text"""
        # Common Shakespearean bigrams to start with
        starting_bigrams = [
            ('to', 'be'),
            ('my', 'lord'),
            ('fair', 'lady'),
            ('thou', 'art')
        ]
        
        # Try each starting bigram and check if at least one produces valid text
        generated_valid_text = False
        for start_bigram in starting_bigrams:
            text = generate_text_from_bigram(start_bigram, 10)
            if (isinstance(text, str) and 
                len(text.split()) > 3 and 
                text != "Initial bigram not found in training data"):
                generated_valid_text = True
                break
        
        self.assertTrue(generated_valid_text, "Should generate valid Shakespearean-style text")

    def test_sample_next_token(self):
        """Test if sample_next_token returns something for valid input"""
        # Test with common Shakespeare bigram
        next_token = sample_next_token(('to', 'be'))
        self.assertIsInstance(next_token, str)

    def test_sample_next_token_ngram(self):
        """Test if sample_next_token_ngram handles valid and invalid inputs"""
        # Test with invalid trigram (should return None)
        next_token = sample_next_token_ngram(('not', 'a', 'real'), 
                                           self.test_trigram_probs)
        self.assertIsNone(next_token)

    def test_generate_text_from_bigram(self):
        """Test if text generation produces a string of appropriate length"""
        text = generate_text_from_bigram(('to', 'be'), 5)
        self.assertIsInstance(text, str)
        # Check if it generated some text (at least the initial bigram)
        self.assertTrue(len(text.split()) >= 2)

    def test_generate_text_from_ngram(self):
        """Test if n-gram text generation handles invalid input"""
        # Test with invalid input
        text = generate_text_from_ngram(('not', 'in', 'data'), 6, 
                                      self.test_trigram_probs)
        self.assertEqual(text, "Initial sequence not found in training data")

    def test_shakespeare_text_processing(self):
        """Test basic properties of processed Shakespeare text"""
        #check if we have tokens and they're strings
        self.assertTrue(len(tokens) > 0)
        self.assertTrue(all(isinstance(token, str) for token in tokens))

if __name__ == '__main__':
    unittest.main()
