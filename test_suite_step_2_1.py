import os
import json
import json
import numpy as np
import matplotlib.pyplot as plt
import sentencepiece as spm
from typing import List, Dict, Any

class IndianLanguageTokenizerTestSuite:
    def __init__(self, model_path: str, test_data_paths: List[str]):
        """
        Initialize the test suite with tokenizer model and test data for Indian languages
        """
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(model_path)
        
        self.languages = [
            'bengali', 'hindi', 'kannada', 
            'malayalam', 'telugu', 'tamil', 'gujarati'
        ]
        
        self.edge_cases = {
            'bengali': {
                'script_test': 'আমি বাংলাদেশ থেকে এসেছি। কলকাতা একটি সুন্দর শহর।',
                'unicode_test': 'বাংলা ০১২৩৪৫৬৭৮৯ vowels: অ আ ই ঈ উ ঊ',
                'special_chars': 'বাংলা! @ # $ % ^ & * ( ) _ + = [ ] { }',
            },
            'hindi': {
                'script_test': 'नमस्ते, मैं भारत से हूँ। दिल्ली बहुत बड़ा शहर है।',
                'unicode_test': 'हिन्दी १२३४५६७८९ vowels: अ आ इ ई उ ऊ',
                'special_chars': 'हिन्दी! @ # $ % ^ & * ( ) _ + = [ ] { }',
            },
            'kannada': {
                'script_test': 'ನಾನು ಬೆಂಗಳೂರಿನಿಂದ ಬಂದಿದ್ದೇನೆ। ಕನ್ನಡ ಒಂದು ಸೋಂಪಿನ ಭಾಷೆ।',
                'unicode_test': 'ಕನ್ನಡ ೦೧೨೩೪೫೬೭೮೯ vowels: ಅ ಆ ಇ ಈ ಉ ಊ',
                'special_chars': 'ಕನ್ನಡ! @ # $ % ^ & * ( ) _ + = [ ] { }',
            },
            'malayalam': {
                'script_test': 'ഞാൻ കേരളത്തിൽ നിന്നാണ്. കൊച്ചി ഒരു സുന്ദര നഗരം.',
                'unicode_test': 'മലയാളം ൦൧൨൩൪൫൬൭൮൯ vowels: അ ആ ഇ ഈ ഉ ഊ',
                'special_chars': 'മലയാളം! @ # $ % ^ & * ( ) _ + = [ ] { }',
            },
            'telugu': {
                'script_test': 'నేను తెలంగాణ నుంచి వచ్చాను. హైదరాబాద్ అద్భుతమైన నగరం.',
                'unicode_test': 'తెలుగు ౦౧౨౩౪౫౬౭൮൯ vowels: అ ఆ ఇ ఈ ఉ ఊ',
                'special_chars': 'తెలుగు! @ # $ % ^ & * ( ) _ + = [ ] { }',
            },
            'tamil': {
                'script_test': 'நான் தமிழ்நாட்டைச் சேர்ந்தவன். சென்னை ஒரு பெரிய நகரம்.',
                'unicode_test': 'தமிழ் ௦௧௨௩௪௫௬௭௮௯ vowels: அ ஆ இ ஈ உ ஊ',
                'special_chars': 'தமிழ்! @ # $ % ^ & * ( ) _ + = [ ] { }',
            },
            'gujarati': {
                'script_test': 'હું ગુજરાત થી આવ્યો છું। અમદાવાદ એક સુંદર શહેર છે।',
                'unicode_test': 'ગુજરાતી ૦૧૨૩૪૫૬૭૮૯ vowels: અ આ ઇ ઈ ઉ ઊ',
                'special_chars': 'ગુજરાતી! @ # $ % ^ & * ( ) _ + = [ ] { }',
            }
        }
        
        self.test_data = self._load_test_data(test_data_paths)
        
        self.results = {
            'coverage': {},
            'complexity': {},
            'language_analysis': {},
            'edge_cases': {}
        }
    
    def _load_test_data(self, data_paths: List[str]) -> Dict[str, List[str]]:
        """
        Load test data from multiple files
        """
        test_data = {lang: [] for lang in self.languages}
        
        for path in data_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    texts = f.readlines()
                    
                for i, text in enumerate(texts):
                    lang = self.languages[i % len(self.languages)]
                    test_data[lang].append(text.strip())
            except Exception as e:
                print(f"Error loading {path}: {e}")
        
        return test_data
    
    def unicode_coverage_analysis(self) -> Dict[str, Any]:
        """
        Analyze Unicode coverage for each language
        """
        unicode_results = {}
        
        for lang, edge_cases in self.edge_cases.items():
            unicode_test = edge_cases['unicode_test']
            tokens = self.sp_model.encode(unicode_test, out_type=str)
            
            unicode_results[lang] = {
                'original_text': unicode_test,
                'tokens': tokens,
                'token_count': len(tokens),
                'unique_tokens': len(set(tokens)),
                'coverage_ratio': len(set(tokens)) / len(tokens)
            }
        
        self.results['unicode_coverage'] = unicode_results
        return unicode_results
    
    def language_specific_edge_cases(self) -> Dict[str, Any]:
        """
        Test tokenization with language-specific edge cases
        """
        edge_case_results = {}
        
        for lang, cases in self.edge_cases.items():
            language_results = {}
            
            for case_name, text in cases.items():
                try:
                    tokens = self.sp_model.encode(text, out_type=str)
                    language_results[case_name] = {
                        'tokens': tokens,
                        'token_count': len(tokens),
                        'unique_tokens': len(set(tokens))
                    }
                except Exception as e:
                    language_results[case_name] = {
                        'error': str(e)
                    }
            
            edge_case_results[lang] = language_results
        
        self.results['edge_cases'] = edge_case_results
        return edge_case_results
    
    def script_complexity_analysis(self) -> Dict[str, Any]:
        """
        Analyze tokenization complexity across different scripts
        """
        complexity_results = {}
        
        for lang, texts in self.test_data.items():
            text = self.edge_cases[lang]['script_test']
            
            tokens = self.sp_model.encode(text, out_type=str)
            
            complexity_results[lang] = {
                'original_text_length': len(text),
                'tokens': tokens,
                'token_count': len(tokens),
                'avg_token_length': np.mean([len(token) for token in tokens]),
                'token_diversity': len(set(tokens)) / len(tokens)
            }
        
        self.results['script_complexity'] = complexity_results
        return complexity_results
    
    def generate_unicode_visualization(self):
        plt.figure(figsize=(15, 10))
        
        unicode_results = self.results.get('unicode_coverage', {})
        
        plt.subplot(2, 1, 1)
        token_counts = [result['token_count'] for result in unicode_results.values()]
        plt.bar(unicode_results.keys(), token_counts)
        plt.title('Token Count in Unicode Test Texts')
        plt.xlabel('Language')
        plt.ylabel('Number of Tokens')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 1, 2)
        coverage_ratios = [result['coverage_ratio'] for result in unicode_results.values()]
        plt.bar(unicode_results.keys(), coverage_ratios)
        plt.title('Token Diversity Ratio')
        plt.xlabel('Language')
        plt.ylabel('Unique Tokens / Total Tokens')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('unicode_token_analysis.png')
        plt.close()
    
    def run_all_tests(self):
        print("Running Comprehensive Indian Language Tokenizer Test Suite...")
        
        print("1. Unicode Coverage Analysis...")
        self.unicode_coverage_analysis()
        
        print("2. Language-Specific Edge Cases...")
        self.language_specific_edge_cases()
        
        print("3. Script Complexity Analysis...")
        self.script_complexity_analysis()
        
        print("4. Generating Unicode Visualizations...")
        self.generate_unicode_visualization()
        
        print("Test Suite Complete!")
        
        return self.results

if __name__ == "__main__":
    MODEL_PATH = 'path/to/your/tokenizer.model'
    TEST_DATA_PATHS = [
        'path/to/indian_languages/test_data1.txt',
        'path/to/indian_languages/test_data2.txt'
    ]
    
    test_suite = IndianLanguageTokenizerTestSuite(MODEL_PATH, TEST_DATA_PATHS)
    results = test_suite.run_all_tests()
    #optional
    with open('indian_language_tokenizer_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
