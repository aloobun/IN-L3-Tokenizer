import sentencepiece as spm
import numpy as np

class IndianLanguageEncodeDecode:
    def __init__(self, model_path):
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(model_path)

    def test_languages(self):
        test_texts = {
            'Bengali': 'আমি বাংলাদেশ থেকে এসেছি। কলকাতা একটি সুন্দর শহর।',
            'Hindi': 'नमस्ते, मैं भारत से हूँ। दिल्ली बहुत बड़ा शहर है।',
            'Kannada': 'ನಾನು ಬೆಂಗಳೂರಿನಿಂದ ಬಂದಿದ್ದೇನೆ। ಕನ್ನಡ ಒಂದು ಸೋಂಪಿನ ಭಾಷೆ।',
            'Malayalam': 'ഞാൻ കേരളത്തിൽ നിന്നാണ്. കൊച്ചി ഒരു സുന്ദര നഗരം.',
            'Telugu': 'నేను తెలంగాణ నుంచి వచ్చాను. హైదరాబాద్ అద్భుతమైన నగరం.',
            'Tamil': 'நான் தமிழ்நாட்டைச் சேர்ந்தவன். சென்னை ஒரு பெரிய நகரம்.',
            'Gujarati': 'હું ગુજરાત થી આવ્યો છું। અમદાવાદ એક સુંદર શહેર છે।'
        }

        results = {}

        for language, text in test_texts.items():
            try:
                token_ids = self.sp_model.encode(text)

                token_strings = self.sp_model.encode(text, out_type=str)

                decoded_text = self.sp_model.decode(token_ids)

                results[language] = {
                    'original_text': text,
                    'token_ids_count': len(token_ids),
                    'token_strings_count': len(token_strings),
                    'decoded_text': decoded_text,
                    'text_match': text == decoded_text,
                    'token_id_stats': {
                        'min': min(token_ids),
                        'max': max(token_ids),
                        'mean': np.mean(token_ids)
                    }
                }

                print(f"\n{language} Analysis:")
                print(f"Original Text Length: {len(text)} characters")
                print(f"Token IDs Count: {len(token_ids)}")
                print(f"Token Strings: {token_strings}")
                print(f"Text Reconstruction: {results[language]['text_match']}")

            except Exception as e:
                results[language] = {'error': str(e)}
                print(f"{language} Error: {e}")

        return results

    def detailed_token_analysis(self, text):
        """
        Perform detailed token analysis on a given text
        """
        token_ids = self.sp_model.encode(text)

        token_strings = self.sp_model.encode(text, out_type=str)

        id_to_piece = [self.sp_model.id_to_piece(token_id) for token_id in token_ids]

        analysis = {
            'original_text': text,
            'original_length': len(text),
            'tokens': {
                'ids': token_ids,
                'strings': token_strings,
                'pieces': id_to_piece
            },
            'token_stats': {
                'total_tokens': len(token_ids),
                'unique_tokens': len(set(token_ids)),
                'avg_token_length': np.mean([len(token) for token in token_strings])
            }
        }

        return analysis

def main():
    MODEL_PATH = 'tokenizer.model'

    tokenizer = IndianLanguageEncodeDecode(MODEL_PATH)

    results = tokenizer.test_languages()

    sample_text = 'नमस्ते, मैं भारत से हूँ। दिल्ली बहुत बड़ा शहर है।'
    detailed_result = tokenizer.detailed_token_analysis(sample_text)

    #Optional
    import json
    with open('indian_language_tokenization_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
