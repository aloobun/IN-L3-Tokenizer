import os
import fire
import logging
import regex as re
from typing import List, Set, Optional
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
from tokenizers import Tokenizer

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def contains_non_english(text: str) -> bool:
    """
    Check if text contains non-English characters.
    """
    return bool(re.search(r'[^\x00-\x7F]', text))

def validate_tokenizer_model(tokenizer_path: str) -> bool:
    """
    Validate the existence and readability of the tokenizer model file.
    """
    model_path = os.path.join(tokenizer_path, "tokenizer.model")
    if not os.path.exists(model_path):
        logger.error(f"Tokenizer model file not found: {model_path}")
        return False
    
    try:
        with open(model_path, "rb") as f:
            f.read()
        return True
    except IOError as e:
        logger.error(f"Error reading tokenizer model: {e}")
        return False

def filter_and_validate_new_tokens(
    new_tokens: List[str], 
    original_tokens: Set[str],
    max_token_length: int = 50
) -> List[str]:
    """
    Filter and validate new tokens before adding.
    """
    unique_new_tokens = list(set(new_tokens) - original_tokens)
    
    validated_tokens = []
    for token in unique_new_tokens:
        if not token or not token.strip():
            continue
        
        if len(token) > max_token_length:
            logger.warning(f"Long token encountered and skipped: {token}")
            continue
        
        if contains_non_english(token):
            validated_tokens.append(token)
    
    return validated_tokens

def test_tokenization(
    tokenizer: AutoTokenizer, 
    test_texts: List[str], 
    description: str = "Tokenization Test"
) -> None:
    """
    Test tokenization and log results.
    """
    logger.info(f"\n--- {description} ---")
    for text in test_texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        decoded_tokens = [tokenizer.decode([token]) for token in tokens]
        
        logger.info(f"Text: {text}")
        logger.info(f"Tokens: {decoded_tokens}")
        logger.info(f"Number of tokens: {len(tokens)}\n")

def merge_tokenizers(
    llama_model_id: str, 
    new_tokenizer_path: str, 
    extended_tokenizer_save_path: str,
    test_texts: Optional[List[str]] = None
) -> None:
    """
    Merge tokenizers with comprehensive error handling and validation.
    """
    try:
        if not os.path.exists(new_tokenizer_path):
            raise ValueError(f"New tokenizer path does not exist: {new_tokenizer_path}")
        
        if not validate_tokenizer_model(new_tokenizer_path):
            raise ValueError("Invalid tokenizer model")
        
        try:
            original_tokenizer = AutoTokenizer.from_pretrained(llama_model_id)
        except Exception as e:
            logger.error(f"Failed to load original tokenizer: {e}")
            raise
        
        if test_texts:
            test_tokenization(
                original_tokenizer, 
                test_texts, 
                "Initial Tokenization Before Adding New Tokens"
            )
        
        new_tokenizer_spm = sp_pb2_model.ModelProto()
        with open(os.path.join(new_tokenizer_path, "tokenizer.model"), "rb") as f:
            new_tokenizer_spm.ParseFromString(f.read())
        
        original_tokenset = set(original_tokenizer.get_vocab().keys())
        logger.info(f"Original tokenizer size: {len(original_tokenset)} tokens")
        logger.info(f"New tokenizer size: {len(new_tokenizer_spm.pieces)} tokens")
        
        new_tokens = [p.piece for p in new_tokenizer_spm.pieces]
        validated_new_tokens = filter_and_validate_new_tokens(
            new_tokens, 
            original_tokenset
        )
        
        try:
            added_tokens = original_tokenizer.add_tokens(validated_new_tokens)
            logger.info(f"Added {added_tokens} new tokens")
        except Exception as e:
            logger.error(f"Failed to add tokens: {e}")
            raise
        
        os.makedirs(extended_tokenizer_save_path, exist_ok=True)
        
        original_tokenizer.save_pretrained(extended_tokenizer_save_path)
        logger.info(f"Extended tokenizer saved to {extended_tokenizer_save_path}")
        
        if test_texts:
            test_tokenization(
                original_tokenizer, 
                test_texts, 
                "Tokenization After Adding New Tokens"
            )
        
        verify_tokenizer_integrity(llama_model_id, extended_tokenizer_save_path)
    
    except Exception as e:
        logger.error(f"Tokenizer merging failed: {e}")
        raise

def verify_tokenizer_integrity(
    original_model_id: str, 
    extended_tokenizer_path: str
) -> bool:
    """
    Verify that the extended tokenizer maintains the original tokenizer's mapping.
    """
    try:
        tok1 = AutoTokenizer.from_pretrained(original_model_id)
        tok2 = Tokenizer.from_file(os.path.join(extended_tokenizer_path, "tokenizer.json"))
        
        check_range = min(1000, len(tok1))
        for i in range(check_range):
            assert tok1.convert_ids_to_tokens(i) == tok2.id_to_token(i), \
                f"Token mismatch at index {i}."
        
        logger.info("Tokenizer integrity verified successfully")
        return True
    except AssertionError as e:
        logger.error(f"Tokenizer integrity check failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Error during tokenizer integrity verification: {e}")
        return False

def main(
    new_tokenizer_path: str, 
    extended_tokenizer_save_path: str, 
    test_texts: Optional[List[str]] = None
):
    """
    Main function to merge tokenizers.
    """
    llama_model_id = "./meta-llama-3-model"
    
    if test_texts is None:
        test_texts = [
            "সুস্মিতা থেকে দিয়া, সৌন্দর্য প্রতিযোগিতায় দেশের মুখ উজ্জ্বল করেছেন এই বলিউড অভিনেত্রীরা",
            "नमस्ते, मैं भारत से हूँ। दिल्ली बहुत बड़ा शहर है।",
            "Machine learning is transforming various industries.",
            "ನಾನು ಬೆಂಗಳೂರಿನಿಂದ ಬಂದಿದ್ದೇನೆ। ಕನ್ನಡ ಒಂದು ಸೋಂಪಿನ ಭಾಷೆ।",
            "ഞാൻ കേരളത്തിൽ നിന്നാണ്. കൊച്ചി ഒരു സുന്ദര നഗരം.",
            "నేను తెలంగాణ నుంచి వచ్చాను. హైదరాబాద్ అద్భుతమైన నగరం.",
            "நான் தமிழ்நாட்டைச் சேர்ந்தவன். சென்னை ஒரு பெரிய நகரம்.",
            "ગુજરાત થી આવ્યો છું। અમદાવાદ એક સુંદર શહેર છે।"
        ]
    
    try:
        merge_tokenizers(
            llama_model_id, 
            new_tokenizer_path, 
            extended_tokenizer_save_path, 
            test_texts
        )
    except Exception as e:
        logger.error(f"Tokenizer merging process failed: {e}")
        raise

if __name__ == "__main__":
    fire.Fire(main)
