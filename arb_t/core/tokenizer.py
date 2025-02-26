from typing import List, Dict, Optional, Union
import torch
from transformers import PreTrainedTokenizer, AutoTokenizer
import re
from collections import defaultdict
import numpy as np

class AdaptiveTokenizer:
    """A tokenizer that can adapt its vocabulary based on observed text."""
    
    def __init__(
        self,
        base_tokenizer: Optional[str] = "gpt2",
        max_vocab_size: int = 50000,
        min_freq: int = 2
    ):
        # Initialize with a pre-trained tokenizer as base
        self.base_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(base_tokenizer)
        
        # Set padding configuration
        self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
        self.base_tokenizer.padding_side = "right"
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        
        # Tracking new tokens
        self.token_frequencies = defaultdict(int)
        self.new_tokens: Dict[str, int] = {}
        self.next_token_id = len(self.base_tokenizer)
        
    def update_vocab(self, text: str):
        """Update vocabulary based on new text."""
        # Tokenize with base tokenizer first
        base_tokens = self.base_tokenizer.tokenize(text)
        
        # Find unknown or partial tokens
        unknown_sequences = []
        current_sequence = []
        
        for token in base_tokens:
            if token.startswith('�') or token in self.base_tokenizer.all_special_tokens:
                if current_sequence:
                    unknown_sequences.append(''.join(current_sequence))
                    current_sequence = []
            else:
                current_sequence.append(token)
                
        if current_sequence:
            unknown_sequences.append(''.join(current_sequence))
            
        # Extract potential new tokens using various methods
        for sequence in unknown_sequences:
            # Word-level tokens
            words = re.findall(r'\w+|[^\w\s]', sequence)
            for word in words:
                self.token_frequencies[word] += 1
            
            # Subword tokens using byte-pair encoding approach
            subwords = self._extract_subwords(sequence)
            for subword in subwords:
                self.token_frequencies[subword] += 1
                
        # Update vocabulary if needed
        self._update_vocab_if_needed()
        
    def _extract_subwords(self, sequence: str) -> List[str]:
        """Extract subword tokens using a simplified BPE-like approach."""
        # Start with character-level splits
        chars = list(sequence)
        
        # Find common pairs
        pairs = defaultdict(int)
        for i in range(len(chars) - 1):
            pair = (chars[i], chars[i + 1])
            pairs[pair] += 1
            
        # Merge most common pairs
        common_pairs = sorted(pairs.items(), key=lambda x: x[1], reverse=True)[:10]
        
        subwords = []
        for (first, second), _ in common_pairs:
            merged = first + second
            if len(merged) <= 10:  # Limit subword length
                subwords.append(merged)
                
        return subwords
        
    def _update_vocab_if_needed(self):
        """Update vocabulary with new frequent tokens."""
        # Cache the base vocabulary for efficiency
        if not hasattr(self, '_base_vocab_cache'):
            self._base_vocab_cache = set(self.base_tokenizer.get_vocab().keys())
        
        # Find frequent tokens not in base vocabulary
        new_candidates = {
            token: freq
            for token, freq in self.token_frequencies.items()
            if freq >= self.min_freq and token not in self._base_vocab_cache
        }
        
        # Sort by frequency and filter existing tokens
        sorted_candidates = sorted(
            [(t, f) for t, f in new_candidates.items() if t not in self.new_tokens],
            key=lambda x: x[1],
            reverse=True
        )[:100]  # Limit number of new tokens per update
        
        # Add new tokens while respecting max_vocab_size
        available_slots = self.max_vocab_size - len(self.base_tokenizer) - len(self.new_tokens)
        
        for token, _ in sorted_candidates[:available_slots]:
            if token not in self.new_tokens:
                self.new_tokens[token] = self.next_token_id
                self.next_token_id += 1
                
    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        padding: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None
    ) -> torch.Tensor:
        """Encode text into token IDs."""
        if isinstance(text, str):
            text = [text]
            
        # First encode with base tokenizer
        base_encoding = self.base_tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Handle new tokens
        if self.new_tokens:
            # Convert to list for modification
            token_ids = base_encoding['input_ids'].tolist()
            
            # Replace unknown tokens with new token IDs
            for i, seq in enumerate(token_ids):
                for j, token_id in enumerate(seq):
                    if token_id == self.base_tokenizer.unk_token_id:
                        # Try to match new tokens
                        token_text = self.base_tokenizer.decode([token_id])
                        if token_text in self.new_tokens:
                            seq[j] = self.new_tokens[token_text]
                            
            # Convert back to tensor
            base_encoding['input_ids'] = torch.tensor(token_ids)
            
        return base_encoding
        
    def decode(self, token_ids: torch.Tensor) -> Union[str, List[str]]:
        """Decode token IDs back to text."""
        # Convert to list for processing
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
            
        decoded = []
        for seq in token_ids.tolist():
            # Split into base and new tokens
            base_tokens = []
            text_pieces = []
            
            for token_id in seq:
                if token_id < len(self.base_tokenizer):
                    base_tokens.append(token_id)
                else:
                    # Decode and append any accumulated base tokens
                    if base_tokens:
                        text_pieces.append(self.base_tokenizer.decode(base_tokens))
                        base_tokens = []
                    
                    # Add new token text
                    for token, idx in self.new_tokens.items():
                        if idx == token_id:
                            text_pieces.append(token)
                            break
                            
            # Decode any remaining base tokens
            if base_tokens:
                text_pieces.append(self.base_tokenizer.decode(base_tokens))
                
            decoded.append(''.join(text_pieces))
            
        return decoded[0] if len(decoded) == 1 else decoded
