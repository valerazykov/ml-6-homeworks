from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import xml.etree.ElementTree as ET
from collections import Counter


@dataclass(frozen=True)
class SentencePair:
    """
    Contains lists of tokens (strings) for source and target sentence
    """
    source: List[str]
    target: List[str]


@dataclass(frozen=True)
class TokenizedSentencePair:
    """
    Contains arrays of token vocabulary indices (preferably np.int32) for source and target sentence
    """
    source_tokens: np.ndarray
    target_tokens: np.ndarray


@dataclass(frozen=True)
class LabeledAlignment:
    """
    Contains arrays of alignments (lists of tuples (source_pos, target_pos)) for a given sentence.
    Positions are numbered from 1.
    """
    sure: List[Tuple[int, int]]
    possible: List[Tuple[int, int]]


def extract_sentences(filename: str) -> Tuple[List[SentencePair], List[LabeledAlignment]]:
    """
    Given a file with tokenized parallel sentences and alignments in XML format, return a list of sentence pairs
    and alignments for each sentence.

    Args:
        filename: Name of the file containing XML markup for labeled alignments

    Returns:
        sentence_pairs: list of `SentencePair`s for each sentence in the file
        alignments: list of `LabeledAlignment`s corresponding to these sentences
    """
    with open(filename, 'r', encoding='utf-8') as file:
        xml_content = file.read()
        # Замена '&' на '&amp;' за исключением уже экранированных последовательностей
        xml_content = xml_content.replace('&', '&amp;').replace('&amp;amp;', '&amp;')

    root = ET.fromstring(xml_content)

    sentence_pairs = []
    alignments = []

    for sentence in root.findall('s'):
        # Extract sentences in each language
        english = sentence.find('english').text
        czech = sentence.find('czech').text

        # Tokenize sentences (simple whitespace tokenization)
        source_tokens = english.split()
        target_tokens = czech.split()

        # Create a SentencePair instance
        sentence_pair = SentencePair(source=source_tokens,
                                     target=target_tokens)
        sentence_pairs.append(sentence_pair)

        # Extract sure and possible alignments
        sure_text = sentence.find('sure').text
        possible_text = sentence.find('possible').text

        sure_alignments = parse_alignments(sure_text)
        possible_alignments = parse_alignments(possible_text)

        # Create a LabeledAlignment instance
        alignment = LabeledAlignment(sure=sure_alignments,
                                     possible=possible_alignments)
        alignments.append(alignment)

    return sentence_pairs, alignments


def parse_alignments(alignment_text: str) -> List[Tuple[int, int]]:
    """
    Parse alignment text into a list of tuples.
    """
    alignments = []
    if alignment_text:
        for pair in alignment_text.split():
            source_idx, target_idx = map(int, pair.split('-'))
            alignments.append((source_idx, target_idx))
    return alignments


def get_token_to_index(sentence_pairs: List[SentencePair], freq_cutoff=None) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Given a parallel corpus, create two dictionaries token->index for source and target language.

    Args:
        sentence_pairs: list of `SentencePair`s for token frequency estimation
        freq_cutoff: if not None, keep only freq_cutoff -- natural number -- most frequent tokens in each language

    Returns:
        source_dict: mapping of token to a unique number (from 0 to vocabulary size) for source language
        target_dict: mapping of token to a unique number (from 0 to vocabulary size) target language
        
    Tip:
        Use cutting by freq_cutoff independently in src and target. Moreover, in both cases of freq_cutoff (None or not None) - you may get a different size of the dictionary

    """
    # Collect all tokens from each language
    source_tokens = [token for pair in sentence_pairs for token in pair.source]
    target_tokens = [token for pair in sentence_pairs for token in pair.target]

    # Count tokens
    source_counts = Counter(source_tokens).items()
    target_counts = Counter(target_tokens).items()

    # Apply frequency cutoff if specified
    if freq_cutoff is not None:
        source_counts = sorted(source_counts, key=lambda item: -item[1])[:freq_cutoff]
        target_counts = sorted(source_counts, key=lambda item: -item[1])[:freq_cutoff]

    # Create token to index dictionaries
    source_dict = {token: idx for idx, (token, _) in enumerate(source_counts)}
    target_dict = {token: idx for idx, (token, _) in enumerate(target_counts)}

    return source_dict, target_dict


def tokenize_sents(sentence_pairs: List[SentencePair], source_dict, target_dict) -> List[TokenizedSentencePair]:
    """
    Given a parallel corpus and token_to_index for each language, transform each pair of sentences from lists
    of strings to arrays of integers. If either source or target sentence has no tokens that occur in corresponding
    token_to_index, do not include this pair in the result.
    
    Args:
        sentence_pairs: list of `SentencePair`s for transformation
        source_dict: mapping of token to a unique number for source language
        target_dict: mapping of token to a unique number for target language

    Returns:
        tokenized_sentence_pairs: sentences from sentence_pairs, tokenized using source_dict and target_dict
    """
    tokenized_sentence_pairs = []

    for pair in sentence_pairs:
        # Convert source and target sentences to indices
        source_indices = [source_dict[token] for token in pair.source if token in source_dict]
        target_indices = [target_dict[token] for token in pair.target if token in target_dict]

        # Include the pair only if both sentences are non-empty
        if source_indices and target_indices:
            tokenized_sentence_pair = TokenizedSentencePair(
                source_tokens=np.array(source_indices, dtype=np.int32),
                target_tokens=np.array(target_indices, dtype=np.int32)
            )
            tokenized_sentence_pairs.append(tokenized_sentence_pair)

    return tokenized_sentence_pairs
