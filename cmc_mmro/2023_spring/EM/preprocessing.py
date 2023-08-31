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

    f = open(filename, 'r')
    root = ET.fromstring(f.read().replace('&', '&amp;'))
    sentence_pairs = []
    alignments = []
    for sent in root:
        sure = []
        possible = []
        source = sent[0].text.split(' ')
        target = sent[1].text.split(' ')
        sentence_pairs.append(SentencePair(source, target))
        for k in [2, 3]:
            if sent[k].text is not None:
                pairs = sent[k].text.split(' ')
                imgs = [tuple(map(int, pair.split('-'))) for pair in pairs]
                if k == 2:
                    sure = imgs
                else:
                    possible = imgs
        alignment = LabeledAlignment(sure, possible)
        alignments.append(alignment)
    #print(sentence_pairs)
    #print(alignments)
    return sentence_pairs, alignments


pass
if __name__ == '__main__':
    extract_sentences('/Users/nickpan/PythonProjects/sem-6/ml/task2/data/data/named_entities/project_syndicate_ne01.j.wa')


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
        Use cutting by freq_cutoff independently in src and target. Moreover in both cases of freq_cutoff (None or not None) - you may get a different size of the dictionary
    """
    s = Counter()
    t = Counter()
    for sentence_pair in sentence_pairs:
        s.update(sentence_pair.source)
        t.update(sentence_pair.target)

    most_fr = s.most_common(freq_cutoff)
    source_dict = dict()
    for unique_id, word_count in enumerate(most_fr):
        source_dict[word_count[0]] = unique_id
    
    most_fr = t.most_common(freq_cutoff)
    target_dict = dict()
    for unique_id, word_count in enumerate(most_fr):
        target_dict[word_count[0]] = unique_id
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
    for sent_pair in sentence_pairs:
        tokenized_source = []
        tokenized_target = []
        is_in_tokendict = True
        for word_source in sent_pair.source:
            is_in_tokendict = word_source in source_dict
            if not is_in_tokendict:
                break
            tokenized_source.append(source_dict[word_source])
        if is_in_tokendict:
            for word_target in sent_pair.target:
                is_in_tokendict = word_target in target_dict
                if not is_in_tokendict:
                    break
                tokenized_target.append(target_dict[word_target])
        if is_in_tokendict:
            tokenized_sentence_pairs.append(
                TokenizedSentencePair(np.array(tokenized_source),
                                      np.array(tokenized_target)))
    return tokenized_sentence_pairs
    pass