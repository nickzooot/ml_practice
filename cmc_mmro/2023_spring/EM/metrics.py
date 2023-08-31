from typing import List, Tuple

from preprocessing import LabeledAlignment


def compute_precision(reference: List[LabeledAlignment], predicted: List[List[Tuple[int, int]]]) -> Tuple[int, int]:
    """
    Computes the numerator and the denominator of the precision for predicted alignments.
    Numerator : |predicted and possible|
    Denominator: |predicted|
    Note that for correct metric values `sure` needs to be a subset of `possible`, but it is not the case for input data.
    Args:
        reference: list of alignments with fields `possible` and `sure`
        predicted: list of alignments, i.e. lists of tuples (source_pos, target_pos)
    Returns:
        intersection: number of alignments that are both in predicted and possible sets, summed over all sentences
        total_predicted: total number of predicted alignments over all sentences
    """
    global_predicted = 0
    global_predicted_and_possible = 0
    for r, pr in zip(reference, predicted):
        possible = set(r.possible).union(r.sure)
        global_predicted_and_possible += len(possible.intersection(set(pr)))
        global_predicted += len(pr)
    return global_predicted_and_possible, global_predicted


def compute_recall(reference: List[LabeledAlignment], predicted: List[List[Tuple[int, int]]]) -> Tuple[int, int]:
    """
    Computes the numerator and the denominator of the recall for predicted alignments.
    Numerator : |predicted and sure|
    Denominator: |sure|
    Args:
        reference: list of alignments with fields `possible` and `sure`
        predicted: list of alignments, i.e. lists of tuples (source_pos, target_pos)
    Returns:
        intersection: number of alignments that are both in predicted and sure sets, summed over all sentences
        total_predicted: total number of sure alignments over all sentences
    """
    global_predicted_and_sure = 0
    global_predicted = 0
    for r, pr, in zip(reference, predicted):
        global_predicted_and_sure += len(set(r.sure).intersection(set(pr)))
        global_predicted += len(r.sure)
    return global_predicted_and_sure, global_predicted
    pass


def compute_aer(reference: List[LabeledAlignment], predicted: List[List[Tuple[int, int]]]) -> float:
    """
    Computes the alignment error rate for predictions.
    AER=1-(|predicted and possible|+|predicted and sure|)/(|predicted|+|sure|)
    Please use compute_precision and compute_recall to reduce code duplication.
    Args:
        reference: list of alignments with fields `possible` and `sure`
        predicted: list of alignments, i.e. lists of tuples (source_pos, target_pos)
    Returns:
        aer: the alignment error rate
    """
    recall = compute_recall(reference, predicted)
    precision = compute_precision(reference, predicted)
    return 1 - (recall[0] + precision[0]) / (precision[1] + recall[1])
    pass