import math


def calculate_f1_score(tp, fp, fn):
    """
    Calculate the F1-Score for a classification task.

    Args:
        tp (int): true positives, >= 0
        fp (int): false positives, >= 0
        fn (int): false negatives, >= 0

    Formula:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (precision * recall) / (precision + recall)

    Returns:
        f1_score (float): the F1-Score value,
        or None if the input is invalid.
    """
    check = {tp: "TP", fp: "FP", fn: "FN"}
    for item in check:
        # 1) Validate types (reject bool because bool is a subclass of int)
        if isinstance(item, bool) or not isinstance(item, int):
            print(f"{check[item]} must be an int")
            return None
        # 2) Validate non-negative values
        if item < 0:
            print(f"{check[item]} must be zero or a positive number")
            return None
    # 3) Calculate Precision, Recall, F1-Score
    try:
        if tp == 0:
            return 0.0
            
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score
    except ZeroDivisionError:
        return None


if __name__ == "__main__":
    # Basic test
    assert round(calculate_f1_score(tp=2, fp=3, fn=5), 2) == 0.33
    print(round(calculate_f1_score(tp=2, fp=4, fn=5), 2))

    # Edge cases
    assert calculate_f1_score(True, 3, 5) is None       # bool rejected
    assert calculate_f1_score(2.5, 3, 5) is None        # float rejected
    assert calculate_f1_score(-1, 3, 5) is None         # negative rejected
    assert calculate_f1_score(0, 3, 5) == 0.0           # tp=0 is valid
    print("All tests passed.")
