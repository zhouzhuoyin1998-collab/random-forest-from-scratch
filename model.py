"""
Random Forest from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impurity
import numpy as np
def impurity(labels):
    """Return a non-negative impurity score for a 1D array of integer class labels."""
    # TODO: score how mixed the labels are; 0 for a pure set, larger for more mixed sets.
    labels = np.array(labels)
    n,counts = np.unique(labels, return_counts = True)
    total = len(labels)
    p = counts/total
    return 1 - np.sum(p**2)

# Step 2 - split_dataset
import numpy as np

def split_dataset(features, labels, feature_index, threshold):
    # TODO: partition rows into left (feature <= threshold) and right (feature > threshold)
    labels = np.array(labels)
    features = np.array(features)

    feature_values = features[:, feature_index]

    left_mask = feature_values <= threshold
    right_mask = feature_values > threshold

    left_features = features[left_mask]
    left_labels = labels[left_mask]

    right_features = features[right_mask]
    right_labels = labels[right_mask]

    return left_features, left_labels, right_features, right_labels

# Step 3 - split_score
def split_score(parent_labels, left_labels, right_labels):
    # TODO: return a score where higher means the children are purer than the parent.
    n = len(parent_labels)

    parent_imp = impurity(parent_labels)

    left_imp = impurity(left_labels)
    right_imp = impurity(right_labels)

    weighted_children = (
        len(left_labels) / n * left_imp
        + len(right_labels) / n * right_imp
    )

    return parent_imp - weighted_children

# Step 4 - best_split
import numpy as np

def best_split(features, labels, feature_indices):
    # TODO: search feature_indices for the (feature, threshold) that best improves purity.
    features = np.array(features)
    labels = np.array(labels)

    best_feature = None
    best_threshold = None
    best_score = 0.0

    for feature_index in feature_indices:
        values = np.unique(features[:, feature_index])

        # Try midpoints between consecutive unique values
        for i in range(len(values) - 1):
            threshold = (values[i] + values[i + 1]) / 2

            left_features, left_labels, right_features, right_labels = split_dataset(
                features,
                labels,
                feature_index,
                threshold
            )

            score = split_score(
                labels,
                left_labels,
                right_labels
            )

            if score > best_score:
                best_score = score
                best_feature = feature_index
                best_threshold = threshold

    return {
        "feature_index": best_feature,
        "threshold": best_threshold,
        "score": float(best_score)
    }

# Step 5 - should_stop
def should_stop(labels, depth, max_depth, min_samples_split):
    """Return True if this node should become a leaf instead of splitting further."""
    # TODO: decide whether to stop growing based on purity, depth, and size...
    labels = np.array(labels)

    # 1. Pure node: all labels are the same
    if len(np.unique(labels)) == 1:
        return True

    # 2. Reached maximum depth
    if depth >= max_depth:
        return True

    # 3. Too few samples to split
    if len(labels) < min_samples_split:
        return True

    return False

# Step 6 - leaf_prediction
def leaf_prediction(labels):
    # TODO: choose a single class label to output for a leaf given the labels that reached it
    labels = np.array(labels)

    values, counts = np.unique(labels, return_counts=True)

    return int(values[np.argmax(counts)])

# Step 7 - build_tree
def build_tree(features, labels, max_depth=10, min_samples_split=2, feature_subset=None, depth=0):
    # TODO: recursively grow a decision tree, returning a nested dict of leaf/internal nodes.
    features = np.array(features)
    labels = np.array(labels)

    # 1. Stopping condition
    if should_stop(
        labels,
        depth,
        max_depth,
        min_samples_split
    ):
        return {
            "leaf": True,
            "prediction": leaf_prediction(labels)
        }

    # 2. Decide which features may be used
    if feature_subset is None:
        feature_indices = list(range(features.shape[1]))
    else:
        feature_indices = feature_subset

    # 3. Find best split
    split = best_split(
        features,
        labels,
        feature_indices
    )

    # 4. No useful split
    if split["feature_index"] is None or split["score"] <= 0:
        return {
            "leaf": True,
            "prediction": leaf_prediction(labels)
        }

    feature_index = split["feature_index"]
    threshold = split["threshold"]

    # 5. Split dataset
    left_features, left_labels, right_features, right_labels = split_dataset(
        features,
        labels,
        feature_index,
        threshold
    )

    # 6. Recursively build children
    left_node = build_tree(
        left_features,
        left_labels,
        max_depth,
        min_samples_split,
        feature_subset,
        depth + 1
    )

    right_node = build_tree(
        right_features,
        right_labels,
        max_depth,
        min_samples_split,
        feature_subset,
        depth + 1
    )

    # 7. Return internal node
    return {
        "leaf": False,
        "feature_index": feature_index,
        "threshold": threshold,
        "left": left_node,
        "right": right_node
    }

# Step 8 - predict_example_tree (not yet solved)
# TODO: implement

# Step 9 - predict_tree (not yet solved)
# TODO: implement

# Step 10 - bootstrap_sample (not yet solved)
# TODO: implement

# Step 11 - feature_subset (not yet solved)
# TODO: implement

# Step 12 - train_forest (not yet solved)
# TODO: implement

# Step 13 - combine_predictions (not yet solved)
# TODO: implement

# Step 14 - predict_forest (not yet solved)
# TODO: implement

# Step 15 - accuracy (not yet solved)
# TODO: implement

