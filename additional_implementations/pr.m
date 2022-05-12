function [precision, recall, f1_score, acc] = pr(gt, label)
    %{
    Description:
        Function to calculate scores for precision, recall, f1 score and accuracy
    Arguments:
        gt: Ground Truth Segmentation Mask
        label: Predicted Label Mask
    Returns: 
        precision: Classwise Precision score (P = TP / (TP + FP))
        recall: Classwise Recall score (R = TP / (TP + FN))
        f1_score: Classwise F1 score (2 * (P * R) / (R + P))
        acc: Classwise Accuracy score(Acc = (TP + TN)/ (TP + TN + FP + FN))
    %}
    % Create dummy arrays to hold scores
    precision = zeros(6,1,'double');
    recall = zeros(6,1,'double');
    f1_score = zeros(6,1,'double');
    acc = zeros(6,1,'double');
    for i=0:5   % Iterate over all classes
        % Create ground truth binary mask
        gt_mask = zeros((size(gt)));
        gt_vals = gt == i;
        gt_mask(gt_vals) = 1;

        % Create prediction binary mask
        pred_mask = zeros((size(gt)));
        vals = label == i;
        pred_mask(vals) = 1;

        %Calculate scores
        TP = length(find((gt_mask == 1) & (pred_mask == 1)));
        TN = length(find((gt_mask == 0) & (pred_mask == 0)));
        FP = length(find((gt_mask == 0) & (pred_mask == 1)));
        FN = length(find((gt_mask == 1) & (pred_mask == 0)));
        precision_val = TP / (TP + FP);
        recall_val = TP / (TP + FN);
        acc(i+1) = (TP + TN)/ (TP + TN + FP + FN); 
        precision(i+1) = precision_val;
        recall(i+1) = recall_val;
        f1_score(i+1) = 2 * (precision_val * recall_val) / (recall_val + precision_val);
    end
end
