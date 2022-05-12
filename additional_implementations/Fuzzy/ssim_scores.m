function score=ssim_scores(gt, label)
    %{
    Description:
        Function to calculate Structural Similarity score 
    Arguments:
        gt: Ground Truth Segmentation Mask
        label: Predicted Label Mask
    Returns: 
        ssim_scores: Classwise SSIM score
    %}
    % Create dummy arrays to hold scores
    score = zeros(6,1,'double');
    for i=0:5
        % Create ground truth binary mask
        gt_mask = zeros((size(gt)));
        gt_vals = gt == i;
        gt_mask(gt_vals) = 1;

        % Create prediction binary mask
        mask = zeros((size(gt)));
        vals = label == i;
        mask(vals) = 1;
        
        %Calculate scores
        [ssimval, ~] = ssim(gt_mask, mask);
        score(i+1) = score(i+1) + ssimval;
    end
end
