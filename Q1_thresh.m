% Load Dataset
load Brain.mat
% Set seed for random number generator 
rng('default');

% Initialize arrays to hold scores
similarity_score = zeros(6,1,'double');
dice_score = zeros(6, 1, 'double');
ssim_array = zeros(6, 1, 'double');
acc_array = zeros(6, 1, 'double');
precision_array = zeros(6, 1, 'double');
recall_array = zeros(6, 1, 'double');
f1_array = zeros(6, 1, 'double');
mean_score = zeros(6, 1, 'double');

% Iterate over all images and labels
for j=1:10
    img = T1(:,:,j);    % Read current image
    lab = label(:,:,j); % Read current label
    
    % Display each mask from ground truth  
    figure(); colormap gray; axis equal; axis off;
    k=0;
    for i=1:2:12
        subplot(6, 2, i);
        imagesc(lab==k)
        k=k+1;
    end

    % Create dummy mask to hold predicted segmentation mask
    label_mask = zeros(size(lab));
    label_idx=0;

    % Threshold original image to form binary image
    thresh = multithresh(img, 1);
    L_outer = imquantize(img,thresh);

    % Extract only white component
    mask = zeros(size(lab));
    threshed_vals = L_outer == 2;
    mask(threshed_vals) = 1;

    % Detect connected components in the binary mask, and sort them by size
    [L, num] = bwlabel(mask, 8); 
    counts = sum(bsxfun(@eq,L(:),1:num)); 
    [vals, inds] = maxk(counts, 2); % Find two biggest contours
    ind1 = inds(1); % Index corresponding to biggest contour
    ind2 = inds(2); % Index corresponding to second biggest contour

    % Create seperate masks for largest components 
    outer_ring_mask = zeros(size(L_outer));
    inner_ring_mask = zeros(size(L_outer));
    outer_vals = L== ind2;
    inner_vals = L == ind1;
    outer_ring_mask(outer_vals) = 1;
    inner_ring_mask(inner_vals) = 1;
    % Create background mask by filling outer ring
    bg_mask = imcomplement(imfill(outer_ring_mask, 'holes'));
    % Fill inner mask to close holes
    filled_inner_ring_mask = imfill(inner_ring_mask, 'holes');
    
    % Create background mask, and update final predicted segmentation mask
    subplot(6,2,2);
    label_vals = bg_mask == 1;
    label_mask(label_vals)=label_idx;
    label_idx=label_idx+1;
    imagesc(bg_mask);
  
    % Create an image mask containing only outer ring
    outer_img_mask = img;
    outer_vals = filled_inner_ring_mask == 1;
    outer_img_mask(outer_vals) = 0;
    % Threshold and quantize the image mask into 2 classes
    thresh = multithresh(outer_img_mask, 2);
    L = imquantize(outer_img_mask, thresh);  
    
    %Extract the mask of Class 1 from the segmented mask, and update 
    % final segmented label mask
    outer_ring_mask = zeros((size(lab)));
    outer_ring_vals = L == 1;
    outer_ring_mask(outer_ring_vals) = 1;
    outer_ring_mask = imcomplement(outer_ring_mask);
    outer_ring_label_vals = outer_ring_mask == 1;
    label_mask(outer_ring_label_vals)=label_idx;
    label_idx = label_idx+1;
    subplot(6,2,4);
    imagesc(outer_ring_mask);
    
    %Extract the mask of Class 2 from the segmented mask, and update 
    % final segmented label mask
    inner_ring_mask = zeros((size(lab)));
    inner_ring_vals = L == 2;
    inner_ring_mask(outer_ring_vals) = 1;
    bg_vals = bg_mask == 1;
    inner_ring_mask(bg_vals) = 0;
    inner_vals = filled_inner_ring_mask ==1;
    inner_ring_mask(inner_vals) = 0;
    inner_mask_vals = inner_ring_mask == 1;
    label_mask(inner_mask_vals)=label_idx;
    label_idx = label_idx+1;
    subplot(6,2,6);
    imagesc(L==2);

    % Create an image mask containing only inner components
    se = strel('disk', 5);
    inner_img_mask = img;
    inner_vals = imerode(filled_inner_ring_mask == 0, se);
    inner_img_mask(inner_vals) = 0;
    % Perform Multi Otsu Thresholding to segment inner image mask
    thresh = multithresh(inner_img_mask, 3);
    L_inner = imquantize(inner_img_mask, thresh);   

    % Create mask for class 3
    lab4_mask = zeros((size(lab)));
    lab4_vals = L_inner == 2;
    lab4_mask(lab4_vals) = 1;
    label_mask(lab4_vals) = label_idx;
    label_idx = label_idx+1;
    subplot(6,2,8);
    imagesc(lab4_mask);
    
    % Create mask for class 4
    lab5_mask = zeros((size(lab)));
    lab5_vals = L_inner == 3;
    lab5_mask(lab5_vals) = 1;
    label_mask(lab5_vals) = label_idx;
    label_idx = label_idx+1;
    subplot(6,2,10);
    imagesc(lab5_mask);
    
    % Create mask for class 5
    lab6_mask = zeros((size(lab)));
    lab6_vals = L_inner == 4;
    lab6_mask(lab6_vals) = 1;
    lab6_vals = lab6_mask == 1;
    label_mask(lab6_vals) = label_idx;
    label_idx = label_idx+1;
    subplot(6,2,12);
    imagesc(lab6_mask);

    % Calculate metrics to compare ground truth and segmentation mask
    similarity = jaccard(categorical(lab), categorical(label_mask));
    similarity_score = similarity_score + similarity;
    dice_val = dice(categorical(lab), categorical(label_mask));
    dice_score = dice_score + dice_val;
    ssim_score = ssim_scores(lab, label_mask);
    ssim_array = ssim_array + ssim_score;
    [precision, recall, f1_score, acc] = pr(lab, label_mask);
    precision_array = precision_array + precision;
    recall_array = recall_array + recall;
    f1_array = f1_array + f1_score;
    acc_array = acc_array + acc;
end

% Compute mean score
similarity_score = similarity_score / 10;
dice_score = dice_score / 10;
ssim_array = ssim_array / 10;
precision_array = precision_array / 10;
recall_array = recall_array / 10;
f1_array = f1_array / 10;
acc_array = acc_array / 10;
mean_score = (similarity_score + f1_array + ssim_array + acc_array) / 4;
meanval = mean(mean_score);

% Plot final ground truth and segmented result
figure();colormap gray; axis equal; axis off;
subplot(1,2,1);
imagesc(lab);
subplot(1,2,2);
imagesc(label_mask);
