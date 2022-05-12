% Load Dataset
load Brain.mat
% Set seed for random number generator 
rng('default');

% Read Images and Labels
img = T1;
lab = label;

% Create dummy mask to hold final segmented labels
label_mask = zeros(size(lab));
label_idx = 0;

% Threshold original image to form binary image
thresh = multithresh(img, 1);
L_outer = imquantize(img, thresh);
% Extract only white portion
mask = zeros(size(L_outer));
threshed_vals = L_outer == 2;
mask(threshed_vals) = 1;
% Find connected components in 3D
L = bwlabeln(mask);

% Extract masks for two largest components
outer_ring_mask = zeros(size(L_outer));
inner_ring_mask = zeros(size(L_outer));
outer_vals = L== 1;
inner_vals = L == 3;
outer_ring_mask(outer_vals) = 1;
inner_ring_mask(inner_vals) = 1;

% Fill the masks to close holes in 3D
filled_outer_ring_mask = imfill3(outer_ring_mask);
filled_inner_ring_mask = imfill3(inner_ring_mask);

% Extract background mask
bg_mask = zeros((size(lab)));
bg_vals = filled_outer_ring_mask == 0;
bg_mask(bg_vals) = 1;
label_mask(bg_vals) = label_idx;
label_idx = label_idx+1;

% Create an 3D mask containing only outer ring
outer_img_mask = img;
outer_vals = filled_inner_ring_mask == 1;
outer_img_mask(outer_vals) = 0;
% Segment the 3D mask into 3 classes
L = imsegkmeans3(outer_img_mask, 3);

%Extract the mask of Class 1 from the segmented mask, and update 
% final segmented label mask
t_outer_mask = L==1 | imcomplement(bg_mask);
outer_ring_mask = zeros((size(lab)));
outer_ring_vals = L == 1;
outer_ring_mask(outer_ring_vals) = 1;
outer_ring_mask = imcomplement(outer_ring_mask);
outer_ring_label_vals = outer_ring_mask == 1;
label_mask(outer_ring_label_vals)=label_idx;
label_idx = label_idx+1;

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

% Create a 3D mask containing only inner components
se = strel('disk', 5);
inner_img_mask = img;
inner_vals = imdilate(filled_inner_ring_mask, se)==0;
inner_img_mask(inner_vals) = 0;
% Perform 3D K Means segmentation on inner matter mask
L_inner = imsegkmeans3(inner_img_mask, 4);

% Iterate through each of classes 3-5
for i=3:5
    % Create ground truth mask for ith class
    gt_mask = zeros((size(lab)));
    gt_vals = lab == i;
    gt_mask(gt_vals) = 1;
    for j=1:4   % Iterate through each k means segmentation class
        % Extract mask corresponding to jth k means class
        mask = zeros((size(lab)));
        vals = L_inner == j;
        mask(vals) = 1;
        % Compute SSIM between GT mask and predicted k means mask
        [ssimval,ssimmap] = ssim(gt_mask, mask); 
        if ssimval>0.8  % Check if match is greater than 80%
            % If match is greater than 80%, then create predicted mask
            label_mask(vals) = label_idx;
            label_idx = label_idx+1;
            break;
        end
    end
end

% Calculate metrics to compare ground truth and segmentation mask
similarity = jaccard(categorical(lab), categorical(label_mask));
dice_score = dice(categorical(lab), categorical(label_mask));
ssim_score = ssim_scores(lab, label_mask);
[precision, recall, f1_score, acc] = pr(lab, label_mask);
mean_score = (similarity + f1_score + ssim_score + acc) / 4;
mean_val = mean(mean_score);

% Show final ground truth and segmented result
figure();
volshow(label_mask);
figure();
volshow(lab);

