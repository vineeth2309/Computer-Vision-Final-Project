function mtx=imfill3(mtx)
    %{
    Description:
        Function to fill scan along three dimensions
    Arguments:
        mtx: Input 3D array(W,H,N)
    Returns: 
        mtx: Output Matrix filled along 3 dimensions 
    %}
    for i=1:3   % Iterate over each dimension
        for j=1:size(mtx,3) % Iterate over each scan
            mtx(:,:,j)=imfill(mtx(:,:,j),'holes');
        end
    end
    for i=1:3   % Iterate over each dimension
        for j=1:size(mtx,3) % Iterate over each scan
            mtx(:,:,j)=imfill(mtx(:,:,j),'holes');
        end
    end
end
