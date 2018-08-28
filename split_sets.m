% This function split a matrix vertically into two matrices by given percentage

function [set1, set2] = split_sets(orig_matrix, split_idx)

set1 = orig_matrix(:,split_idx+1:end);
set2 = orig_matrix(:,1:split_idx);

end

