% This function first appends one matrix to another, then randomize the
% columns of the combined matrix

function [random_matrix] = randomize_data(images,labels)

% Firstly, create a matrix which combines images and labels
combined_matrix = [images; labels];

% Randomize the combined_matrix
random_matrix = combined_matrix(:,randperm(end));

end