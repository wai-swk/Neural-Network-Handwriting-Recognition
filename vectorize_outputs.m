function [vectorized] = vectorize_outputs(digits, output)

[r,c] = size(output);
vectorized = zeros(digits,c);

for idx = 1:c
    digit = output(:,idx);
    vectorized(digit+1,idx) = 1;
end
end

