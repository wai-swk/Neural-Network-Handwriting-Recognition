% vector representation of each digit
zero = [-1 1 1 1 1 -1      1 -1 -1 -1 -1 1    1 -1 -1 -1 -1 1     1 -1 -1 -1 -1 1    -1 1 1 1 1 -1]';
one = [-1 -1 -1 -1 -1 -1  1 -1 -1 -1 -1 -1   1 1 1 1 1 1       -1 -1 -1 -1 -1 -1   -1 -1 -1 -1 -1 -1]';
two = [1 -1 -1 -1 -1 -1   1 -1 -1 1 1 1      1 -1 -1 1 -1 1    -1 1 1 -1 -1 1      -1 -1 -1 -1 -1 1]';
P = [zero one two];

% target output
t0 = [1 0 0]';
t1 = [0 1 0]';
t2 = [0 0 1]';
T = [t0 t1 t2];

learning_rate = 0.1;
iterations = 500;
neurons = 4;    % number of neurons used
[pr,pc] = size(P);
[tr,tc] = size(T);

% initialize weights and bias with random number between -0.5 to 0.5
W1 = rand(neurons,pr) - 0.5;
b1 = rand(neurons,1) - 0.5;
W2 = rand(tr,neurons) - 0.5;
b2 = rand(tr,1) - 0.5;

[E, final_W1, final_b1, final_W2, final_b2] = training(iterations, P, T, W1, b1, W2, b2, learning_rate);

% plot MSE vs Training Iterations
plot(E);
title('MSE vs Training Iterations')
xlabel('Training Iterations') % x-axis label
ylabel('MSE') % y-axis label
set(gca,'fontsize',15)

% test trained network with noise pixels
repeat = 30;
e0 = test_network(P, T, 0, pc, repeat, final_W1, final_b1, final_W2, final_b2);
e4 = test_network(P, T, 4, pc, repeat, final_W1, final_b1, final_W2, final_b2);
e8 = test_network(P, T, 8, pc, repeat, final_W1, final_b1, final_W2, final_b2);
x = 0:4:8;
noise_E = [e0 e4 e8];

% plot Average MSE vs Numbers of Noise Pixels
bar(x,noise_E);
title('Average MSE vs Numbers of Noise Pixels')
xlabel('Numbers of Noise Pixels') % x-axis label
ylabel('Average MSE') % y-axis label
set(gca,'fontsize',15)

% training network
function [E, W1, b1, W2, b2] = training(iterations, P, T, W1, b1, W2, b2, learning_rate)
    E = ones(1, iterations);
    for it = 1:iterations
        sum_error = 0;
        for d = 1:3
            [error, W1, b1, W2, b2] = backpropagation(P(:,d), T(:,d), W1, b1, W2, b2, learning_rate);
            sum_error = sum_error + error;
        end
        E(it) = sum_error/3;
    end
end

% backpropagation algorithm
function [error, W1, b1, W2, b2] = backpropagation(p, t, W1, b1, W2, b2, learning_rate)
    a1 = prop_forward(p, W1, b1);
    a2 = prop_forward(a1, W2, b2);

    e = t - a2;
    error = e'*e;

    s2 = final_layer_sens(a2, t);
    s1 = hidden_layer_sens(a1, s2, W2);

    [W1, b1] = update_weights_biases(W1, b1, s1, p, learning_rate);
    [W2, b2] = update_weights_biases(W2, b2, s2, a1, learning_rate);
end

% propagate forward using logsig
function a_next = prop_forward(a, W, b)
    a_next = nndlogsig(W * a + b);
end

% find final layer sensitivity
function final_s = final_layer_sens(a, t)
    fdot = (1 - a).* a;
    Fdot = diag(fdot);
    e = t - a;
    
    final_s = -2 * Fdot * e;
end

% find hidden layer sensitivity
function hidden_s = hidden_layer_sens(a, s, W)
    fdot = (1 - a).* a;
    Fdot = diag(fdot);
    
    hidden_s = Fdot * W' * s;
end

% updates weights and bias
function [newW, newb] = update_weights_biases(oldW, oldb, s, a, learning_rate)
    newW = oldW - learning_rate * s * a';
    newb = oldb - learning_rate * s;
end

% test trained network with noise pixels
function E = test_network(orig_P, T, pixel_errors, digits, repeat, W1, b1, W2, b2)
    E = 0;
    for i = 1:repeat
        for d = 1:digits  
            newP = apply_noise(orig_P, pixel_errors, digits);
            input = newP(:,d);
            t = T(:,d);
            a1 = prop_forward(input, W1, b1);
            a2 = prop_forward(a1, W2, b2);
        
            e = t - a2;
            E = E + e'*e;
        end
    end
    E = E / (digits * repeat);
end

% This is the function that applies noise to input pattern matrix by
% randomly changing pixel of each digit
function noisyP = apply_noise(orig_p, pixels, digits)
    [r,c] = size(orig_p);   % get original input pattern dimension
    noisyP = orig_p;        % matrix that will be updated with noise
    
    for i = 1:digits
        flipped_matrix = zeros(r,c);    % matrix to keep track of flipped pixel
        flipped = 0;        % counter for flipped pixels
        while (flipped ~= pixels)
            rand_num = randi([1,r]);    % generate random number within the range of row
            
            % if element at rand_num hasn't been flipped yet
            if (flipped_matrix(rand_num,i)~= 1)
                noisyP(rand_num,i) = orig_p(rand_num,i)*-1; % update matrix with noise
                flipped = flipped + 1;
            end
            
            flipped_matrix(rand_num,i) = 1; % update flipped status
        end
    end
end

function a = nndlogsig(n)

% Copyright 1995-2015 Martin T. Hagan and Howard B. Demuth

a = 1 ./ (1 + exp(-n));
i = find(~isfinite(a));
a(i) = sign(n(i));
end

