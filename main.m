% Change the filenames if you've saved the files under different names
% On some platforms, the files might be saved as 
% train-images.idx3-ubyte / train-labels.idx1-ubyte
train_images = loadMNISTImages('train-images-idx3-ubyte');
train_labels = loadMNISTLabels('train-labels-idx1-ubyte');

test_images = loadMNISTImages('t10k-images-idx3-ubyte');
test_labels = loadMNISTLabels('t10k-labels-idx1-ubyte');

% Get training set and validation set (~15%)
rand_matrix = randomize_data(train_images, train_labels');
[training_set, validation_set] = split_sets(rand_matrix, 10000);

% Get training input and output
training_images = training_set(1:784,:);
training_labels = training_set(end,:);

% Get validation input and output
validation_images = validation_set(1:784,:);
validation_labels = validation_set(end,:);

% Target outputs in the training set
training_T = vectorize_outputs(10, training_labels);

% Target outputs in the validation set
validation_T = vectorize_outputs(10, validation_labels);

% Target outputs in the testing set
testing_T = vectorize_outputs(10, test_labels');

% Initialize values
learning_rate = 0.1;
iterations = 200;
neurons = 400;    % number of neurons used
[pr,pc] = size(training_images);
[tr,tc] = size(training_T);

% initialize weights and bias with random number between -0.5 to 0.5
W1 = rand(neurons,pr) - 0.5;
b1 = rand(neurons,1) - 0.5;
W2 = rand(tr,neurons) - 0.5;
b2 = rand(tr,1) - 0.5;


[it, E, Ve, final_W1, final_b1, final_W2, final_b2] = training(iterations, training_images, training_T, W1, b1, W2, b2, learning_rate, validation_images, validation_T, neurons);
[v,v_idx] = min(Ve(:,1:it-1)); % idx is the index where the smallest element is at


[Te] = test_network(test_images, testing_T, final_W1(:,:,v_idx), final_b1(:,:,v_idx), final_W2(:,:,v_idx), final_b2(:,:,v_idx));
[result] = testing_test_network(test_images, testing_T, final_W1(:,:,v_idx), final_b1(:,:,v_idx), final_W2(:,:,v_idx), final_b2(:,:,v_idx));

% plot MSE vs Training Iterations
axis auto
x = zeros(1,it-1);
for i = 1:it-1
    x(:,i) = i; 
end
plot(x,E(:,1:it-1),'--bo',x,Ve(:,1:it-1),'-r*')
title('MSE vs Training Iterations(500 Hidden Units)')
xlabel('Training Iterations') % x-axis label
ylabel('MSE') % y-axis label
legend('Training MSE','Validation MSE')
set(gca,'fontsize',15)

disp ("Accuracy on test set")
disp(result);
disp ("MSE on test set")
disp(Te);

% training network
function [it, E, Ve, track_W1, track_b1, track_W2, track_b2] = training(iterations, P, T, W1, b1, W2, b2, learning_rate, valid_p, valid_t, HU)
    E = zeros(1, iterations);
    Ve = zeros(1, iterations);
    track_W1 = zeros(HU,784);
    track_W2 = zeros(10,HU);
    track_b1 = zeros(HU,1);
    track_b2 = zeros(10,1);
    consecutive_up = false;
    it = 1;
    while consecutive_up == false && it <= iterations
        disp(it);
        track_W1(:,:,it) = W1;
        track_W2(:,:,it) = W2;
        track_b1(:,:,it) = b1;
        track_b2(:,:,it) = b2;
        sum_error = 0;
        for d = 1:50000
            [error, W1, b1, W2, b2] = backpropagation(P(:,d), T(:,d), W1, b1, W2, b2, learning_rate);
            sum_error = sum_error + error;
        end
        E(it) = sum_error/50000;
        Ve(it) =  test_network(valid_p, valid_t, W1, b1, W2, b2);
        if it >= 4
            if Ve(:,it) > Ve(:,it-1) && Ve(:,it-1) > Ve(:,it-2) && Ve(:,it-2) > Ve(:,it-3)
                consecutive_up = true;
            end
        end
        it = it + 1;
        
    
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


function E = test_network(input, output, W1, b1, W2, b2)
E = 0;
[r,c] = size(input);
for i = 1:c  
    a1 = prop_forward(input(:,i), W1, b1);
    a2 = prop_forward(a1, W2, b2);

    e = output(:,i) - a2;
    E = E + e'*e;
end
E = E / c;

end

function performance = testing_test_network(input, output, W1, b1, W2, b2)
matched = 0;
[r,c] = size(input);
for i = 1:c
    a1 = prop_forward(input(:,i), W1, b1);
    a2 = prop_forward(a1, W2, b2);
    
    [a2_val, a2_index] = max(a2);
    [t_val, t_index] = max(output(:,i));
    
    if a2_index == t_index
        matched = matched + 1;
    end
end
performance = matched / c;
end



function a = nndlogsig(n)

% Copyright 1995-2015 Martin T. Hagan and Howard B. Demuth

a = 1 ./ (1 + exp(-n));
i = find(~isfinite(a));
a(i) = sign(n(i));
end


