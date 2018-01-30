function prediction = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);

% Layer 1
activations1 = [ones(m,1), X];
hypothesis2 = activations1 * Theta1';

% Layer 2
activations2 = [ones(size(hypothesis2,1), 1), sigmoid(hypothesis2)];
hypothesis3 = activations2 * Theta2';

% Output Layer
output_activation = sigmoid(hypothesis3);

[max_activation, prediction] = max(output_activation, [], 2);

end
