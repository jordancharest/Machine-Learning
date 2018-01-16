function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

% Vector of predictions
p = zeros(m, 1);

% Calculate the hypothesis for all training examples
hypothesis = X .* repmat(theta', [m 1]);
hypothesis =  sum(hypothesis, 2);

% Generate the logistic distribution
hypothesis = sigmoid(hypothesis);


% Make predictions for each probability
% >= 0.5 results in positive prediction
for i = 1:m
    if hypothesis(i) >= 0.5
        p(i) = 1;
    else
        p(i) = 0;
    end
end







% =========================================================================


end
