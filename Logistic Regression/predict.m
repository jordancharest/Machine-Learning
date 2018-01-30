function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

% Generate the logistic distribution
hypothesis = sigmoid(X * theta);

% Make predictions for each probability
% >= 0.5 results in positive prediction
p = hypothesis >= 0.5;

end
