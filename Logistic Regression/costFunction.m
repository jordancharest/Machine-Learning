function [cost, gradient] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% Generate the logistic distribution
hypothesis = sigmoid(X * theta);

% Calculate cost
cost = (1/m) * (-y'*log(hypothesis) - (1-y)' * log(1-hypothesis));


%% Gradient updates for fminunc() (find min of unconstrained function)
% calculate the cost (by subtracting y)
% then calculate the partial derivative for each feature in X (just
% multiply by X)
gradient = (1/m) * (X' * (hypothesis - y));

end
