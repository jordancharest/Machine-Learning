function [cost, gradient] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% Calculate the hypothesis for all training examples
hypothesis = X .* repmat(theta', [m 1]);
hypothesis =  sum(hypothesis, 2);

% Generate the logistic distribution
hypothesis = sigmoid(hypothesis);

% Calculate cost
cost = (1/m) * (-y'*log(hypothesis) - (1-y)' * log(1-hypothesis));


%% Gradient updates for fminunc() (find min of unconstrained function)
% calculate the cost (by subtracting y)
% then calculate the partial derivative for each feature in X (just
% multiply by X)
delta = repmat((sum(hypothesis, 2) - y), [1 size(X,2)]) .* X;

% Sum the partial derivatives
delta = (1/m) * sum(delta, 1);

% store the updates in a vector for fminunc() (NOTE: DOES NOT include alpha, as is the case with gradient descent)
gradient = delta';








% =============================================================

end
