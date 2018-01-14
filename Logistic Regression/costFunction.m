function [cost, gradient] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
cost = 0;
gradient = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%

hypothesis = X .* repmat(theta', [m 1]);
hypothesis =  sum(hypothesis, 2);
hypothesis = sigmoid(hypothesis);
cost = (1/m) * (-y'*log(hypothesis) - (1-y)' * log(1-hypothesis));








% =============================================================

end
