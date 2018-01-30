function [cost, gradient] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% Calculate the hypothesis for all training examples
% hypothesis = X .* repmat(theta', [m 1]);
% hypothesis =  sum(hypothesis, 2);

% Generate the logistic distribution
hypothesis = sigmoid(X * theta);

% Calculate cost
cost = (1/m) * (-y'*log(hypothesis) - (1-y)' * log(1-hypothesis));

% Add regularization term (Don't regularize theta_0 a.k.a. theta(1))
cost = cost + lambda/(2*m) * (sum(theta(2:end).^2));


%% Gradient updates for fminunc() (find min of unconstrained function)
% calculate the cost (by subtracting y)
% then calculate the partial derivative for each feature in X (just
% multiply by X)
gradient = (1/m) * (X' * (hypothesis - y));

% store theta_0 because we don't want to regularize it
temp = gradient(1);

% Add regularization term, remove regularization from theta_0
gradient = gradient + (lambda/m * theta);
gradient(1) = temp;

end
