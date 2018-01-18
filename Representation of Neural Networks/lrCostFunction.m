function [cost, gradient] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

% Calculate the hypothesis for all training examples
hypothesis = X .* repmat(theta', [m 1]);
hypothesis =  sum(hypothesis, 2);

% Generate the logistic distribution
hypothesis = sigmoid(hypothesis);

% Calculate cost
cost = (1/m) * (-y'*log(hypothesis) - (1-y)' * log(1-hypothesis));

% Add regularization term (Don't regularize theta_0 a.k.a. theta(1))
cost = cost + lambda/(2*m) * (sum(theta(2:end).^2));


%% Gradient updates for fminunc() (find min of unconstrained function)
% calculate the cost (by subtracting y)
% then calculate the partial derivative for each feature in X (just
% multiply by X)
delta = repmat((sum(hypothesis, 2) - y), [1 size(X,2)]) .* X;

% Sum the partial derivatives
delta = (1/m) * sum(delta, 1);

% store the updates in a vector for fminunc() (NOTE: DOES NOT include alpha, as is the case with gradient descent)
gradient = delta';

% store theta_0 because we don't want to regularize it
temp = gradient(1);

% Add regularization term, remove regularization from theta_0
gradient = gradient + (lambda/m * theta);
gradient(1) = temp;


end
