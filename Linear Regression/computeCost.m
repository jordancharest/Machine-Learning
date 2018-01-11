function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y


% Initialize some useful values
m = length(y); % number of training examples

% Compute the sum of squares
hypothesis = X .* repmat(theta', [m 1]); 
SoS =  sum(hypothesis, 2) - y;
SoS = sum(SoS.^2);

% Cost function
J = 1/(2*m) * SoS

end
