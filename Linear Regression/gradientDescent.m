function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % calculate the expected value
    hypothesis = X .* repmat(theta', [m 1]);
    
    % calculate the cost (by subtracting y)
    % then calculate the partial derivative for each feature in X (just
    % multiply by X)
    delta = repmat((sum(hypothesis, 2) - y), [1 size(X,2)]) .* X;
    
    % sum the partial derivatives
    delta = (1/m) * sum(delta, 1);
    
    % Update theta
    theta = theta - alpha*delta';
    
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
