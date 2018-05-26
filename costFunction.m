function [cost, grad] = costFunction(theta, X, y, lambda)
    % num of training
    m = size(X, 1);
    
    % regularize term for cost function
    reg = (lambda / (2 * m)) * (theta(2:end)' * theta(2:end));
    
    % cost
    cost = (1 / m) * sum((-y .* log(sigmoid(X * theta))) - ((1 - y) .* log(1 - sigmoid(X * theta)))) + reg;
    
    % gradient
    grad = (1 / m) * (X' * (sigmoid(X * theta) - y));
    grad(2:end) = grad(2:end) + ((lambda / m) * theta(2:end)); % applying reg from the second term
end