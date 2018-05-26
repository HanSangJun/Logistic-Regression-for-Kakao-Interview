function [pred, acc] = pred_accuracy(X, theta, y)
    [m, n] = size(X);
    pred = zeros(m, 1); acc = 0;
    pred(find(sigmoid(X * theta) <= 0.5)) = 0;
    pred(find(sigmoid(X * theta) > 0.5)) = 1;

    if nargin == 3 % if num of argument is 3, return acc else only predicted value (acc = 0)
        acc = (size(find((pred == y) == 1), 1) / m) * 100;
    end
end