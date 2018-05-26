clc; clear; close all;

% data load
data = load('houses.txt');
X = data(:, [1, 2]); y = data(:, 3);
[m, n] = size(X);

% generate bias term and non-linear feaqures
[m, n] = size(X);
X = [ones(m, 1), X, X(:, 1).^2, X(:, 1).*X(:, 2), X(:, 2).^2, ... % square term
      X(:, 1).^3, X(:, 1).^2.*X(:, 2), X(:, 1).*X(:, 2).^2, X(:, 2).^3, ... % cubic term
        X(:, 1).^4 X(:, 1).*X(:, 2).^3 X(:, 1).^2.*X(:, 2).^2 X(:, 1).^3.*X(:, 2) X(:, 2).^4]; % four square term
[m, n] = size(X); % update size of X

% split dataset as train, val, and test
[X_train, y_train, X_val, y_val, X_test, y_test] = split_data(X, y);

% initialize theta
theta = randn(n, 1);

iter = 1000; % num of iteration
lambda = 0.05; % lambda for reg
step_size = 2.5; % step_size for update

tic; % training
for idx=1:iter
    [cost, grad] = costFunction(theta, X_train, y_train, lambda); % cost and gradient
    theta = theta - (step_size * grad); % update theta
    [pred_train, acc_train] = pred_accuracy(X_train, theta, y_train); % train accuracy
    [pred_val, acc_val] = pred_accuracy(X_val, theta, y_val); % val accuracy
    fprintf('%d iterations - cost: %2.4f, train_acc: %2.4f, val_acc: %2.4f\n', idx, round(cost, 5), round(acc_train, 5), round(acc_val, 5));
end; time = toc; fprintf('gradient decent takes time : %2.2f sec\n', time);

% test - if you know y, you can get prediced values and accuracy
[pred_test, acc_test] = pred_accuracy(X_test, theta, y_test);
fprintf('test_acc: %2.4f\n', round(acc_test, 5));

% test - if you don't know y, you can get only prediced values
% [pred_test] = pred_accuracy(X_test, theta);
% disp(pred_test);