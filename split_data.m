function [X_train, y_train, X_val, y_val, X_test, y_test] = split_data(X, y)
    [m, n] = size(X); % update size of X
    shuffled_idx = randperm(m);
    
    % (train, val, test) = (60%, 20%, 20%)
    train_num = floor(m * 0.6);
    val_num = floor(m * 0.2);
    
    % trainining set
    X_train = X(shuffled_idx(1:train_num), :);
    y_train = y(shuffled_idx(1:train_num));
    
    % validation set
    X_val = X(shuffled_idx(train_num+1:train_num+val_num+1), :);
    y_val = y(shuffled_idx(train_num+1:train_num+val_num+1));
    
    % test set
    X_test = X(shuffled_idx(train_num+val_num+2:end), :);
    y_test = y(shuffled_idx(train_num+val_num+2:end));
end