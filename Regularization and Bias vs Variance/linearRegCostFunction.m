function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%X = [ones(m, 1), X]; % Add a column of ones to x
% the 1 column is already added at the call of this function
%size(X)

predictions= X*theta;
sqrErrors= (predictions-y).^2;
temp=theta(2:end);
J=1/(2*m) * sum(sqrErrors) + lambda/(2*m) * sum(temp.^2);

%size(theta)

temp=theta;
temp(1)=0;

grad = 1/m * sum((predictions-y)' * X, 1);% + lambda/m * temp;
grad=grad' + lambda/m * temp;
%size(grad)
%size(lambda/m * temp)




% =========================================================================

grad = grad(:);

end
