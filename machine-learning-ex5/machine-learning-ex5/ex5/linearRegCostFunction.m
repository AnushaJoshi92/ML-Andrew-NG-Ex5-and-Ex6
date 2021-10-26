function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
h=0;

% You need to return the following variables correctly 
J = 0;
J1=J2=0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


h= X*theta;
J1=power(h-y,2);
J2=sum(J1);
J3=power(theta(2:end),2);
J4=sum(J3);
J=(1/(2*m)*J2)+((lambda/(2*m))*J4);

g1= transpose(X)*(h-y);

grad=(1/m)*g1;
theta(1)=0;
grad= grad+((lambda/m)*theta);


% =========================================================================

grad = grad(:);

end
