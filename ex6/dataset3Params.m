function [C, sigma] = dataset3Params(X, y, Xval, yval,model)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 1;
C_list=[0.01 0.03 0.1 0.3 1 3 10 30];
sigma_list=[0.01 0.03 0.1 0.3 1 3 10 30];

s1=length(C_list);
s2=length(sigma_list);
m=length(yval);

err_val= 0;
results = zeros(s1*s2, 3);
error=ind=0;
predictions=zeros(m,1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

row=1;
for i=1:s1
for j=1:s2
C_val=C_list(i);
sigma_val=sigma_list(j);
model= svmTrain(X, y, C_val, @(x1, x2) gaussianKernel(x1, x2, sigma_val)); 
predictions = svmPredict(model,Xval);
err_val= mean(double(predictions~=yval));
results(row,:) = [C_val sigma_val err_val];
row=row+1;
endfor
endfor

[error ind]= min(results(:,3));
C=results(ind,1);
sigma=results(ind,2);

% =========================================================================

end
