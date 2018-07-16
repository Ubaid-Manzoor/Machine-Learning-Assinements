function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0.1;
sigma = 0.03;

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
%C_Vec = zeros(1,5);
%sigma_Vec = zeros(1,5);
%C_Vec(1) = 0.1;
%sigma_Vec(1) = 0.03
%for i = 1:5
%  C_Vec(i+1) = C * 3^i;
%  sigma_Vec(i+1) = sigma * 3^i;
%end;
%disp(C_Vec);
%disp(sigma_Vec);
%error = zeros(5*5,3);
%index = 1;
%for i = 1:5
%  for j = 1:5
%     C = C_Vec(i);
%     sigma = sigma_Vec(j);
%     disp("->");
%     disp(C);
%     disp(sigma);
%     disp("->");
%     model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
%     predictions = svmPredict(model, Xval);
%     error(index,1) = mean(double(predictions ~= yval));
%     error(index,2) = C;
%     error(index,3) = sigma;
%     index += 1;
%  end;
%end;
%disp(error);
%Prediction_error = error(:,1);
%
%[a , i ] = min(Prediction_error);
%disp(a);
%C = error(i,2);
%sigma = error(i,3);
%disp(C);
%disp(sigma);
% =========================================================================
C = 0.3;
sigma = 0.09;
end
