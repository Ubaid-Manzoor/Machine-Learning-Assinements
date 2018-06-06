load('ex3data1.mat');
[a b] = displayData(X,20);
theta = zeros(400,1);
lambda = 1;
[J,grad] = lrCostFunction(theta,X,y,lambda);