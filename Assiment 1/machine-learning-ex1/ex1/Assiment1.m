data = load('ex1data1.txt');
X = data(:,1);
y = data(:,2);
X = [ones(length(X), 1),X]; 
theta = zeros(2, 1);
iterations = 1500;
alpha = 0.01;
[theta,J] = gradientDescent(X,y,theta,alpha,iterations);
plotData(X(:,2),y,theta);