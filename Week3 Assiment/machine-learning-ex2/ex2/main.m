data = load('ex2data1.txt');
X = data(:,1:2);
y = data(:,3);
%theta = zeros(3,1);

%plotData(X,y);

x = [ones(length(X), 1) X];
 
%[J,Gradient] = costFunction(theta,x,y);

options = optimset('GradObj', 'on', 'MaxIter', 400);

initialtheta = zeros(3,1);

[theta, cost,exitflag] = fminunc(@(t)(costFunction(t, x, y)), initialtheta, options);

plotDecisionBoundary(theta,x,y);
%disp(theta);
%disp(cost);
%disp(exitflag);
