function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
disp(size(Theta1));
disp(size(Theta2));
% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

  a1 = [ones(m,1) X];

  a2 = sigmoid(Theta1 * a1');
  a2 = [ones(size(a2,2),1)';a2];
  h = sigmoid(Theta2 * a2);

  yVec = eye(size(y,1),size(Theta2,1))(y,:);

  J = - (1/m) * sum(sum(yVec .* log(h)' + (1 - yVec) .* log(1 - h)')) + (lambda/(2*m)) ...
                                                  * (sum(sum(Theta1(:,2:end).^2)) ...
                                                  + sum(sum(Theta2(:,2:end).^2)));
  
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
DELTA_one = 0;
DELTA_two = 0;
%UB = Unbiased
UB_a1 = a1(:,2:end);
UB_a2 = a2(2:end,:)';
UB_Theta1 = Theta1(:,2:end);
UB_Theta2 = Theta2(:,2:end);
disp(size(UB_Theta1));
disp(size(UB_Theta2));
disp("->");
disp(size(UB_a1));
disp(size(UB_a2));
disp("->");
for i = 1:m,
  UB_a1_i = UB_a1(i,:);%1 400
  UB_a2_i = UB_a2(i,:);%1 25
  a3_i = h'(i,:);%1 10
  yVec_i = yVec(i,:);%1 10
  
  delta3_i = (a3_i .- yVec_i)';%10 1
  
  g_2_i = UB_a2_i .* (1-UB_a2_i);%1 25
  delta2_i = (UB_Theta2' * delta3_i) .* g_2_i';% 25 10 * 10 1 .* 25 1-> 25 1
  disp("->");
  disp(size(delta2_i));
  disp(size(g_2_i));
  disp("->");
  DELTA_one = DELTA_one + delta2_i * UB_a1_i;%  + 25 1 * 1 400
  DELTA_two = DELTA_two + delta3_i * UB_a2_i;%  + 10 1 * 1 25
end;
Theta1_grad = DELTA_one/m + lambda .* UB_Theta1;
Theta2_grad = DELTA_two/m + lambda .* UB_Theta2;
disp(size(Theta1_grad));
disp(size(Theta2_grad));
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% -------------------------------------------------------------
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
