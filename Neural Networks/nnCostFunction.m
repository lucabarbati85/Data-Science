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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

ym=zeros(m,num_labels);
for i=1:m,
    ym(i,y(i))=1;
end;
%size(ym);
%size(Theta1)


X = [ones(m, 1) X];
a2=sigmoid(X*Theta1');
a2 = [ones(size(a2,1), 1) a2];
a3=sigmoid(a2*Theta2');
%size(a3);

a=0;
for i=1:m,
    for j=1:num_labels,
        a=a+(-ym(i,j)*log(a3(i,j)) - (1-ym(i,j))*log(1-a3(i,j)));
           
    end;
end;

J=a/m;
%size(J);

a=0;
    for i=1:size(Theta1,1),
        for j=2:size(Theta1,2),
           a=a+(Theta1(i,j))^2;
        end;
    end;
b=0;
for i=1:size(Theta2,1),
    for j=2:size(Theta2,2),
        b=b+(Theta2(i,j))^2;
    end;
end;
c=a+b;
J=J+(lambda/(2*m))*c;

D1 = zeros(size(Theta1));
D2 = zeros(size(Theta2));

a1=X;
%size(a1)
           
z2=(a1*Theta1');
%size(z2)
a2=sigmoid(z2);
%size(a2)
a2 = [ones(size(a2,1), 1) a2];
%size(a2)

z3=(a2*Theta2');
%size(z3)
a3=sigmoid(z3);
%size(a3)
    
%size(ym(t,:))

delta3=a3-ym;
%size(delta3)
    
delta2=Theta2'*delta3';
%size(delta2)
delta2=delta2(2:end,:);
%size(delta2)
delta2=delta2' .* sigmoidGradient(z2);
%size(delta2)

D2 = D2 + delta3'*a2;
D1 = D1 + delta2'*a1;



%for t=1:m,
    

%a1=X(t,:);
%size(a1);
     
%z2=(a1*Theta1');
%z2 = [ones(size(z2,1), 1) z2];
%size(z2);
%a2=sigmoid(z2);
%size(a2)
%a2 = [ones(size(a2,1), 1) a2];
%size(a2)

%z3=(a2*Theta2');
%size(z3)
%a3=sigmoid(z3);
%size(a3)

%a = 1:num_labels; % a is a temp para
%Y = (a == y(t)); % making Y matrix as classification label

%delta3 = a3 - ym(t,:);

%delta2=Theta2'*delta3';
%delta2=delta2(2:end);
%delta2=delta2' .* sigmoidGradient(z2);

    
%D2 = D2 + delta3'*a2;
%D1 = D1 + delta2'*a1;

%end;

Theta1_grad=(1/m)*D1 + lambda/m*[zeros(size(Theta1,1),1) Theta1(:, 2:end)];
Theta2_grad=(1/m)*D2 + lambda/m*[zeros(size(Theta2,1),1) Theta2(:, 2:end)];


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
