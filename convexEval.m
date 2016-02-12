function y = convexEval(alpha,beta,x)

% This function takes the alphas and betas exported by convexTree, fits a
% function and then evaluates it at x, returning y
[n,d] = size(x);

[y, iList] = max([ones(n,1),x]*[alpha; beta],[],2);