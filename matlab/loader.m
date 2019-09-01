function [f,grad_f,f_best,n,m,sigma,L] = loader(tau)

filename = 'dataset/sonar/sonar.csv';
data = readtable(filename);

X = table2array(data(:,1:end-1));
y = table2array(data(:,end));
y = [y{:}] ~= 'R';
%y = 1- y;

n = size(X,1);
m = size(X,2);

f = @(w) sum(log(1+exp(-diag(y)*X*w)))+0.5*tau*(w'*w);

h = @(w) exp(-diag(y)*X*w);
grad_f = @(w) -sum((eye(n)+diag(h(w))) \ (diag(h(w))*diag(y)*X) ) + tau*w';

sigma = tau;
L = 0.25*norm(X,'fro')^2+tau;

% f_best is found running the algorithms for 10000 iterations
if(tau == 1e1)
    f_best = 78.172065242606990;
elseif(tau == 1e-1)
    f_best = 67.667820745302360;
elseif(tau == 1e-3)
    f_best = 67.245844716666310;
else % tau == 1e-6
    f_best = 67.235303085186970;
end

end

