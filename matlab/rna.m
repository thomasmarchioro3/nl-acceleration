function [x_hat,mu_star] = rna(f,x_list,mu_min,mu_max)
k = size(x_list, 1)-2;

log_mu = linspace(log(mu_min), log(mu_max), k+1);
mu = exp(log_mu);

x_ii = x_list(1:k+1, :);
x_ss = x_list(2:k+2, :);

R = x_ss-x_ii;
M = R*R';
%M = M/norm(M, 'fro');

x0 = x_list(1,:)';

x_star = x0;
f_star = f(x0);
i_star = 1;

xx = x_list(1:end-1,:);

for i = 1:k+1
    A = M+mu(i)*eye(size(M,1));
    b = ones(1,size(M,1))';
    c = A \ b;
    c = c/norm(c);
    x_extr = xx'*c;
    f_extr = f(x_extr);
    if(f_extr < f_star)
        f_star = f_extr;
        x_star = x_extr;
        i_star = i;
    end
end

%disp(num2str(i_star))
%disp(num2str(length(mu)))
if(length(mu)>1)
    mu_star = mu(i_star);
else
    mu_star = mu;
end


t = 1;
F = @(t) f(x0+t*(x_star-x0));
while(F(2*t) < F(t))
    t = 2*t;
end

x_hat = x0 + t*(x_star-x0);

end