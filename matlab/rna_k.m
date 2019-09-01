function [x_list_new,f_list_new,mu_list] = rna_k(f,x_list,mu_min,mu_max,k)
j_max = size(x_list,1)-1;

iters = j_max-k+1;
x_list_new = zeros(iters+1, size(x_list,2));
f_list_new = zeros(j_max+1,1);
mu_list = zeros(j_max+1,1);
x0 = x_list(1,:)';
x_list_new(1,:) = x0;
f_list_new(1) = f(x0);


for j = 1:j_max
    i_min = max(1, j-k+1);
    i_max = j+1;
    x_list_k = x_list(i_min:i_max,:);
    %disp(num2str(size(x_list_k)))
    [x_hat,mu_star] = rna(f,x_list_k,mu_min,mu_max);
    x_list_new(j+1,:) = x_hat;
    f_list_new(j+1) = f(x_hat);
    mu_list(j+1) = mu_star;
end

mu_list(1) = mu_list(2);

end