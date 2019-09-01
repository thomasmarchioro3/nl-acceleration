function [x_list, f_list] = gradient_method(f,grad_f, alpha,x0,iters)
k = iters;
m = length(x0);    
x_list = zeros(k+1,m);
f_list = zeros(k,1);
x_list(1,:) = x0;
f_list(1) = f(x0);

x_next = x0;

for i = 1:k
    x_prev = x_next;
    x_next = x_prev - alpha*grad_f(x_prev)';
    %g = norm(grad_f(x_prev));
    %disp(num2str(g));
    x_list(i+1,:) = x_next;
    f_list(i+1) = f(x_next);
end

end