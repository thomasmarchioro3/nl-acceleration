clc, close, clear;

% params
tau = 1e-6;
mu_min = 1e-4;
mu_max = 100;

k = [3 5 10 30];
iters = 1e3;
%save_tex = false;
save_tex = true;

% load objective function
[f,grad_f,f_best,n,m,sigma,L] = loader(tau);

x0 = zeros(m,1);

eta = L/sigma;

alpha = 2/(sigma+L);
%alpha = 1/L;
[x_list0, f_list0] = gradient_method(f,grad_f,alpha,x0,iters);

for i = 1:length(k)
    [x_list{i},f_list{i},mu_list{i}] = rna_k(f,x_list0,mu_min,mu_max,k(i));
end

%% PLOT THE RESULTS

figure(1);
set(0,'defaultTextInterpreter','latex') % to use LaTeX format
set(gcf, 'Position', [500, 300, 420, 320]);
semilogy(0:iters, f_list0-f_best,'^-', 'DisplayName', 'Gradient');
for i = 1:length(k)
    hold on;
    name = ['RNA' num2str(k(i))];
    semilogy(0:iters, f_list{i}-f_best,'^-', 'DisplayName', name);
end
title(['Comparison for $\tau=10^{' num2str(log10(tau)) '}$']);
hold off;
ylabel('$f(x_k)-f(x^*)$');
xlabel('Number of iterations $k$');
grid;
legend;

if(save_tex)
    addpath('src/');
    outfile = ['tex/comparison_tau1e' num2str(log10(tau)) '.tex'];
    matlab2tikz(outfile);
end

