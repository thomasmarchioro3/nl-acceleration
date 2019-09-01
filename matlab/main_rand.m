clc, close, clear;

seed = 1184445;
%seed = 1099362;
%seed = 91;
rng(seed);

% params
tau = 1e-6;
mu_min = 1e-4;
mu_max = 100;
k1 = 3;
k2 = 5;
iters = 100;

[f,grad_f,f_best,n,m,sigma,L] = loader(1e-6);

%x0 = zeros(m,1);
x0 = 1*(rand(m,1)-0.5);

alpha = 2/(sigma+L);
[x_list1, f_list1] = gradient_method(f,grad_f,alpha,x0,iters);

%alpha = 2/(sigma+L);
alpha = 1/L;
beta = (sqrt(L)-sqrt(sigma))/(sqrt(L)+sqrt(sigma));
[x_list2, f_list2] = nesterov_method(f,grad_f,alpha,beta,x0,iters);

[x_list3,f_list3,mu_list3] = rna_k(f,x_list1,mu_min,mu_max,k1);
[x_list4,f_list4,mu_list4] = rna_k(f,x_list1,mu_min,mu_max,k2);

%% PLOT THE RESULTS

figure(1);
set(0,'defaultTextInterpreter','latex') % to use LaTeX format
set(gcf, 'Position', [500, 300, 420, 320]);
semilogy(0:iters, f_list1-f_best,'^-', 'DisplayName', 'Gradient');
hold on;
semilogy(0:iters, f_list2-f_best,'^-', 'DisplayName', 'Nesterov');
hold on;
name = ['RNA' num2str(k1)];
semilogy(0:iters, f_list3-f_best,'^-', 'DisplayName', name);
hold on;
name = ['RNA' num2str(k2)];
semilogy(0:iters, f_list4-f_best,'^-', 'DisplayName', name);
title(['Results for $\tau=$' num2str(tau)]);
hold off;
ylabel('$f(x_k)-f(x^*)$');
xlabel('Number of iterations $k$');
grid;
legend;

%% PRODUCE TIKZ PLOTS

addpath('src/');
outfile = ['tex/plot_tau1e' num2str(log10(tau)) '_k' num2str(iters) '_r' num2str(seed) '.tex'];
matlab2tikz(outfile);
