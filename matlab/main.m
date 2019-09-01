clc, close, clear;

% params
tau = 1e-3;
mu_min = 1e-4;
mu_max = 100;
k1 = 3;
k2 = 5;
iters = 1e3;
%save_tex = false;
save_tex = true;

% load objective function
[f,grad_f,f_best,n,m,sigma,L] = loader(tau);

x0 = zeros(m,1);

eta = L/sigma;

alpha = 2/(sigma+L);
%alpha = 1/L;
[x_list1, f_list1] = gradient_method(f,grad_f,alpha,x0,iters);

%alpha = 2/(sigma+L);
alpha = 1/L;
beta = (sqrt(L)-sqrt(sigma))/(sqrt(L)+sqrt(sigma));
[x_list2, f_list2] = nesterov_method(f,grad_f,alpha,beta,x0,iters);

[x_list3,f_list3,mu_list3] = rna_k(f,x_list1,mu_min,mu_max,k1);
[x_list4,f_list4,mu_list4] = rna_k(f,x_list1,mu_min,mu_max,k2);

%f_best1 = min(f_list1);
%f_best2 = min(f_list2);
%f_best3 = min(f_list4);

%f_best = min([f_best1, f_best2, f_best3]);

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

%save('results/gradient_x.mat', 'x_list1')
%save('results/nesterov_x.mat', 'x_list2')

%% PRODUCE TIKZ PLOTS

if(save_tex)
    addpath('src/');
    outfile = ['tex/plot_tau1e' num2str(log10(tau)) '_k' num2str(iters) '_RNA' num2str(k1) '_' num2str(k2) '.tex'];
    matlab2tikz(outfile);
end

