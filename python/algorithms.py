import numpy as np

def gradient_method(f, grad_f, alpha, x0, iters=1e2):
    x_list = []
    f_list = []
    
    x_next = x0
    x_list.append(x_next.tolist())
    f_list.append(f(x_next))
    
    for k in range(int(iters)):
        x_prev = x_next
        x_next = x_prev - alpha*grad_f(x_prev)
        x_list.append(x_next.tolist())
        f_list.append(f(x_next))
    return x_list, f_list

def nesterov_method(f, grad_f, alpha, beta, x0, iters=1e2):
    x_list = []
    f_list = []
    
    x_next = x0
    y_next = x0
    x_list.append(x_next.tolist())
    f_list.append(f(x_next))
    
    for k in range(int(iters)):
        x_prev = x_next
        y_prev = y_next
        x_next = y_prev - alpha*grad_f(y_prev)
        y_next = x_next + beta*(x_next-x_prev)
        
        x_list.append(x_next.tolist())
        f_list.append(f(x_next))
    return x_list, f_list

def rna(x_list, f, mu_min, mu_max):
    x_list = np.transpose(x_list)
    #print(x_list.shape)
    k = x_list.shape[1] - 2
    #print("Value of k: {0}".format(k))
    log_step = np.log(mu_max/mu_min)/(max(1,k+1))
    log_mu = np.arange(np.log(mu_min), np.log(mu_max), log_step)
    mu = np.exp(log_mu)
    #print(mu.shape)
    
    # Compute the residue matrix
    x_ii = x_list[:, 1:k+1] # samples from x_1 rto x_k
    x_ss = x_list[:,2:k+2] # samples from x_2 to x_(k+1)
    R = np.matrix(x_ss-x_ii)
    M = np.matmul(np.transpose(R), R) # M is symmetric
    #print(M.shape)
    
    # Find x_star
    x0 = np.array(x_list[:,0])
    #print(x0)
    f_star = f(x0)
    x_star = x0
    i_star = 0
    
    for i in range(k+1):
        a = M+mu[i]*np.eye(M.shape[0]) # a is symmetric
        b = np.ones(M.shape[0])
        c = np.linalg.solve(a, b)
        c = c/np.linalg.norm(c)
        x_extr = np.matmul(c, np.transpose(x_list[:,0:k]))
        x_extr = np.array(x_extr)

        f_extr = f(x_extr)
        if(f_extr < f_star):
            f_star = f_extr
            x_star = x_extr
            i_star = i
    
    #print(mu[i_star])
    
    # find the best value of t that minimizes f(x0 + t*(x_star-x0))
    t = 1
    F = lambda t : f(x0 + t*(x_star-x0))
    while(F(2*t) < F(t)):
        t = 2*t

    x_hat = x0 + t*(x_star-x0)
    
    return x_hat

def rna_k(x_list, f, mu_min, mu_max, k=5):
    x_list_new = []
    f_list_new = []
    
    jmax = len(x_list)
    #print("jmax: {0}".format(jmax))
    for j in range(jmax):
        index_min = max(0, j-k)
        index_max = j+1
        x_list_k = x_list[index_min:index_max]
        #print(len(x_list_k))
        x_hat = rna(x_list_k, f, mu_min, mu_max)
        x_list_new.append(x_hat.tolist())
        f_list_new.append(f(x_hat))
        
    return x_list_new, f_list_new