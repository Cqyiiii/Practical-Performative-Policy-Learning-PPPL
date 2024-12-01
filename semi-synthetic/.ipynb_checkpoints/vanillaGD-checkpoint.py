import numpy as np
import torch
import torch.optim as optim
from torch.distributions.bernoulli import Bernoulli
from causalml.inference.meta import BaseDRRegressor, BaseTRegressor
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from function import allocation_policy, y_generate, manipulation, cate_generate, cate_fit, cate_predict


def vanillaGD(data_list, data_eval, dim_v, card_u, H_prime, f, seed, lr_pi=0.01, cost_coef = 0.1):
    print("Vanilla PG")
    torch.set_default_tensor_type(torch.cuda.FloatTensor)    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)    
    np.random.seed(seed)

    # cate_learner = BaseDRRegressor(learner=XGBRegressor(random_state = 0), treatment_effect_learner=LinearRegression())
    cate_learner = BaseTRegressor(learner=XGBRegressor(random_state = 0))        
    (u_list, v_list, p_list, loan_list) = data_list
    (u_eval, v_eval, p_eval, loan_eval) = data_eval

    num_hiddens_pi = 2 * (dim_v + card_u)
    model_pi = allocation_policy(dim_v + card_u, num_hiddens_pi)

    seq_revenue = []
    seq_x = []
    seq_z = []
    seq_y = []
    seq_pi = []
    optimizer_pi = optim.Adam(model_pi.parameters(), lr=lr_pi)
    
    
    for t, (u, v, p, loan) in enumerate(zip(u_list, v_list, p_list, loan_list)):
        inputs, u_category, u_shift = manipulation(u, v, card_u, model_pi, c=cost_coef)

        x = torch.cat((u_shift, v), dim=1)
        pi = model_pi(x)
        
        torch.manual_seed(seed)
        z = torch.bernoulli(pi.detach())        
        y = y_generate(u_category, p, loan, z)

        # print(torch.mean(pi))                    
        # print(torch.mean(z))        

        seq_x.append(x)
        seq_z.append(z)
        seq_y.append(y)
        seq_pi.append(pi)

        if t < H_prime:
            cate_fit(seq_x, seq_y, seq_z, seq_pi, cate_learner)

        tau = cate_predict(x, pi, cate_learner).detach()
        B = (tau.sum() - tau) / (tau.shape[0] - 1)

        revenue = - torch.mean(pi * (tau - B))
        optimizer_pi.zero_grad()
        revenue.backward()
        optimizer_pi.step()

        # evaluation of allocation policy
        _, u_eval_category, u_eval_shift = manipulation(u_eval, v_eval, card_u, model_pi, c=cost_coef)
        x_eval = torch.cat((u_eval_shift, v_eval), dim=1)
        pi_eval = model_pi(x_eval)
        tau_eval = cate_generate(u_eval_category.view(-1, 1), p_eval, loan_eval)
        value = torch.mean(pi_eval * tau_eval)

        seq_revenue.append(value.item())
        print(value.item())
        
        
        # if t == H_prime:
        #     optimizer_pi = optim.Adagrad(model_pi.parameters(), lr=lr_pi)            
    
    plt.figure(figsize=(8,6))
    plt.plot(seq_revenue)
    plt.savefig("curves/pg_curve_cost{}_seed{}.pdf".format(cost_coef, seed))
    plt.show()
    
    # save an instance

    # torch.save(model_pi, "models/vani_model_pi_c{}.pkl".format(cost_coef))

    torch.save(seq_revenue, "value/vani_policy_value_cost{}_seed{}.pkl".format(cost_coef, seed))    
    return seq_revenue
