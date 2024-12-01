import numpy as np
import torch
import json
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.distributions.bernoulli import Bernoulli
import matplotlib.pyplot as plt
from causalml.inference.meta import BaseDRRegressor, BaseTRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from utils import *


def strategicGD(data_list, data_eval, dim_v, card_u, H, len_seq, num_epoch, seed, lr_pi = 0.01, lr_prob = 0.01, cost_coef = 0.1):
    print("Strategic PG")
    torch.set_default_dtype(torch.float32) 
    torch.set_default_device("cuda")
    
    np.random.seed(seed) #     
    torch.cuda.manual_seed_all(seed)    
    torch.manual_seed(seed)  

    cate_learner = BaseTRegressor(learner=XGBRegressor(random_state = 0))    
    (u_list, v_list, p_list, loan_list) = data_list
    (u_eval, v_eval, p_eval, loan_eval) = data_eval

    # define the network
    num_hiddens_pi = 2 * (dim_v + card_u)
    num_hiddens_prob = 2 * (dim_v + card_u)
    model_pi = allocation_policy(dim_v + card_u, num_hiddens_pi)
    optimizer_pi = optim.Adam(model_pi.parameters(), lr=lr_pi)       
    model_prob = shifted_distribution(card_u, num_hiddens_prob, card_u)
        
    # early stop for behavior model
    patience = 3

    seq_revenue = []
    seq_loss = []
    seq_val = []
    seq_inputs = []
    seq_category = []
    seq_x = []
    seq_z = []
    seq_y = []
    seq_pi = []
    
    for t, (u, v, p, loan) in enumerate(zip(u_list, v_list, p_list, loan_list)):   
          
        if t < H - 1:
            inputs, u_category, u_shift = manipulation(u, v, card_u, model_pi, c=cost_coef)            
            inputs = inputs[:,-card_u:]
            seq_inputs.append(inputs.detach())
            seq_category.append(u_category)
            if len(seq_inputs) > len_seq:
                seq_inputs.pop(0)
                seq_category.pop(0)

            x = torch.cat((u_shift, v), dim=1)
            pi = model_pi(x)
            
            z = torch.bernoulli(pi.detach())
            y = y_generate(u_category, p, loan, z)
            
            seq_x.append(x)
            seq_z.append(z)
            seq_y.append(y)
            seq_pi.append(pi)

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

        # warmup ends
        elif t == H - 1:
            inputs, u_category, u_shift = manipulation(u, v, card_u, model_pi, c=cost_coef)
            inputs = inputs[:,-card_u:] 
            seq_inputs.append(inputs.detach())
            seq_category.append(u_category)

            x = torch.cat((u_shift, v), dim=1)
            pi = model_pi(x)
            torch.manual_seed(seed)        
            z = torch.bernoulli(pi.detach())
            y = y_generate(u_category, p, loan, z)

            seq_x.append(x)
            seq_z.append(z)
            seq_y.append(y)
            seq_pi.append(pi)

            cate_fit(seq_x, seq_y, seq_z, seq_pi, cate_learner)
            tau = cate_predict(x, pi, cate_learner).detach()
            B = (tau.sum() - tau) / (tau.shape[0] - 1)

            # minus baseline
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

            # training of the shifted distribution
            stacked_inputs = torch.cat(seq_inputs, dim=0)
            stacked_category = torch.cat(seq_category, dim=0)
            random_indices = torch.randperm(stacked_inputs.shape[0])
            stacked_inputs = stacked_inputs[random_indices]
            stacked_category = stacked_category[random_indices]
            n_sample = int(stacked_inputs.shape[0] * 0.8)
            train_inputs = stacked_inputs[0:n_sample, :]
            train_category = stacked_category[0:n_sample, :]
            val_inputs = stacked_inputs[n_sample:, :]
            val_category = stacked_category[n_sample:, :]
            early_stop_counter = 0
            best_val_loss = float('inf')

            loss_function = nn.CrossEntropyLoss()
            optimizer_prob = optim.SGD(model_prob.parameters(), lr=lr_prob)            
            
            for _ in range(num_epoch):
                loss = loss_function(model_prob(train_inputs), train_category.view(-1))
                optimizer_prob.zero_grad()
                loss.backward()
                optimizer_prob.step()
                seq_loss.append(loss.item())
                loss_val = loss_function(model_prob(val_inputs), val_category.view(-1))
                seq_val.append(loss_val.item())
                if loss_val < best_val_loss:
                    best_val_loss = loss_val
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                if early_stop_counter >= patience:
                    break
                                 
            optimizer_pi = optim.Adagrad(model_pi.parameters(), lr=lr_pi)
            print("warmup end \n \n \n")

        else:
            n = u.shape[0]
            inputs, u_category, u_shift = manipulation(u, v, card_u, model_pi, c=cost_coef)            
            inputs = inputs[:,-card_u:]
            seq_inputs.append(inputs.detach())
            seq_category.append(u_category)
            if len(seq_inputs) > len_seq:
                seq_inputs.pop(0)
                seq_category.pop(0)

            x = torch.cat((u_shift, v), dim=1)
            pi = model_pi(x)
            torch.manual_seed(seed)        
            z = torch.bernoulli(pi.detach())
            y = y_generate(u_category, p, loan, z)
            seq_x.append(x)
            seq_z.append(z)
            seq_y.append(y)
            seq_pi.append(pi)

            tau = cate_predict(x, pi, cate_learner).detach()
            log_prob = nn.LogSoftmax(dim=1)(model_prob(inputs))[torch.arange(n), u_category.view(-1)]
            revenue = - torch.mean(pi * tau) - torch.mean(pi.detach() * tau * log_prob) 
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


    # plot the result
    plt.figure(figsize = (16,6))
    plt.subplot(1, 2, 1)
    plt.plot(seq_revenue)

    plt.subplot(1, 2, 2)
    plt.plot(seq_loss, label='train loss')
    plt.plot(seq_val, label='val loss')
    plt.legend()
    plt.savefig("curves/spg_curve_nov_cost{}_seed{}.pdf".format(cost_coef, seed))
    plt.show()
    
    # if seed == 10: # save an instance    
    torch.save(model_pi, "models/stra_model_nov_pi_c{}.pkl".format(cost_coef))
    torch.save(model_prob, "models/stra_model_nov_prob_c{}.pkl".format(cost_coef))        
    torch.save(seq_revenue, "value/spg_policy_value_nov_cost{}_seed{}.pkl".format(cost_coef, seed))
    
    
    return seq_revenue