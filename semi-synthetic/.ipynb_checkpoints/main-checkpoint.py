import numpy as np
import pandas as pd
import json
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns;sns.set()
import warnings

from cutoff import cutoff, cutoff_oracle
from vanillaGD import vanillaGD
from strategicGD import strategicGD
from end2endGD import end2endGD

# torch.use_deterministic_algorithms(True)

warnings.filterwarnings("ignore")

torch.set_default_tensor_type(torch.cuda.FloatTensor)
# torch.manual_seed(1)

df = pd.read_csv('features_with_pred_prob.csv')
df_loan = df['AMT_CREDIT'].copy()
df_prob = df['PROB_PRED'].copy()

df['AMT_INCOME_TOTAL'] = (df['AMT_INCOME_TOTAL'] - df['AMT_INCOME_TOTAL'].mean()) / df['AMT_INCOME_TOTAL'].std()
df['AMT_CREDIT'] = (df['AMT_CREDIT'] - df['AMT_CREDIT'].mean()) / df['AMT_CREDIT'].std()
df['REGION_POPULATION_RELATIVE'] = (df['REGION_POPULATION_RELATIVE'] - df['REGION_POPULATION_RELATIVE'].mean()) / df['REGION_POPULATION_RELATIVE'].std()
df['DAYS_BIRTH'] = (df['DAYS_BIRTH'] - df['DAYS_BIRTH'].mean()) / df['DAYS_BIRTH'].std()
df['DAYS_EMPLOYED'] = (df['DAYS_EMPLOYED'] - df['DAYS_EMPLOYED'].mean()) / df['DAYS_EMPLOYED'].std()
df['DAYS_REGISTRATION'] = (df['DAYS_REGISTRATION'] - df['DAYS_REGISTRATION'].mean()) / df['DAYS_REGISTRATION'].std()
df['DAYS_ID_PUBLISH'] = (df['DAYS_ID_PUBLISH'] - df['DAYS_ID_PUBLISH'].mean()) / df['DAYS_ID_PUBLISH'].std()
df['DAYS_LAST_PHONE_CHANGE'] = (df['DAYS_LAST_PHONE_CHANGE'] - df['DAYS_LAST_PHONE_CHANGE'].mean()) / df['DAYS_LAST_PHONE_CHANGE'].std()
df['HOUR_APPR_PROCESS_START'] = (df['HOUR_APPR_PROCESS_START'] - df['HOUR_APPR_PROCESS_START'].mean()) / df['HOUR_APPR_PROCESS_START'].std()
df['ORGANIZATION_TYPE'] = (df['ORGANIZATION_TYPE'] - df['ORGANIZATION_TYPE'].mean()) / df['ORGANIZATION_TYPE'].std()

u = torch.tensor(df.iloc[:, -1].values)
v = torch.tensor(df.iloc[:, :-3].values, dtype=torch.float32)
p = torch.tensor(df_prob, dtype=torch.float32).view(-1, 1)
loan = torch.tensor(df_loan, dtype=torch.float32).view(-1, 1)





dim_v = v.shape[1]
card_u = len(torch.unique(u))
T = 300
n_train = int(u.shape[0] * 0.5)

H = 100
H_prime = 100
len_seq = 40
num_epoch = 300
repeat_num = 1


u = torch.nn.functional.one_hot(u, num_classes=card_u)


u_eval = u
v_eval = v
p_eval = p
loan_eval = loan

torch.manual_seed(0)

random_indices = torch.randperm(u.shape[0])
u = u[random_indices]
v = v[random_indices]
p = p[random_indices]
loan = loan[random_indices]

u_train = u
v_train = v
p_train = p
loan_train = loan

u_list = torch.chunk(u_train, T, dim=0)
v_list = torch.chunk(v_train, T, dim=0)
p_list = torch.chunk(p_train, T, dim=0)
loan_list = torch.chunk(loan_train, T, dim=0)

data_list = (u_list, v_list, p_list, loan_list)
data_train = (u_train, v_train, p_train, loan_train)
data_eval = (u_eval, v_eval, p_eval, loan_eval)

# TODO: random seed should only impact data sequence

value_vanillaGD, best_vanillaGD, final_vanillaGD = [], [], []
value_strategicGD, best_strategicGD, final_strategicGD  = [], [], []
# value_end2endGD, best_end2endGD, final_end2endGD = [], [], []
# value_cutoff = []

init_seed = 10

## for cost_coef = 0.1
# lr_pi = 1.5/100

## for cost_coef = 0.15
# lr_pi = 0.75/100

# for cost_coef = 0.2
# lr_pi = 5e-3

# for cost_coef = 0.25
lr_pi = 1e-2


lr_prob = 1e-2
repeat_num = 10


# cost_coef = 0.1
# cost_coef = 0.15
# cost_coef = 0.2
# cost_coef = 0.25


f = 300

repeat_num = 10

init_seed = 10


for cost_coef in [0.1, 0.15, 0.2]:
    value_vanillaGD, best_vanillaGD, final_vanillaGD = [], [], []
    value_strategicGD, best_strategicGD, final_strategicGD  = [], [], []
    
    for rep_index in range(repeat_num):
        seed = init_seed + rep_index
        print(seed)

        # result_strategicGD = strategicGD(data_list, data_eval, dim_v, card_u, H, f, len_seq, num_epoch, lr_pi=lr_pi, lr_prob = lr_prob,  seed=seed, cost_coef = cost_coef)
        result_vanillaGD = vanillaGD(data_list, data_eval, dim_v, card_u, H_prime, f, seed=seed, lr_pi = lr_pi, cost_coef = cost_coef)
        # result_cutoff = cutoff(data_train, data_eval, card_u, seed=seed, c=cost_coef)

        value_vanillaGD.extend(result_vanillaGD)
        # value_strategicGD.extend(result_strategicGD)
        # value_cutoff.append(result_cutoff)

        torch.save(result_vanillaGD, "result/vani_seed{}_c{}.pkl".format(seed, cost_coef))
        # torch.save(result_strategicGD, "result/stra_seed{}_c{}.pkl".format(seed, cost_coef))    




    with open('result/vanillaGD_{}.json'.format(cost_coef), 'w') as file:
        json.dump(value_vanillaGD, file)

#     with open('result/strategicGD_nov_{}.json'.format(cost_coef), 'w') as file:
#         json.dump(value_strategicGD, file)

#     with open('result/cutoff_{}.json'.format(cost_coef), 'w') as file:
#         json.dump(value_cutoff, file)


