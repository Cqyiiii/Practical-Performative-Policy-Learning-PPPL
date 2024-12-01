import numpy as np
import seaborn as sns;sns.set()
from torch.distributions.bernoulli import Bernoulli
from causalml.inference.meta import BaseDRRegressor, BaseTRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from utils import *


# learn the CATE
def cutoff(data_train, data_eval, card_u, seed, c=0.1):
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    torch.manual_seed(seed)
    np.random.seed(seed)

    cate_learner = BaseTRegressor(learner=XGBRegressor(random_state = 0))  
    (u_train, v_train, p_train, loan_train) = data_train
    (u_eval, v_eval, p_eval, loan_eval) = data_eval

    # training of cate (CRE)
    _, u_category = torch.where(u_train == 1)
    u_category = u_category.view(-1, 1)

    x_train = torch.cat((u_train, v_train), dim=1)
    pi_train = torch.ones_like(u_category) * 0.5
    z_train = Bernoulli(probs=pi_train).sample()
    y_train = y_generate(u_category, p_train, loan_train, z_train)

    pi_train = torch.minimum(torch.tensor(0.999), torch.maximum(torch.tensor(0.001), pi_train))
    cate_learner.fit(X=x_train.cpu().numpy(), treatment=z_train.view(-1).cpu().numpy(), y=y_train.view(-1).cpu().numpy(), p=pi_train.detach().view(-1).cpu().numpy())

    # evaluation of allocation policy
    u_eval_shift = manipulation_cutoff(u_eval, v_eval, card_u, cate_learner, c=c)
    _, u_eval_category_shift = torch.where(u_eval_shift == 1)

    # evaluate tau
    card_u = 10
    n = u_eval.shape[0]
    x_target = torch.cat((u_eval, v_eval), dim=1)        
    tau_pred = cate_predict(x=x_target, pi=None, learner=cate_learner)
    pi_eval = torch.where(tau_pred > 0, torch.tensor(1), torch.tensor(0))
    tau_eval = cate_generate(u_eval_category_shift.view(-1, 1), p_eval, loan_eval)        
    value = torch.mean(pi_eval * tau_eval)
    torch.save(cate_learner, "models/cate_model_c{}.pkl".format(c))    
    
    return value.item()


    
    

def cutoff_oracle(data_eval, seed, c=0.1):
    torch.manual_seed(seed)
    np.random.seed(seed)
    (u_eval, v_eval, p_eval, loan_eval) = data_eval
    u_eval_category_shift = manipulation_cutoff_oracle(u_eval, p_eval, loan_eval, c=c)
    tau_eval = cate_generate(u_eval_category_shift, p_eval.reshape(-1), loan_eval.reshape(-1))
    pi_eval = torch.where(tau_eval > 0, torch.tensor(1), torch.tensor(0))    
    value = torch.mean(pi_eval * tau_eval)
    return value.item()