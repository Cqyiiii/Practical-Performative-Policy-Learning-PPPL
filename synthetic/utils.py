import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions.multinomial import Multinomial



def allocation_policy(num_input, num_hidden):
    model = nn.Sequential(
        nn.Linear(num_input, num_hidden),
        nn.ReLU(),
        nn.Linear(num_hidden, 1),
        nn.Sigmoid()
    )
    return model


def shifted_distribution(num_input, num_hidden, num_strata):
    model = nn.Sequential(
        nn.Linear(num_input, num_hidden),
        nn.ReLU(),
        nn.Linear(num_hidden, num_hidden),
        nn.ReLU(),
        nn.Linear(num_hidden, num_strata)
    )
    return model


def x_generate(n, dim_v, trans, seed=0):
    torch.manual_seed(seed)
    sigma = torch.eye(dim_v)
    mu = torch.zeros(dim_v)
    v = MultivariateNormal(loc=mu, covariance_matrix=sigma).sample((n, ))
    prob_u = nn.Softmax(dim=1)(v @ trans)
    u = Multinomial(probs=prob_u).sample()
    return u, v



def y_generate(x, z, n, beta, seed=0):
    torch.manual_seed(seed)
    epsilon = Normal(loc=0, scale=0.5).sample((n, 1))
    y = (z + 1) * (x @ beta) + epsilon
    return y



def manipulation(u, v, n, card_u, model, c=0.05):
    v_target = v.repeat_interleave(torch.tensor([card_u]).repeat_interleave(n), dim=0)
    u_target = torch.eye(card_u).repeat(n, 1)
    x_target = torch.cat((u_target, v_target), dim=1)
    pi = model(x_target).view(-1, card_u)
    u_init = u.repeat_interleave(torch.tensor([card_u]).repeat_interleave(n), dim=0)
    _, u_init_category = torch.where(u_init == 1)
    _, u_taregt_category = torch.where(u_target == 1)
    cost = torch.abs(u_taregt_category - u_init_category).view(-1, card_u)
    inputs = torch.cat((v, pi), dim=1)
    utility = pi - c * cost
    index = utility.argmax(dim=1)
    u_shift = torch.zeros((n, card_u))
    u_shift[torch.arange(n), index] = 1
    return inputs, index.view(-1, 1), u_shift



def manipulation_theta(u, v, n, card_u, model, c=0.05):
    v_target = v.repeat_interleave(torch.tensor([card_u]).repeat_interleave(n), dim=0)
    u_target = torch.eye(card_u).repeat(n, 1)
    x_target = torch.cat((u_target, v_target), dim=1)
    pi = model(x_target).view(-1, card_u)

    u_init = u.repeat_interleave(torch.tensor([card_u]).repeat_interleave(n), dim=0)
    _, u_init_category = torch.where(u_init == 1)
    _, u_taregt_category = torch.where(u_target == 1)
    cost = torch.abs(u_taregt_category - u_init_category).view(-1, card_u)

    theta_detach = nn.utils.parameters_to_vector(model.parameters()).detach().repeat(n, 1)
    theta = nn.utils.parameters_to_vector(model.parameters()).repeat(n, 1)
    
    inputs_for_policy = torch.cat((v, theta), dim=1)
    inputs_for_behavior = torch.cat((v, theta_detach), dim=1)
    utility = pi - c * cost
    index = utility.argmax(dim=1)

    u_shift = torch.zeros((n, card_u))
    u_shift[torch.arange(n), index] = 1

    return inputs_for_behavior, inputs_for_policy, index.view(-1, 1), u_shift



def manipulation_cutoff(u, v, n, card_u, beta, c=0.05):
    v_target = v.repeat_interleave(torch.tensor([card_u]).repeat_interleave(n), dim=0)
    u_target = torch.eye(card_u).repeat(n, 1)
    x_target = torch.cat((u_target, v_target), dim=1)
    tau = x_target @ beta
    pi = torch.where(tau > 0, torch.tensor(1), torch.tensor(0)).view(-1, card_u)
    u_init = u.repeat_interleave(torch.tensor([card_u]).repeat_interleave(n), dim=0)
    _, u_init_category = torch.where(u_init == 1)
    _, u_taregt_category = torch.where(u_target == 1)
    cost = torch.abs(u_taregt_category - u_init_category).view(-1, card_u)
    utility = pi - c * cost
    index = utility.argmax(dim=1)
    u_shift = torch.zeros((n, card_u))
    u_shift[torch.arange(n), index] = 1
    return u_shift



def cate_fit(seq_x, seq_y, seq_z, seq_pi, learner):
    stacked_x = torch.cat(seq_x, dim=0)
    stacked_z = torch.cat(seq_z, dim=0)
    stacked_y = torch.cat(seq_y, dim=0)
    stacked_pi = torch.cat(seq_pi, dim=0)
    stacked_pi = torch.minimum(torch.tensor(0.999), torch.maximum(torch.tensor(0.001), stacked_pi))
    learner.fit(X=stacked_x.cpu().numpy(), treatment=stacked_z.view(-1).cpu().numpy(), y=stacked_y.view(-1).cpu().numpy(), p=stacked_pi.detach().view(-1).cpu().numpy())



def cate_predict(x, pi, learner):
    cate = learner.predict(X=x.cpu().numpy(), p=pi)
    tau = torch.from_numpy(cate)
    return tau



def model_params_flatten(model):
    params_flatten = torch.cat([param.data.view(-1) for param in model.parameters()])
    dim_theta = params_flatten.shape[0]

    return dim_theta, params_flatten