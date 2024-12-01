import torch
import torch.nn as nn


def allocation_policy(num_input, num_hidden):

    model = nn.Sequential(
        nn.Linear(num_input, num_hidden),
        nn.ReLU(),
        nn.Linear(num_hidden, num_hidden),
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


def y_generate(u_category, p, loan, z, beta = 4, weight = 0.2):
    p1 = p
    p2 = (u_category + 0.5) / 10
    p = weight * p1 + (1 - weight) * p2

    y0 = 0
    y1 = loan * ((beta+1) * p - beta)
    y = y0 * (1 - z) + y1 * z

    return y


def cate_generate(u_category, p, loan, beta = 4, weight = 0.4):

    weight = 0.4
    p1 = p
    p2 = (u_category + 0.5) / 10
    p = weight * p1 + (1 - weight) * p2

    cate = loan * ((beta+1) * p - beta)

    return cate


def manipulation(u, v, card_u, model, c=0.1):
    n = u.shape[0]

    v_target = v.repeat_interleave(torch.tensor([card_u]).repeat_interleave(n), dim=0)
    u_target = torch.eye(card_u).repeat(n, 1)
    x_target = torch.cat((u_target, v_target), dim=1)
    pi = model(x_target).view(-1, card_u)
    # print(pi.shape, pi)

    u_init = u.repeat_interleave(torch.tensor([card_u]).repeat_interleave(n), dim=0)
    _, u_init_category = torch.where(u_init == 1)
    _, u_taregt_category = torch.where(u_target == 1)
    cost = torch.abs(u_taregt_category - u_init_category).view(-1, card_u)

    inputs = torch.cat((v, pi), dim=1)
    utility = pi - c * cost

    u_category = utility.argmax(dim=1)
    u_shift = torch.nn.functional.one_hot(u_category, num_classes=card_u)

    return inputs, u_category.view(-1, 1), u_shift


def pi_var(u, v, card_u, model):
    n = u.shape[0]
    v_target = v.repeat_interleave(torch.tensor([card_u]).repeat_interleave(n), dim=0)
    u_target = torch.eye(card_u).repeat(n, 1)
    x_target = torch.cat((u_target, v_target), dim=1)
    pi = model(x_target).view(-1, card_u)
    return pi.var(dim=1)


def manipulation_end2end(u, v, card_u, model, c=0.1):
    n = u.shape[0]

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

    u_category = utility.argmax(dim=1)
    u_shift = torch.nn.functional.one_hot(u_category, num_classes=card_u)

    return inputs_for_behavior, inputs_for_policy, u_category.view(-1, 1), u_shift


def manipulation_cutoff(u, v, card_u, cate_learner, c=0.1):
    n = u.shape[0]

    v_target = v.repeat_interleave(torch.tensor([card_u]).repeat_interleave(n), dim=0)
    u_target = torch.eye(card_u).repeat(n, 1)
    x_target = torch.cat((u_target, v_target), dim=1)
    tau = cate_predict(x=x_target, pi=None, learner=cate_learner)
    pi = torch.where(tau > 0, torch.tensor(1), torch.tensor(0)).view(-1, card_u)

    u_init = u.repeat_interleave(torch.tensor([card_u]).repeat_interleave(n), dim=0)
    _, u_init_category = torch.where(u_init == 1)
    _, u_taregt_category = torch.where(u_target == 1)
    cost = torch.abs(u_taregt_category - u_init_category).view(-1, card_u)

    utility = pi - c * cost
    u_category = utility.argmax(dim=1)
    u_shift = torch.nn.functional.one_hot(u_category, num_classes=card_u)

    return u_shift



def manipulation_cutoff_oracle(u, p, loan, c=0.1):
    n = len(u)
    u_target = torch.ones((10, n))
    for i in range(10):
        u_target[i] *= i
    cate_target = cate_generate(u_target, p.reshape(-1), loan.reshape(-1))
    pi_target = (cate_target > 0).int()        
    mani_cost = c * torch.abs(u_target - u)
    u_shift = (pi_target - mani_cost).argmax(dim=0)
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
    tau = torch.from_numpy(cate).to(device="cuda:0")

    return tau