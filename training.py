import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset



# def train(
#     model,
#     erasure_criterion,
#     X,
#     Z,
#     batch_size,
#     learning_rate, 
#     weight_decay,
#     gamma,
#     num_epochs,
#     scheduler_milestones,
#     scheduler_gamma,
#     device,
#     seed = None,
#     **args
# ):
#     model = model.to(device)
#     model.train()

#     if seed is not None:
#         torch.manual_seed(seed) 

#     # Initialize the optimizer and the scheduler
#     if len(list(model.parameters())) > 0:
#         optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#         scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_milestones, gamma=scheduler_gamma)

#     # Initialize the data loader
#     # indices_X = torch.arange(n).to(device)
#     dataset = TensorDataset(X, Z)
#     train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     # Training loop
#     from tqdm import tqdm
#     iterations = tqdm(range(num_epochs))
    
#     for _, epoch in enumerate(iterations, 0):
#         for _, (inputs, labels) in enumerate(train_loader):
#             inputs, labels = inputs.to(device), labels.to(device)
#             if not len(inputs) < batch_size: # ignore incomplete batches
#                 optimizer.zero_grad()
#                 outputs = model(inputs, reduction=False)
#                 erasure_loss = erasure_criterion(outputs, inputs, labels) # MMD or KRaM

#                 # Add the projection loss if it is defined as a model method
#                 projection_loss =  model.projection_properties_eval() if (hasattr(model,'projection_properties_eval') and callable(model.projection_properties_eval)) else None
                
#                 loss = erasure_loss + gamma*projection_loss if projection_loss is not None else erasure_loss

#                 loss.backward()
#                 optimizer.step()

#                 # Training status display
#                 iterations.set_description(f"[train] loss: {loss}, erasure loss: {erasure_loss}, projection loss: {projection_loss}")
#         scheduler.step() # update the scheduler at the end of each epoch


def train(
    model,
    erasure_criterion,
    X,
    Z,
    batch_size,
    learning_rate, 
    weight_decay,
    gamma,
    num_epochs,
    scheduler_milestones,
    scheduler_gamma,
    device,
    seed = None,
    gradient_acumulation_steps = 1,
    **kwargs
):
    model = model.to(device)
    model.train()

    if seed is not None:
        torch.manual_seed(seed) 

    # Initialize the optimizer and the scheduler
    if len(list(model.parameters())) > 0:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_milestones, gamma=scheduler_gamma)

    # Initialize the data loader
    # indices_X = torch.arange(n).to(device)
    dataset = TensorDataset(X, Z)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    from tqdm import tqdm
    iterations = tqdm(range(num_epochs))
    
    grad_acc = 0
    for _, epoch in enumerate(iterations, 0):
        for _, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            if not len(inputs) < batch_size: # ignore incomplete batches
                optimizer.zero_grad() if grad_acc == 0 else None
                


                outputs = model(inputs, reduction=False)
                erasure_loss = erasure_criterion(outputs, inputs, labels) # MMD or KRaM

                # Add the projection loss if it is defined as a model method
                projection_loss =  model.projection_properties_eval() if (hasattr(model,'projection_properties_eval') and callable(model.projection_properties_eval)) else None
                
                loss_unit = erasure_loss + gamma*projection_loss if projection_loss is not None else erasure_loss

                loss = loss_unit if grad_acc == 0 else loss + loss_unit
                grad_acc = (grad_acc + 1) % gradient_acumulation_steps
                loss = loss / gradient_acumulation_steps if grad_acc == 0 else loss
                loss.backward() if grad_acc == 0 else None
                optimizer.step() if grad_acc == 0 else None

                # Training status display
                iterations.set_description(f"[train] loss: {loss}, erasure loss: {erasure_loss}, projection loss: {projection_loss}")
        scheduler.step() # update the scheduler at the end of each epoch