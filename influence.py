import torch
from torch_influence import BaseObjective
from torch_influence import AutogradInfluenceModule, LiSSAInfluenceModule, CGInfluenceModule

class MyObjective(BaseObjective):
    def train_outputs(self, model, batch):
        return model(batch[0])

    def train_loss_on_outputs(self, outputs, batch):
        criterion=torch.nn.CrossEntropyLoss(reduction="mean")
        return criterion(outputs, batch[1])  # mean reduction required

    def train_regularization(self, params):
        return 0.01 * torch.square(params.norm())

    def test_loss(self, model, params, batch):
        criterion=torch.nn.CrossEntropyLoss(reduction="mean")
        return criterion(model(batch[0]), batch[1])  # no regularization in test loss


def get_influence(train_loader, test_loader, model, seed, method, num_layer_freeze):
    for param in model.parameters():
        param.requires_grad = False
        
    for name, param in list(model.classifier.named_parameters()):
        param.requires_grad = False
        
    for name, param in list(model.classifier.named_parameters())[-2 * num_layer_freeze:]:  # Exclude the last layer
        param.requires_grad = True
        
    print("unfrozen layers:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print (name)
            
    torch.manual_seed(seed)
    model.eval()
    approx_total_num_data = len(train_loader) * train_loader.batch_size
    if method == "lissa":
        module = LiSSAInfluenceModule(
            model=model,
            objective=MyObjective(),
            train_loader=train_loader,
            test_loader=test_loader,
            device="cuda:5",
            damp=0.001,
            repeat=1,
            depth=int(approx_total_num_data/10),
            scale=1e4,
        )
    
    if method == "cg":
        module = CGInfluenceModule(
            model=model,
            objective=MyObjective(),
            train_loader=train_loader,
            test_loader=test_loader,
            device="cuda:5",
            damp=0.001,
            gnh=True,
        )
    
    if method == "autograd":
        module = AutogradInfluenceModule(
            model=model,
            objective=MyObjective(),  
            train_loader=train_loader,
            test_loader=test_loader,
            device="cuda:5",
            damp=0.001
        )

    num_train_datapt = len(train_loader.dataset)
    num_test_datapt = len(test_loader.dataset)
    influences = module.influences(range(num_train_datapt), range(num_test_datapt))
    return influences

def load_training_influence(path, domain, method) -> torch.tensor:
    influences = torch.load(path+method+"_"+domain+"_training.pt")
    return influences
def load_val_influence(path, domain, method) -> torch.tensor:
    influences = torch.load(path+method+"_"+domain+"_val.pt")
    return influences
