from train_process import*
#from model_final05 import fusion_model
#from model_txt_cross import DFusion_DFI_model as fusion_model
import torch.nn.functional as F
from atom_singal_block3 import DFIFusionModel as fusion_model
import torch.nn as nn
from dataloader_part import create_data_loaders2
import random
from rdkit import RDLogger

class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, alpha=torch.tensor([0.4, 0.6])):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha.to(device)

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        p = torch.exp(-ce_loss)
        loss = (1 - p) ** self.gamma * ce_loss
        return loss.mean()

# 关闭RDKit的警告日志（Level.INFO及以下包含WARNING）
RDLogger.DisableLog('rdApp.warning')  # 屏蔽warning级别
device = "cuda" if torch.cuda.is_available() else'cpu'
def set_randam(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
def untitled(file_name = './datasets/dfi_final.csv', flag = 1, save_name="best_DFI.ckpt"):

    
    early_step,final_score = 0,0
    train_loader, val_loader, test_loader = create_data_loaders2(
        file_name,  batch_size=256, flag=flag)
    model = fusion_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    n_epochs = 100

    # 类别权重：[负样本权重，正样本权重] 缓解Recall过高、Precision偏低
    criterion = FocalLoss(gamma=1.5)
    for epoch in range(n_epochs):
        print("epoch:",epoch+1," / ", n_epochs)
        alpha_weight = torch.tensor([1.0/22, 21.0/22])
        loss_function = FocalLoss(gamma=2.0, alpha=alpha_weight)
        train_part(train_loader,optimizer,model,loss_function)
        metric = valid_part(val_loader,model)
        for key, value in metric.items():
            print(key, value)
        scheduler.step()
        early_step +=1
        if final_score < (metric['auc_roc']+metric["auc_pr"]+metric["f1"]+metric["pre"]+metric["recall"]):
            early_step = 0
            final_score = metric['auc_roc']+metric["auc_pr"]+metric["f1"]+metric["pre"]+metric["recall"]
            torch.save(model.state_dict(),"./save_model02/"+save_name) #保存训练模型
            print('saving model with high auc plus aupr {:.5f}...'.format(final_score))
            # temp_metric = valid_part(test_loader,model)
            # for key, value in temp_metric.items():
            #    print(key, value)
            test_model = fusion_model().to(device)
            test_model.load_state_dict(torch.load("./save_model02/"+save_name))
            print("The validation set reached the maximum value. \n test:")
            test_metric = valid_part(test_loader, test_model)
            for key, value in test_metric.items():
                print(key, value)
        else:
            print("The validation set did not reach the maximum value.:")
            # temp_metric = valid_part(test_loader,model)
            for key, value in test_metric.items():
                print(key, value)
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        print("\n")

set_randam(2025)
k = untitled(flag=1, save_name="cyp450atom03_drug1e3.ckpt") 