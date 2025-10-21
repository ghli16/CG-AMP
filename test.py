import torch
from torch.utils.data import DataLoader

from get_data import data
from model import newModel
from mydataset import MyDataset, collate_fn1, collate_fn2
from utils import get_metrics

BATCH_SIZE = 32
def test(test_loader):
    model = newModel().to('cuda')

    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    y_pred_all = []
    real_all = []
    with torch.no_grad():
        for feature_test1, feature_test2, labels_test in test_loader:
            y_pred_test = model.trainmodel(feature_test1, feature_test2)
            y_pred_all.append(y_pred_test)
            real_all.append(labels_test)

        y_pred_all = torch.cat(y_pred_all, dim=0).cpu().reshape(-1).numpy()
        real_all = torch.cat(real_all, dim=0).cpu().reshape(-1).numpy()
        metric_tmp = get_metrics(real_all, y_pred_all)

        # if (epoch + 1) % 10 == 0 or epoch == 0:
        print(
            'testset: f1:{} acc:{} recall:{} precision:{} mcc:{} :rocauc:{}'.format(metric_tmp[0], metric_tmp[1],
                                                                          metric_tmp[2],
                                                                          metric_tmp[3], metric_tmp[
                                                                              4], metric_tmp[5]))
																			  
_, _, test_set = data('dataset/AMPlify/AMPlify.fasta', 'dataset/AMPlify/amplify_esm2.npy')
test_dataset = MyDataset(test_set)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn2)
test(test_loader)