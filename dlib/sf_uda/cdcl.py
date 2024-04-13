import torch
import os
import torch.nn.functional as F
from tqdm import tqdm
from math import ceil
from scipy.optimize import linear_sum_assignment
from torch import nn, Tensor
from typing import Dict, Iterable, Callable

from dlib.datasets.wsol_loader import get_data_loader
from dlib.configure import constants
import dlib.dllogger as DLLogger
from dlib.utils.shared import fmsg







__all__ = ['Cdcl']

def to_cuda(x):
    return x.cuda()

def to_onehot(label, num_classes):
    identity = to_cuda(torch.eye(num_classes))
    onehot = torch.index_select(identity, 0, label)
    return onehot

class DIST(object):
    def __init__(self, dist_type='cos'):
        self.dist_type = dist_type

    def get_dist(self, pointA, pointB, cross=False):
        return getattr(self, self.dist_type)(
            pointA, pointB, cross)

    def cos(self, pointA, pointB, cross):
        pointA = F.normalize(pointA, dim=1)
        pointB = F.normalize(pointB, dim=1)
        if not cross:
            return 0.5 * (1.0 - torch.sum(pointA * pointB, dim=1))
        else:
            # NA = pointA.size(0)
            # NB = pointB.size(0)
            assert (pointA.size(1) == pointB.size(1))
            return 0.5 * (1.0 - torch.matmul(pointA, pointB.transpose(0, 1)))
        
class Clustering(object):
    def __init__(self, eps, feat_key, max_len=1000, dist_type='cos'):
        self.eps = eps
        self.Dist = DIST(dist_type)
        self.samples = {}
        self.path2label = {}
        self.center_change = None
        self.stop = False
        self.feat_key = feat_key
        self.max_len = max_len

    def set_init_centers(self, init_centers):
        self.centers = init_centers
        self.init_centers = init_centers
        self.num_classes = self.centers.size(0)

    def clustering_stop(self, centers):
        if centers is None:
            self.stop = False
        else:
            dist = self.Dist.get_dist(centers, self.centers)
            dist = torch.mean(dist, dim=0)
            print('dist %.4f' % dist.item())
            self.stop = dist.item() < self.eps

    def assign_labels(self, feats):
        dists = self.Dist.get_dist(feats, self.centers, cross=True)
        _, labels = torch.min(dists, dim=1)
        return dists, labels
    


    def align_centers(self):
        cost = self.Dist.get_dist(self.centers, self.init_centers, cross=True)
        cost = cost.data.cpu().numpy()
        _, col_ind = linear_sum_assignment(cost)
        return col_ind

    def collect_samples(self, net, loader):
        data_feat, data_gt, data_paths, data_truth = [], [], [], []
        #layer_to_extract_feat = 'classification_head.avgpool'
        #self.feature_extractor = FeatureExtractor_for_source_code(model=net, layers=[layer_to_extract_feat])
        for sample in iter(loader):
            #data = sample['Img'].cuda()
            data = sample[0].cuda()
            data_truth += sample[1]
            #data_paths += sample['Path']
            data_paths += sample[3]
            # if 'Label' in sample.keys():
            #     data_gt += [to_cuda(sample['Label'])]

            # output = net.forward(data)
            # feature = output[self.feat_key].data
            #feature = net.forward(data, get_feature=True)[-1].data
            out = net(data)
            feature = net.lin_ft
            #feature = self.feature_extractor(data)[layer_to_extract_feat].squeeze(2).squeeze(2)
            data_feat += [feature]

        self.samples['data'] = data_paths
        # self.samples['gt'] = torch.cat(data_gt, dim=0) \
        #     if len(data_gt) > 0 else None
        self.samples['gt'] = None
        self.samples['feature'] = torch.cat(data_feat, dim=0)
        self.samples['data_truth_label'] = torch.tensor([t.item() for t in data_truth])

    def feature_clustering(self, net, loader):
        centers = None
        self.stop = False

        self.collect_samples(net, loader)
        feature = self.samples['feature']

        refs = to_cuda(torch.LongTensor(range(self.num_classes)).unsqueeze(1))
        num_samples = feature.size(0)
        num_split = ceil(1.0 * num_samples / self.max_len)

        while True:
            self.clustering_stop(centers)
            if centers is not None:
                self.centers = centers
            if self.stop:
                break

            centers = 0
            count = 0

            start = 0
            for N in range(num_split):
                cur_len = min(self.max_len, num_samples - start)
                cur_feature = feature.narrow(0, start, cur_len)
                dist2center, labels = self.assign_labels(cur_feature)
                labels_onehot = to_onehot(labels, self.num_classes)
                count += torch.sum(labels_onehot, dim=0)
                labels = labels.unsqueeze(0)
                mask = (labels == refs).unsqueeze(2).type(torch.cuda.FloatTensor)
                reshaped_feature = cur_feature.unsqueeze(0)
                # update centers
                centers += torch.sum(reshaped_feature * mask, dim=1)
                start += cur_len

            mask = (count.unsqueeze(1) > 0).type(torch.cuda.FloatTensor)
            centers = mask * centers + (1 - mask) * self.init_centers

        dist2center, labels = [], []
        start = 0
        count = 0
        for N in range(num_split):
            cur_len = min(self.max_len, num_samples - start)
            cur_feature = feature.narrow(0, start, cur_len)
            cur_dist2center, cur_labels = self.assign_labels(cur_feature)

            labels_onehot = to_onehot(cur_labels, self.num_classes)
            count += torch.sum(labels_onehot, dim=0)

            dist2center += [cur_dist2center]
            labels += [cur_labels]
            start += cur_len

        self.samples['label'] = torch.cat(labels, dim=0)
        self.samples['dist2center'] = torch.cat(dist2center, dim=0)

        cluster2label = self.align_centers()
        # reorder the centers
        self.centers = self.centers[cluster2label, :]
        # re-label the data according to the index
        num_samples = len(self.samples['feature'])
        for k in range(num_samples):
            self.samples['label'][k] = cluster2label[self.samples['label'][k]].item()


        acc = (self.samples['label'].detach().cpu() == self.samples['data_truth_label']).float().mean() * 100.
        msg = f"SFDE - ACC pseudo-label image-class -- : {acc} %"
        DLLogger.log(fmsg(msg))

        self.center_change = torch.mean(self.Dist.get_dist(self.centers,self.init_centers))

        for i in range(num_samples):
            self.path2label[self.samples['data'][i]] = self.samples['label'][i].item()

        del self.samples['feature']


class FeatureExtractor_for_source_code(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        _ = self.model(x)
        return self._features
    



class Cdcl(object):
    def __init__(self,
                 model_trg,
                 train_loader_trg,
                 task: str,
                 n_cls: int,
                 dist_type = 'cos',
                 support_background = False,
                 threshold = 1.0,
                 ):
        

        self.model = model_trg
        self.train_loader_trg = train_loader_trg
        self.clustering = Clustering(0.001,'feats',1000)
        self.support_background = support_background
        self.threshold = threshold
        #self.source_anchors = F.normalize(torch.randn(2, 2048), dim=1).cpu()

    def solve(self):
            torch.cuda.empty_cache()
            self.model.eval()
            with torch.no_grad():
                self.update_labels()
                self.clustered_target_samples = self.clustering.samples
                target_hypt, filtered_classes = self.filtering()
                sfuda_select_ids_pl = {image_id: pseudo_label.item() for image_id, pseudo_label in zip(target_hypt['data'], target_hypt['label'])}
                
            return sfuda_select_ids_pl, target_hypt,  filtered_classes

    def update_labels(self):
        init_target_centers = self.model.get_linear_weights
        #self.net.module.set_bn_domain(self.bn_domain_map[self.target_name])
        self.clustering.set_init_centers(init_target_centers)
        self.clustering.feature_clustering(self.model, self.train_loader_trg)

    
    def filter_samples_2(self, samples, threshold=0.8):
        batch_size_full = len(samples['data'])
        unique_labels = torch.unique(samples['label'])

        filtered_samples = {'data': [], 'label':  torch.tensor([], dtype=torch.long), 'gt': []}

        for label in unique_labels:
            mask = samples['label'] == label
            data = [samples['data'][i] for i in range(len(samples['data'])) if mask[i]]
            min_dist = torch.min(samples['dist2center'], dim=1)[0]
            dist = torch.masked_select(min_dist, mask)
            gt = torch.masked_select(samples['gt'], mask) if samples['gt'] is not None else None

            sorted_indices = torch.argsort(dist).tolist()

            num_samples = int(threshold * len(sorted_indices))
            selected_indices = sorted_indices[:num_samples]

            filtered_samples['data'].extend([data[i] for i in selected_indices])
            filtered_samples['label'] = torch.cat((filtered_samples['label'], torch.full((num_samples,), label, dtype=torch.long)))
            filtered_samples['gt'].extend(gt[selected_indices] if gt is not None else [])

        assert len(filtered_samples['data']) == len(filtered_samples['label'])
        print('select %f' % (1.0 * len(filtered_samples['data']) / batch_size_full))
        filtered_samples['label'] = filtered_samples['label'].to(samples['label'].device)

        return filtered_samples


    # def filter_samples(self, samples, threshold=0.05):
    #     batch_size_full = len(samples['data'])
    #     min_dist = torch.min(samples['dist2center'], dim=1)[0]
    #     mask = min_dist < threshold

    #     filtered_data = [samples['data'][m]
    #                  for m in range(mask.size(0)) if mask[m].item() == 1]
    #     filtered_label = torch.masked_select(samples['label'], mask)
    #     filtered_gt = torch.masked_select(samples['gt'], mask) \
    #         if samples['gt'] is not None else None

    #     filtered_samples = {}
    #     filtered_samples['data'] = filtered_data
    #     filtered_samples['label'] = filtered_label
    #     filtered_samples['gt'] = filtered_gt

    #     assert len(filtered_samples['data']) == filtered_samples['label'].size(0)
    #     print('select %f' % (1.0 * len(filtered_data) / batch_size_full)) 

    #     return filtered_samples
    
    def filter_class(self, labels, num_min, num_classes):
        filted_classes = []
        for c in range(num_classes):
            mask = (labels == c)
            count = torch.sum(mask).item()
            if count >= num_min:
                filted_classes.append(c)

        return filted_classes


    def filtering(self):
        threshold = self.threshold
        min_sn_cls = 5
        target_samples = self.clustered_target_samples

        # chosen_samples = self.filter_samples(target_samples, threshold=threshold)
        chosen_samples_2 = self.filter_samples_2(target_samples, threshold=threshold)
        # error_found = False

        # for data_item in chosen_samples['data']:
        #     index = chosen_samples['data'].index(data_item)
        #     index2 = chosen_samples_2['data'].index(data_item)
        #     label1 = chosen_samples['label'][index]
        #     label2 = chosen_samples_2['label'][index2]

        #     # Compare labels
        #     if label1 != label2:
        #         print(f"La valeur de data '{data_item}' a des labels différents dans les deux dictionnaires.")
        #         error_found = True
        #         break

        # if error_found:
        #     print("Une erreur a été trouvée : les labels pour au moins un élément diffèrent dans les deux dictionnaires.")
        # else:
        #     print("Aucune erreur trouvée : les labels pour tous les éléments sont les mêmes dans les deux dictionnaires.")

        filtered_classes = self.filter_class(chosen_samples_2['label'], min_sn_cls, 2)
        print('The number of filtered classes: %d' % len(filtered_classes))

        return chosen_samples_2, filtered_classes
    
    def collect_target_feature_mean_std(self, filtered_classes,loader):

        feats, plabels = [], []
        #layer_to_extract_feat = 'classification_head.avgpool'
        #self.feature_extractor = FeatureExtractor_for_source_code(model=self.model, layers=[layer_to_extract_feat])

        for data, label, plabel, _, _, _, _ in iter(loader):
            #feat = self.feature_extractor(data.cuda())[layer_to_extract_feat].squeeze(2).squeeze(2)
            out = self.model(data.cuda())
            feat = self.model.lin_ft
            #feat = self.net(to_cuda(data))[self.opt.CLUSTERING.FEAT_KEY]
            feats += [feat.detach()]
            plabels += [to_cuda(plabel)]

            print((label - plabel).abs().sum())

        feats = torch.cat(feats, dim=0)
        plabels = torch.cat(plabels, dim=0)

        assert feats.shape[0] == plabels.shape[0]

        class_mean = torch.zeros(2, feats.shape[-1])
        class_std = torch.zeros(2, feats.shape[-1])
        for c in filtered_classes:
            index = torch.where(plabels == c)[0]
            _std, _mean = torch.std_mean(feats[index], dim=0, unbiased=True)
            class_std[c] = _std
            class_mean[c] = _mean

        del feats
        del plabels
        return class_mean.cpu(), class_std.cpu()
    

    def construct_surrogate_feature_sampler(self, filtered_classes, loader):
        """
        The time complexity of sampling from a multivariate Gaussian N(mu, cov) is O(n^2).
        Especially when the dimension of feature here is 2048.
        So we only keep those values on diagonal of the covariance matrix,
        in order to make the time complexity of sampling to be near O(n).
        """
        variance_mult = 1
        print(f'Collecting mean and stddev of target features...variance_mult={variance_mult}')
        target_mean, target_std = self.collect_target_feature_mean_std(filtered_classes,loader)

        normal_sampler = {}
        self.source_anchors = F.normalize(self.clustering.init_centers, dim=1).cpu()

        for i in range(2):
            cur_target_norm = target_mean[i].norm(p=2)
            if cur_target_norm < 1e-3:
                normal_sampler[i] = None
                continue

            estimated_source_mean = self.source_anchors[i] * cur_target_norm
            estimated_source_std = target_std[i]
            # eliminate 0 value in case of ValueError() raised by torch.distribution
            estimated_source_std[estimated_source_std == 0] = 1e-4
            estimated_source_std = estimated_source_std * variance_mult
            normal_sampler[i] = torch.distributions.normal.Normal(
                estimated_source_mean, estimated_source_std)

        self.surrogate_feature_sampler = normal_sampler

        return normal_sampler
    
        print('surrogate feature samplers constructed.\n')
    
    def forward_data(self,target_features) -> dict:
        """
        Forward data through the model and return the output.
        """

        results = {
            'target_features': None,
        }	

        results['target_features'] = target_features

        return results