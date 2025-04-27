
import PIL.Image as Image
import time
from .loss import TripletLoss, CrossEntropyLabelSmooth, SoftTripletLoss
import torch
import torch.nn as nn
from .utils.meters import AverageMeter
from .utils.my_tools import *
import numpy as np
from torch.nn import functional as F

class Trainer:
    def __init__(self, args, model, tmodel, optimizer, num_classes,
                 data_loader_train, data_loader_replay, training_phase, add_num=0, replay=False, margin=0.0,
                 ):

        self.model = model
        self.model.cuda()
        self.tmodel = tmodel
        if self.tmodel is not None:
            self.tmodel.cuda()
        self.replay = replay
        self.data_loader_train = data_loader_train
        self.data_loader_replay = data_loader_replay
        self.training_phase = training_phase
        self.add_num = add_num
        self.gamma = 0.5
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
        self.criterion_triple = SoftTripletLoss(margin=margin).cuda()
        self.trip_hard = TripletLoss(margin=margin).cuda()
        self.T = 2
        self.consistency = 0.2
        self.consistency_rampup = 100.0
        self.train_iters = len(self.data_loader_train)
        #self.device, available_gpus = self._get_available_devices(args.gpu)
        self.model = torch.nn.DataParallel(self.model, device_ids=args.device_id)
        self.L1_loss = torch.nn.L1Loss()
        # set optimizer and learning rate
        self.optimizer = optimizer

    @torch.no_grad()
    def update_teachers(self, teacher, keep_rate=0.996):
        # exponential moving average(EMA)
        for ema_param, param in zip(teacher.parameters(), self.model.parameters()):
            ema_param.data = (keep_rate * ema_param.data) + (1 - keep_rate) * param.data

    def predict_with_out_grad(self, imgs):
        with torch.no_grad():
            cls_out_old, features_old, Dec_out_old, feat_fin, _, _,= self.tmodel(imgs)
        return cls_out_old, features_old, Dec_out_old, feat_fin

    def freeze_teachers_parameters(self):
        for p in self.tmodel.parameters():
            p.requires_grad = False

    def get_reliable(self, teacher_predict, student_predict, positive_list, p_name, score_r):
        N = teacher_predict.shape[0]
        score_t = self.iqa_metric(teacher_predict).detach().cpu().numpy()
        score_s = self.iqa_metric(student_predict).detach().cpu().numpy()
        positive_sample = positive_list.clone()
        for idx in range(0, N):
            if score_t[idx] > score_s[idx]:
                if score_t[idx] > score_r[idx]:
                    positive_sample[idx] = teacher_predict[idx]
                    # update the reliable bank
                    temp_c = np.transpose(teacher_predict[idx].detach().cpu().numpy(), (1, 2, 0))
                    temp_c = np.clip(temp_c, 0, 1)
                    arr_c = (temp_c*255).astype(np.uint8)
                    arr_c = Image.fromarray(arr_c)
                    arr_c.save('%s' % p_name[idx])
        del N, score_r, score_s, score_t, teacher_predict, student_predict, positive_list
        return positive_sample

    def train(self, epoch):

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_base = AverageMeter()
        losses_KD = AverageMeter()
        loss_AKD = AverageMeter()

        end = time.time()
        self.model.train()
        if self.tmodel is not None:
            self.freeze_teachers_parameters()

        for i in range(len(self.data_loader_train)):
            train_inputs = self.data_loader_train.next()
            data_time.update(time.time() - end)
            imgs, targets, cids, domains = self._parse_data(train_inputs)
            #print("imgs:", imgs.shape, targets.shape)
            targets += self.add_num
            #Current network output
            cls_out, features, Attr_out, feat_fin, img_proj, text_proj, = self.model(imgs)
            #print("features", g_features.shape, Attr_out.shape, img_proj.shape, text_proj.shape)
            #corss-entroy loss of new samples
            loss_ce = self.CE_loss(cls_out, targets)
            #triplet loss of new samples
            loss_tp = self.Hard_loss(features, targets)
            loss_attr = self.Attr_loss(Attr_out, targets)
            loss_Dissimilar = self.Dissimilar(feat_fin)
            #print("loss_Dissimilar:", loss_Dissimilar)
            #loss_acl = self.ACL(feat_fin, Attr_out)
            loss_i2t = self.SupConLoss(img_proj, text_proj, targets, targets)
            loss_t2i = self.SupConLoss(text_proj, img_proj, targets, targets)
            #print("loss:", loss_ce, loss_tp, loss_attr, L1_loss, (loss_i2t + loss_t2i) / 2)
            loss = loss_ce + loss_tp + loss_Dissimilar + loss_attr + (loss_i2t + loss_t2i) / 2
            #print("loss:", loss, loss_ce, loss_tp, loss_Dissimilar, loss_acl, (loss_i2t + loss_t2i) / 2)

            # rehearsal
            if self.replay is True:
                imgs_r, fnames_r, pid_r, cid_r, domain_r = next(iter(self.data_loader_replay))
                imgs_r = imgs_r.cuda()
                pid_r = pid_r.cuda()
                # Current network output
                cls_out_r, features_r, Attr_out_r, feat_fin_r, _, _ = self.model(imgs_r)

                # triplet loss of memory samples
                loss_tr_r = self.Hard_loss(features_r, pid_r)
                loss += loss_tr_r

                # Memory network output
                cls_out_old, features_old, Attr_out_old, feat_fin_old = self.predict_with_out_grad(imgs)
                cls_out_r_old, features_r_old, Attr_out_r_old, feat_fin_r_old = self.predict_with_out_grad(imgs_r)

                # consostent and logit-level supervisory loss
                #loKD_loss_r = self.loss_kd_old(features_r, features_r_old)
                loKD_loss_r = self.loss_kd_L1(feat_fin_r, feat_fin_r_old)
                loss += loKD_loss_r
                losses_KD.update(loKD_loss_r)
                #attribution consistent
                loss_AKD_r = self.loss_kd_L1(Attr_out_r, Attr_out_r_old)
                loss_AKD.update(loss_AKD_r)
                loss += loss_AKD_r
                loss += self.loss_kd_js(cls_out_old, cls_out)

                del cls_out, cls_out_r_old, cls_out_r, features_r, features_r_old, features

            losses_base.update(loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.tmodel is not None:
                with torch.no_grad():
                    self.update_teachers(teacher=self.tmodel)

            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) == self.train_iters or (i + 1) % (self.train_iters // 4) == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Loss_base {:.3f} ({:.3f})\t'
                      'Loss_kd {:.3f} ({:.3f})\t'
                      'Loss_Akd {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, self.train_iters,
                              batch_time.val, batch_time.avg,
                              losses_base.val, losses_base.avg,
                              losses_KD.val, losses_KD.avg,
                              loss_AKD.val, loss_AKD.avg))

    def ACL(self, g_features, Attr_feat):
        loss = []
        B, N, C = Attr_feat.shape
        for i in range(1, N):
            distance_matrix = self.cosine_distance(g_features[:, i], Attr_feat[:, i])
            #print("distance_matrix:", distance_matrix.shape)
            loss.append(torch.mean(distance_matrix))
        loss = torch.mean(torch.stack(loss))
        return loss

    def loss_kd_old(self, new_feature, old_feature):
        new_features = torch.cat([new_feature[0], new_feature[1]], dim=1)
        #new_features = new_feature[0]
        #print("new_features", new_features.shape)
        new_features = new_features.detach()

        old_features = torch.cat([old_feature[0], old_feature[1]], dim=1)
        #old_features = old_feature[0]
        old_features = old_features.detach()

        L1 = torch.nn.L1Loss()

        old_simi_matrix = self.cosine_distance(old_features, old_features)
        #print("old_simi_matrix:", old_simi_matrix.shape)
        new_simi_matrix = self.cosine_distance(new_features, new_features)

        simi_loss = L1(old_simi_matrix, new_simi_matrix)

        return simi_loss*50

    def loss_kd_L1(self, new_features, old_features):
        L1 = torch.nn.L1Loss()
        B, N, C = new_features.shape
        new_feat = new_features[:, 0]
        for i in range(1, N):
            new_feat = torch.cat([new_feat, new_features[:, i]], dim=1)

        B, N, C = old_features.shape
        old_feat = old_features[:, 0]
        for i in range(1, N):
            old_feat = torch.cat([old_feat, old_features[:, i]], dim=1)
        #print("new_feat", new_feat.shape, old_feat.shape)

        old_simi_matrix = self.cosine_distance(old_feat, old_feat)
        new_simi_matrix = self.cosine_distance(new_feat, new_feat)

        simi_loss = L1(old_simi_matrix, new_simi_matrix)
        #print("simi_loss", simi_loss)
        return simi_loss*50

    def _get_available_devices(self, n_gpu):
        sys_gpu = torch.cuda.device_count()
        if sys_gpu == 0:
            print('No GPUs detected, using the CPU')
            n_gpu = 0
        elif n_gpu > sys_gpu:
            print(f'Nbr of GPU requested is {n_gpu} but only {sys_gpu} are available')
            n_gpu = sys_gpu
        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        available_gpus = list(range(n_gpu))
        return device, available_gpus

    def _parse_data(self, inputs):
        imgs, _, pids, cids, domains = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets, cids, domains

    def CE_loss(self, s_outputs, targets):
        loss_ce = self.criterion_ce(s_outputs, targets)  #ID loss
        return loss_ce

    def Tri_loss(self, s_features, targets):
        fea_loss = []
        for i in range(len(s_features)):
            loss_tr = self.criterion_triple(s_features[i], s_features[i], targets) #tri loss
            fea_loss.append(loss_tr)
        loss_tr = sum(fea_loss)# / len(fea_loss)
        return loss_tr

    def Hard_loss(self, s_features, targets):
        fea_loss = []
        for i in range(0, len(s_features)):
            #print("loss_tr:", s_features[i].shape, targets.shape)
            loss_tr = self.trip_hard(s_features[i], targets)[0]
            fea_loss.append(loss_tr)
        loss_tr = sum(fea_loss)# / len(fea_loss)
        return loss_tr

    def Attr_loss(self, Attr_features, targets):
        fea_loss = []
        B,N,C = Attr_features.shape
        for i in range(0, N):
            #print("Attr_loss:", Attr_features[:, i].shape, targets.shape)
            loss_tr = self.trip_hard(Attr_features[:, i], targets)[0]
            fea_loss.append(loss_tr)
        loss_tr = sum(fea_loss) / N
        return loss_tr

    def cosine_distance(sself, input1, input2):
        """Computes cosine distance.
        Args:
            input1 (torch.Tensor): 2-D feature matrix.
            input2 (torch.Tensor): 2-D feature matrix.
        Returns:
            torch.Tensor: distance matrix.
        """
        input1_normed = F.normalize(input1, p=2, dim=1)
        input2_normed = F.normalize(input2, p=2, dim=1)
        distmat = 1 - torch.mm(input1_normed, input2_normed.t())
        return distmat

    def loss_kd_js(self, old_logit, new_logit):
        old_logits = old_logit.detach()
        new_logits = new_logit
        #print("new_logits:", new_logits.shape, old_logits.shape)
        p_s = F.log_softmax((new_logits + old_logits)/(2*self.T), dim=1)
        p_t = F.softmax(old_logits/self.T, dim=1)
        p_t2 = F.softmax(new_logits/self.T, dim=1)
        loss = 0.5*F.kl_div(p_s, p_t, reduction='batchmean')*(self.T**2) + 0.5*F.kl_div(p_s, p_t2, reduction='batchmean')*(self.T**2)
        return loss

    def Dissimilar(self, g_feat):
        B, N, C = g_feat.shape
        dist_mat = self.cosine_dist(g_feat, g_feat)
        top_triu = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
        _dist = dist_mat[:, top_triu]
        dist = torch.mean(_dist, dim=(0, 1))
        #print("g_feat", g_feat.shape)
        #loss_ort = torch.triu(torch.bmm(g_feat, g_feat.permute(0, 2, 1)), diagonal=1).sum() / (g_feat.size(0))
        return dist

    def cosine_dist(self, x, y):
        """
        Args:
          x: pytorch Variable, with shape [B, m, d]
          y: pytorch Variable, with shape [B, n, d]
        Returns:
          dist: pytorch Variable, with shape [B, m, n]
        """
        B = x.size(0)
        m, n = x.size(1), y.size(1)
        x_norm = torch.pow(x, 2).sum(2, keepdim=True).sqrt().expand(B, m, n)
        y_norm = torch.pow(y, 2).sum(2, keepdim=True).sqrt().expand(B, n, m).transpose(-2, -1)
        xy_intersection = x @ y.transpose(-2, -1)
        dist = xy_intersection / (x_norm * y_norm)
        return torch.abs(dist)

    def SupConLoss(self, text_features, image_features, t_label, i_targets):
            batch_size = text_features.shape[0]
            batch_size_N = image_features.shape[0]
            mask = torch.eq(t_label.unsqueeze(1).expand(batch_size, batch_size_N), \
                            i_targets.unsqueeze(0).expand(batch_size, batch_size_N)).float().cuda()

            logits = torch.div(torch.matmul(text_features, image_features.T), 1)
            # for numerical stability
            logits_max, _ = torch.max(logits, dim=1, keepdim=True)
            logits = logits - logits_max.detach()
            exp_logits = torch.exp(logits)
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
            loss = - mean_log_prob_pos.mean()

            return loss