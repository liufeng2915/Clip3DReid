

import torch.nn.functional as F
import clip
import torch
import torch.nn as nn
from torch.nn import init
from losses.cross_entropy_loss import Classifier
from losses.triplet_loss import TripletLoss
from tiny_vit import encoder_tinyvit


def load_clip_to_cpu(clip_backbone_name):

    model, preprocess = clip.load(clip_backbone_name, device="cpu")

    return model

class ClipModel(nn.Module):
    def __init__(self, clip_backbone_name='laion/CLIP-ViT-H-14-laion2B-s32B-b79K'):
        super().__init__()
        self.clip_model = load_clip_to_cpu(clip_backbone_name)
    def forward(self, image):

        image_feat = self.clip_model.encode_image(image)
        return image_feat


class Model(nn.Module):
    def __init__(self, config, num_classes=632):
        super().__init__()

        ##
        self.max_iter = 100
        feat_size = config.MODEL.FEATURE_DIM
        self.teacher_cla_w = config.LOSS.TEACHER_CLA
        self.student_cla_w = config.LOSS.STUDENT_CLA
        self.global_alignment_w = config.LOSS.GLOBAL_ALIGNMENT
        self.local_alignment_w = config.LOSS.LOCAL_ALIGNMENT

        #  fc probe
        self.sv_fc = nn.Linear(576, feat_size)
        self.tv_fc = nn.Linear(768, feat_size)
        self.tt_fc = nn.Linear(768, 576)
        self.sv_bn = nn.BatchNorm1d(feat_size)
        init.normal_(self.sv_bn.weight.data, 1.0, 0.02)
        init.constant_(self.sv_bn.bias.data, 0.0)
        self.tv_bn = nn.BatchNorm1d(feat_size)
        init.normal_(self.tv_bn.weight.data, 1.0, 0.02)
        init.constant_(self.tv_bn.bias.data, 0.0)

        # student TinyViT model
        self.student_encoder = encoder_tinyvit()

        # criterion
        self.classifier = Classifier(feature_dim=feat_size, num_classes=num_classes)
        self.classifier_teacher = Classifier(feature_dim=feat_size, num_classes=num_classes)
        self.criterion_pair = TripletLoss(margin=0.3)
        self.criterion_cla = CrossEntropyWithLabelSmooth()
        self.criterion_ot = SinkhornDistance(eps=0.1, max_iter=100, reduction='mean')

    def forward_clip_image_feature(self, clip_model, image):

        with torch.no_grad():
            local_feature = clip_model(image)
        feat = self.tv_fc(local_feature)
        feat = self.tv_bn(feat)
        return feat

    def forward_clip_text_feature(self, text_feat):

        return self.tt_fc(text_feat)

    def forward_feature(self, input_image):

        local_feat, global_feat = self.student_encoder(input_image)
        global_feat = self.sv_fc(global_feat)
        #return self.sv_bn(global_feat), local_feat
        return global_feat, local_feat

    def distill_kl_loss(self, teacher_logits, student_logits):

        temperature = 2.0  # Temperature > 1 to produce a softer probability distribution
        teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
        loss_kl = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')

        return loss_kl

    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-2
        for i in range(self.max_iter):
            r0 = r
            r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
            c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break

        T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K

        return T

    def ot_compute(self, student_local_feat, teacher_text_feat):

        # student_local_feat: B*49*576
        # teacher_text_feat: B*16*576
        b = student_local_feat.shape[0]
        eps = 0.1
        student_local_feat = F.normalize(student_local_feat, dim=-1)
        teacher_text_feat = F.normalize(teacher_text_feat, dim=-1)

        sim = torch.einsum('bmd,bnd->bmn', student_local_feat, teacher_text_feat).contiguous()
        wdist = 1.0 - sim
        M = student_local_feat.shape[1]
        N = teacher_text_feat.shape[1]
        xx = torch.zeros(b, M, dtype=sim.dtype, device=sim.device).fill_(1. / M)
        yy = torch.zeros(b, N, dtype=sim.dtype, device=sim.device).fill_(1. / N)

        with torch.no_grad():
            KK = torch.exp(-wdist / eps)
            T = self.Sinkhorn(KK, xx, yy)
        if torch.isnan(T).any():
            return None

        sim_op = torch.sum(T * sim, dim=(1, 2))

        loss_sim = 1 - sim_op.mean()

        # logit_scale = np.exp(self.logit_scale)
        # logits = logit_scale * torch.bmm(global_gt_feat.unsqueeze(1), global_esti_feat.unsqueeze(2))
        # logits = logits.squeeze(-1).squeeze(-1)
        # logits2 = logit_scale * sim_op
        # logits2 = logits + logits2

        return loss_sim


    def forward(self, clip_model, text_feat, input_image, label):

        # # teacher and student features
        teacher_image_feat = self.forward_clip_image_feature(clip_model, input_image)
        teacher_text_feat = self.forward_clip_text_feature(text_feat)
        student_global_feat, student_local_feat = self.forward_feature(input_image)

        # # Hard target loss
        # teacher
        teacher_logits = self.classifier_teacher(teacher_image_feat)
        loss_teacher_cla = self.criterion_cla(teacher_logits, label) + self.criterion_pair(teacher_image_feat, label)
        # student
        student_logits = self.classifier(student_global_feat)
        loss_student_cla = self.criterion_cla(student_logits, label) + self.criterion_pair(student_global_feat, label)
        _, preds = torch.max(student_logits.data, 1)

        # # Aignment loss
        # image feature alignment loss
        loss_global_alignment = self.distill_kl_loss(teacher_logits, student_logits)
        # text feature alignment loss
        loss_local_alignment = self.ot_compute(student_local_feat, teacher_text_feat)

        loss = self.local_alignment_w*loss_local_alignment+self.global_alignment_w*loss_global_alignment + self.teacher_cla_w*loss_teacher_cla + self.student_cla_w*loss_student_cla
        
        return loss, preds