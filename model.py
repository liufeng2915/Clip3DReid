

import torch.nn.functional as F
import clip
import torch
import torch.nn as nn
from torch.nn import init
from loss import Classifier, TripletLoss, CrossEntropyWithLabelSmooth
from eval_metrics import evaluate
import wandb
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
        feat_size = config.MODEL.FEATURE_DIM
        self.teacher_cla_w = config.LOSS.TEACHER_CLA
        self.student_cla_w = config.LOSS.STUDENT_CLA
        self.image_alignment_w = config.LOSS.IMG_ALIGNMENT
        self.text_alignment_w = config.LOSS.TEXT_ALIGNMENT

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

        # classifier
        self.classifier = Classifier(feature_dim=576, num_classes=num_classes)
        self.classifier_teacher = Classifier(feature_dim=feat_size, num_classes=num_classes)
        self.criterion_pair = TripletLoss(margin=0.3)
        self.criterion_cla = CrossEntropyWithLabelSmooth()

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
        #global_feat = self.sv_fc(global_feat)
        #return self.sv_bn(global_feat), local_feat
        return global_feat, local_feat

    def distill_kl_loss(self, teacher_logits, student_logits):

        temperature = 2.0  # Temperature > 1 to produce a softer probability distribution
        teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
        loss_kl = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')

        return loss_kl

    def forward(self, clip_model, text_feat, input_image, label):

        # # teacher and student features
        teacher_image_feat = self.forward_clip_image_feature(clip_model, input_image)
        teacher_text_feat = self.forward_clip_text_feature(text_feat)
        student_global_feat, student_local_feat = self.forward_feature(input_image)

        # # Hard target loss
        # teacher
        teacher_logits = self.classifier_teacher(teacher_image_feat)
        loss_teacher_cla = self.criterion_cla(teacher_logits, label) #+ self.criterion_pair(teacher_image_feat, label)
        # student
        student_logits = self.classifier(student_global_feat)
        loss_student_cla = self.criterion_cla(student_logits, label) + self.criterion_pair(student_global_feat, label)
        _, preds = torch.max(student_logits.data, 1)

        # # Aignment loss
        # image feature alignment loss
        loss_image_alignment = self.distill_kl_loss(teacher_logits, student_logits)
        # text feature alignment loss

        loss = self.image_alignment_w*loss_image_alignment + self.teacher_cla_w*loss_teacher_cla + self.student_cla_w*loss_student_cla
        wandb.log({'loss': loss})
        wandb.log({'loss_teacher_cla': loss_teacher_cla})
        wandb.log({'loss_student_cla': loss_student_cla})
        wandb.log({'loss_image_alignment': loss_image_alignment})
        
        return loss, preds