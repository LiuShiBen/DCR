import os
import torch
import torch.nn.functional as F
import copy
import torch.nn as nn
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .fusion import FusionLayer
from collections import OrderedDict
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID
# from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss, ContrastiveLoss
# Attribute-Net
from .backbones.Attr_factory import build_backbone, build_classifier
from .backbones.Attr_block import FeatClassifier
from .backbones import swin_transformer, resnet50, bninception
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout).cuda()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout).cuda()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm1 = self.norm1.requires_grad_()
        self.norm2 = nn.LayerNorm(d_model)
        # self.norm2 = self.norm2.requires_grad_()
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, prototype, global_feat,
                     pos, query_pos):
        q = k = self.with_pos_embed(prototype, query_pos)
        prototype_2 = self.self_attn(q, k, value=prototype)[0]
        prototype = prototype + self.dropout1(prototype_2)
        prototype = self.norm1(prototype)
        out_prototype = self.multihead_attn(query=self.with_pos_embed(prototype, query_pos),
                                            key=self.with_pos_embed(global_feat, pos),
                                            value=global_feat)[0]
        prototype = prototype + self.dropout2(out_prototype)
        prototype = self.norm2(prototype)
        prototype = self.linear2(self.dropout(self.activation(self.linear1(prototype))))
        prototype = prototype + self.dropout3(prototype)
        prototype = self.norm3(prototype)
        return prototype

    def forward(self, prototype, global_feat, pos=None, query_pos=None):
        return self.forward_post(prototype, global_feat, pos, query_pos)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):  # nn.GELU
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, prototype, global_feat,
                pos=None, query_pos=None):
        output = prototype
        intermediate = []
        for layer in self.layers:
            output = layer(output, global_feat,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        if self.return_intermediate:
            return torch.stack(intermediate)
        return output

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

class Encode_text_img(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.positional_embedding = clip_model.positional_embedding
        self.text_projection = clip_model.text_projection
        self.end_id = clip_model.end_id
    def forward(self, text_tokens):
        text_feat = []
        for items in text_tokens:
            text_token = clip.tokenize(items["text"])
            text_token = text_token.cuda()
            text_feat.append(text_token)

        text = torch.cat(text_feat, dim=0)
        text = text.cuda()  # 64,77
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x


class IM2TEXT(nn.Module):
    def __init__(self, embed_dim=512, middle_dim=512, output_dim=512, n_layer=3, dropout=0.1):
        super().__init__()
        self.fc_out = nn.Linear(middle_dim, output_dim)
        self.bottleneck = nn.BatchNorm1d(output_dim)  # (self.in_planes_proj)
        layers = []
        dim = embed_dim
        for _ in range(n_layer):
            block = []
            block.append(nn.Linear(dim, middle_dim))
            block.append(nn.Dropout(dropout))
            block.append(nn.ReLU())
            dim = middle_dim
            layers.append(nn.Sequential(*block))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = x.cuda()
        for layer in self.layers:
            x = layer(x)
        #x = self.fc_out(x)
        x = self.bottleneck(x)
        return x


import torch
import torch.nn as nn
import math


class CrossAttention(nn.Module):
    def __init__(self, model_dim=768, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        head_dim = model_dim // num_heads
        self.scale = head_dim ** -0.5
        self.kv_linear = nn.Linear(model_dim, model_dim * 2, bias=qkv_bias)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj = nn.Linear(model_dim, model_dim)


    def forward(self, query, key_value):
        #print("query:", query.shape, key_value.shape)
        B, N, C = key_value.shape
        # Split by heads
        query_proj = query.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        kv_proj = self.kv_linear(key_value).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        key_proj, value_proj = kv_proj[0], kv_proj[1]   # make torchscript happy (cannot use tensor as tuple)
        attn = (query_proj @ key_proj.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ value_proj).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Self_Attention(nn.Module):
    def __init__(self, model_dim=768, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super(Self_Attention, self).__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        head_dim = model_dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv1_linear = nn.Linear(model_dim, model_dim * 3, bias=qkv_bias)
        self.qkv2_linear = nn.Linear(model_dim, model_dim * 3, bias=qkv_bias)
        self.softmax = nn.Softmax(dim=-1)
        self.attn1_drop = nn.Dropout(attn_drop)
        self.attn2_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj = nn.Linear(model_dim, model_dim)
        drop_path = 0.
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, qkv):
        B, N, C = qkv.shape

        #first self-attention
        qkv_proj1 = self.qkv1_linear(qkv).reshape(B, N,3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        query_proj1, key_proj1, value_proj1 = qkv_proj1[0], qkv_proj1[1], qkv_proj1[2]  # make torchscript happy (cannot use tensor as tuple)
        attn1 = (query_proj1 @ key_proj1.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn1_drop(attn1)
        attn1 = (attn1 @ value_proj1).transpose(1, 2).reshape(B, N, C)
        attn1 = qkv + self.drop_path(attn1)
        #second self-attention
        qkv_proj2 = self.qkv2_linear(attn1).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        query_proj2, key_proj2, value_proj2 = qkv_proj2[0], qkv_proj2[1], qkv_proj2[2]  # make torchscript happy (cannot use tensor as tuple)
        attn2 = (query_proj2 @ key_proj2.transpose(-2, -1)) * self.scale
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn2_drop(attn2)
        attn2 = (attn2 @ value_proj2).transpose(1, 2).reshape(B, N, C)
        attn2 = attn2 + self.drop_path(attn1)
        x = self.proj(attn2)
        x = self.proj_drop(x)
        return x

class Text_img_interaction(nn.Module):
    def __init__(self, model_dim=768, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super(Text_img_interaction, self).__init__()
        self.cross_Attention = CrossAttention(model_dim=model_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        #self.self_Attention = Self_Attention(model_dim=model_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)

    def forward(self, text, img):
        x = self.cross_Attention(text, img)
        #x = self.self_Attention(x)
        return x
def get_reload_weight(model_path, model, pth='ckpt_max.pth'):
    model_path = os.path.join(model_path, pth)

    load_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    # print(load_dict)
    if isinstance(load_dict, OrderedDict):
        pretrain_dict = load_dict
    else:
        pretrain_dict = load_dict['state_dicts']
        print(f"best performance {load_dict['metric']} in epoch : {load_dict['epoch']}")

    #model.load_state_dict(pretrain_dict, strict=True)
    model.load_state_dict({k.replace('module.', ''): v for k, v in pretrain_dict.items()})

    return model
import re
from collections import  Counter
def remove_specific_duplicate_words(text, words_to_process):
    pattern = r'\b(?:' + '|'.join(re.escape(word) for word in words_to_process) + r')\b'

    parts = re.findall(r'\w+|[^\w\s]', text)

    occurrences = {}
    for i, part in enumerate(parts):
        if re.match(pattern, part):
            if part not in occurrences:
                occurrences[part] = []
            occurrences[part].append(i)

    single_occurrences = {word for word, indices in occurrences.items() if len(indices) == 1}

    result = []
    for i, part in enumerate(parts):
        if part in occurrences:
            if part in single_occurrences:
                continue
            if occurrences[part][-1] == i:
                result.append(part)
        else:
            result.append(part)

    result_text = ''.join(
        ' ' + part if re.match(r'\w+', part) and i != 0 and re.match(r'\w+', result[i - 1]) else part
        for i, part in enumerate(result)
    )

    result_text = re.sub(r'[^\w\s](?=[^\w\s]*$)', '.', result_text)

    return result_text

# Attribute
clas_name = ['Hat', 'Glasses', 'ShortSleeve', 'LongSleeve',
             'UpperStride', 'UpperLogo', 'UpperPlaid', 'UpperSplice',
             'LowerStripe', 'LowerPattern', 'LongCoat', 'Trousers',
             'Shorts', 'Skirt&Dress', 'boots', 'HandBag', 'ShoulderBag',
             'Backpack', 'HoldObjectsInFront', 'AgeOver60', 'Age18-60',
             'AgeLess18', 'Female', 'Front', 'Side', 'Back']

class Attr_generator(nn.Module):

    def __init__(self):
        super(Attr_generator, self).__init__()
        self.TYPE = 'resnet50'
        self.MULTISCALE = False
        self.NAME = 'linear'
        backbone, c_output = build_backbone(self.TYPE, self.MULTISCALE)
        # print(backbone)
        classifier = build_classifier(self.NAME)(
            nattr=26,
            c_in=c_output,
            bn=False,
            pool='avg',
            scale=1
        )
        # print(classifier)

        self.Attr_Net = FeatClassifier(backbone, classifier)
        if torch.cuda.is_available():
            self.Attr_Net = self.Attr_Net.cuda()
            #self.Attr_Net = nn.DataParallel(self.Attr_Net)
        self.Attr_Net = get_reload_weight("./Weights/", self.Attr_Net, pth='ckpt_max_2023-05-15_21_15_23.pth')
        self.Attr_Net = self.Attr_Net.eval()
    def forward(self, Attr):
        # print("Attr", Attr.shape)
        with torch.no_grad():
            valid_logits, attns = self.Attr_Net(Attr)
            valid_probs = torch.sigmoid(valid_logits[0])
        return valid_probs

class Attr_select(nn.Module):

    def __init__(self, num_query = 12):
        super(Attr_select, self).__init__()
        self.num_query = num_query

    def forward(self, Attr):
        # print("Attr", Attr.shape)

        valid_probs = Attr > 0.80


        # valid_probs = (valid_probs -0.5)*2
        bs, c = Attr.shape
        text_list = []
        text0 = "A photo of a person"
        for i in range(bs):
            dictionary = {"text": text0}
            text_list.append(dictionary)
        # False and True Transfor 0 and 1
        text_list = []
        text1 = "A photo of"
        for i in range(bs):
            # man/woman
            if valid_probs[i][22] == False:
                text2 = " a man "
            elif valid_probs[i][22] == True:
                text2 = " a woman "
            else:
                text2 = " a person "

            # HandBag/ ShoulderBag / Backpack
            if valid_probs[i][15] or valid_probs[i][16] or valid_probs[i][17]:
                if valid_probs[i][15] == True:
                    text3 = "while carrying a handbag."
                elif valid_probs[i][16] == True:
                    text3 = "while carrying a shoulder bag."
                elif valid_probs[i][17] == True:
                    text3 = "while carrying a backpack."
                else:
                    text3 = ""
            else:
                text3 = ""

            if valid_probs[i][0]:
                text4 = "and a hat,"
            else:
                text4 = ""

            if valid_probs[i][1]:
                text5 = "and a pair of glasses,"
            else:
                text5 = ""

            # shortsleeve/ longsleeve
            if valid_probs[i][2] == True:
                text6 = "and a short sleeved top,"
            else:
                text6 = ""
            if valid_probs[i][3] == True:
                text7 = "and a long sleeved top,"
            else:
                text7 = ""
            if valid_probs[i][10] == True:
                text8 = "and a long coat,"
            else:
                text8 = ""

            # Trousers/ shorts /skirt and dress
            if valid_probs[i][11] == True:
                text9 = "and a trousers,"
            else:
                text9 = ""
            if valid_probs[i][12] == True:
                text10= "and a shorts,"
            else:
                text10 = ""
            if valid_probs[i][13] == True:
                text11 = "and a skirt,"
            else:
                text11 = ""
            if valid_probs[i][2] or valid_probs[i][3] or valid_probs[i][10] or valid_probs[i][11] or valid_probs[i][12] or valid_probs[i][13]:
                text678 = "wearing " + text4 + text5 + text6 + text7 + text8 + text9 + text10 + text11
            else:
                text678 = ""
            text0 = text1 + text2 + text678 + text3
            #text0 = remove_specific_duplicates_keep_last(text0, "and")
            text0 = remove_specific_duplicate_words(text0, ["and"])
            #print("text0", text0)
            dictionary = {"text": text0}
            text_list.append(dictionary)

        Attrs = torch.cat((valid_probs[:, 0:4], valid_probs[:, 10:14], valid_probs[:, 15:18], valid_probs[:, 22:23]), dim=1)  # delete Age, Front,Side, Back.
        # valid_probs, index = torch.topk(valid_probs, k=self.num_query, largest=True)
        bs, c = Attrs.shape
        # False and True Transfor 0 and 1
        Attrs_score = torch.Tensor(bs, self.num_query).to("cuda")
        for i in range(bs):
            for j in range(self.num_query):  # 9
                if Attrs[i][j] == True:
                    Attrs_score[i][j] = 1
                else:
                    Attrs_score[i][j] = 0
        return text_list, Attrs_score

class AttributionDecoder(nn.Module):

    def __init__(self, Attr_dim=768, nhead=12, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(Attr_dim, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(Attr_dim, nhead, dropout=dropout)
        # Implementation of Feedforward model

        self.norm1 = nn.LayerNorm(Attr_dim)
        self.norm2 = nn.LayerNorm(Attr_dim)
        self.norm3 = nn.LayerNorm(Attr_dim)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.mlp = nn.Sequential(
            nn.Linear(Attr_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, Attr_dim),
        )

        self.fusion = nn.Sequential(
            nn.Linear(Attr_dim * 2, Attr_dim),
            nn.ReLU(),
            nn.Linear(Attr_dim, Attr_dim),
            nn.LayerNorm(Attr_dim),
        )

    def pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, Attr_feat, global_feat, pos=None, query_pos=None):
        #print("Attr_feat, global_feat", Attr_feat.shape, global_feat.shape,)
        q = k = self.pos_embed(Attr_feat, query_pos)
        Attr_1 = self.self_attn(q, k, value=Attr_feat)[0]
        Attr_2 = self.norm1(Attr_feat + self.drop1(Attr_1))

        out_prototype = self.multihead_attn(query=self.pos_embed(Attr_2, query_pos),
                                            key=self.pos_embed(global_feat, pos),
                                            value=global_feat)[0]
        Attr_2 = Attr_2 + self.drop2(out_prototype)
        Attr_2 = self.norm2(Attr_2)  # [10, 64, 768]
        Attr_2 = self.mlp(Attr_2)
        Attr_2 = Attr_2 + self.drop3(Attr_2)
        Attr_2 = self.norm3(Attr_2)

        Attr_3 = torch.cat((Attr_2, Attr_feat), dim=2)
        Attr_out = self.fusion(Attr_3)
        return Attr_out

#Text-Image Fusion
class text_image_Fusion(nn.Module):
    def __init__(self, dim=768, nhead=8, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.cross_attn_text = nn.MultiheadAttention(dim, nhead, dropout=dropout)
        self.cross_attn_img = nn.MultiheadAttention(dim, nhead, dropout=dropout)
        # Implementation of Feedforward model

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)


        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim),
        )
        self.img_kv = nn.Linear(dim, dim * 2)
        self.text_kv = nn.Linear(dim, dim * 2)

        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, text_feat, img_feat):

        # procession one  text_feat = Q, image_feat= K,V
        text_feat = text_feat.unsqueeze(0)
        img_feat = img_feat.permute(1, 0, 2)

        N, B, C = img_feat.shape
        img_featture = self.img_kv(img_feat).reshape(N, B, 2, C).permute(2, 0, 1, 3)
        img_K, img_V = img_featture[0], img_featture[1]
        # print("img_K", img_K.shape, img_V.shape, text_feat.shape)
        Text_1 = self.cross_attn_text(text_feat, img_K, value=img_V)[0]
        # print("Text_1", Text_1.shape)
        Text_2 = self.norm1(text_feat + self.drop1(Text_1))
        # print("Text_2", Text_2.shape)

        # procession two  text_feat = K,V, image_feat= Q
        N, B, C = text_feat.shape
        text_featture = self.text_kv(text_feat).reshape(N, B, 2, C).permute(2, 0, 1, 3)
        text_K, text_V = text_featture[0], text_featture[1]
        # print("text_K", text_K.shape, text_V.shape, img_feat.shape)
        Img_1 = self.cross_attn_img(query=img_feat, key=text_K, value=text_V)[0]
        # print("Img_1", Img_1.shape)
        Img_2 = self.norm2(img_feat + self.drop2(Img_1))
        # print("Img_2", Img_2.shape)

        Img_2 = Img_2.permute(1, 0, 2)
        B, N, C = Img_2.shape
        Text_2 = Text_2.permute(1, 0, 2).repeat(1, N, 1)

        text_img = torch.cat((Text_2, Img_2), dim=2)
        text_img_out = self.fusion(text_img)
        # print("text_img_out", text_img_out.shape)
        return text_img_out

class build_transformer(nn.Module):
    def __init__(self, num_classes, cfg):
        super(build_transformer, self).__init__()
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.class_token = cfg.MODEL.CLASS_TOKEN
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
            self.in_text_proj = 64
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes

        self.classifier1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier1.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_fin = nn.BatchNorm1d(self.in_planes)  #(self.in_planes_proj)
        self.bottleneck_fin.bias.requires_grad_(False)
        self.bottleneck_fin.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)  #(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.bottleneck_text = nn.BatchNorm1d(self.in_planes_proj)  #(self.in_planes_proj)
        self.bottleneck_text.bias.requires_grad_(False)
        self.bottleneck_text.apply(weights_init_kaiming)

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        #clip_model.to("cuda")

        self.image_encoder = clip_model.visual

        self.logit_scale = clip_model.logit_scale.exp()
        self.logit_scale = self.logit_scale.mean()

        self.text_encoder = Encode_text_img(clip_model)

        #self.img_tex_fuse = FusionLayer(cfg.Fusion)
        self.img_tex_fuse = Text_img_interaction()
        self.text_proj = nn.Linear(512, cfg.Fusion.hidden_size)  # 768

        # part view based decoder
        self.dim_forward = 2048
        self.decoder_drop = 0.1
        self.decoder_norm = nn.LayerNorm(self.in_planes)
        self.decoder_numlayer = 6
        self.num_head = 8
        # query setting
        self.query_embed = nn.Embedding(self.class_token, self.in_planes).weight

        self.transformerdecoderlayer = TransformerDecoderLayer(self.in_planes, self.num_head, self.dim_forward, self.decoder_drop, "relu")
        self.transformerdecoder = TransformerDecoder(self.transformerdecoderlayer, self.decoder_numlayer, self.decoder_norm)
        self.transformerdecoder = self.transformerdecoder.cuda()

        self.fusion = text_image_Fusion(self.in_planes, self.num_head, self.dim_forward, self.decoder_drop,  "relu")

        # Attribute Net
        self.Attr_select = Attr_select()
        self.Attr_generator = Attr_generator()

        self.attr_query = 12
        self.attr_decoder_linear = nn.Linear(self.attr_query, self.in_planes)

    def forward(self, x = None):

        valid_probs = self.Attr_generator(x)
        text_attr, attr_score = self.Attr_select(valid_probs)

        if self.model_name == 'RN50':
            image_features, image_features_proj = self.image_encoder(x)
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1)
            img_feature_proj = image_features_proj[0]

        elif self.model_name == 'ViT-B-16':
            #Image Encoder
            image_features, image_features_proj = self.image_encoder(x)
            #print("text_token", text_token.shape, image_features_proj.shape)
            B, N, C = image_features.shape
            #Text Encoder
            text_features = self.text_encoder(text_tokens=text_attr)
            #up dimension
            text_feature_proj = self.text_proj(text_features)
            #print("text_feature", text_feature.shape, image_features.shape)
            #text_feature：[64,768], image_features:[64,129,768]
            text_img_fin = self.fusion(text_feature_proj, image_features)  #64,132,768
            #print("text_img_fin", text_img_fin.shape)

            # Input of decoder
            #
            #print("decoder_value", decoder_value.shape)
            feat_attr = self.attr_decoder_linear(attr_score).unsqueeze(1)  # [bs, 1, 768]
            decoder_value = image_features * feat_attr
            decoder_value = decoder_value.permute(1,0,2)
            #decoder_value = self.attr_decoder_linear(attr_score).unsqueeze(0)  # [1, bs, 768]
            #print("decoder_value", decoder_value.shape)

            # shared information
            query_embed = self.query_embed
            #print("Dec_out:", query_embed.shape)
            query_embed = query_embed.unsqueeze(1).repeat(1, B, 1)
            prototype = torch.zeros_like(query_embed)
            # part-view based decoder
            #print("Dec_out", prototype.shape, decoder_value.shape)
            Dec_out = self.transformerdecoder(prototype, decoder_value, query_pos=query_embed)  #4,64,768
            Dec_out = Dec_out.permute(1,0,2)
            #print("Dec_out", Dec_out.shape, text_img_fin[:, 0:4].shape)
            # AFA Attribute feature alignment
            Dec_out = AFA(Dec_out, text_img_fin[:, 0:self.class_token])  # [bs 5 768]

            img_feature = image_features[:, 0]
            for i in range(1, self.class_token):
                img_feature = img_feature + image_features[:, i]

            text_img_fins = text_img_fin[:, 0]
            for i in range(1, self.class_token):
                text_img_fins = text_img_fins + text_img_fin[:, i]

        feat = self.bottleneck(img_feature)
        feat_fin = self.bottleneck_fin(text_img_fins)

        if self.training:
            cls_score_fin = self.classifier1(feat_fin)
            #print("img_feature_proj:", cls_score_fin.shape, img_feature.shape, image_features.shape, text_img_fin.shape, Dec_out.shape, )
            return cls_score_fin, [img_feature], Dec_out, image_features[:, :self.class_token], img_feature, text_feature_proj
        else:
            #return torch.cat([feat, feat_fin], dim=1)
            return feat_fin


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def AFA(attr_feat, text_img_feat):
    '''
    @matrix shape [bs, 17, 768]
    @matrix1 shape [bs, 17, 768]

    '''
    assert attr_feat.shape[0] == text_img_feat.shape[0], 'Wrong shape'
    assert attr_feat.shape[1] == text_img_feat.shape[1], 'Wrong skt num'

    batch_size = attr_feat.shape[0]  # [bs, attr_num, 768]

    attr_sim_weight = attr_feat * text_img_feat  # [bs, attr_num, 768]

    final_sim = F.cosine_similarity(attr_feat.unsqueeze(2), attr_sim_weight.unsqueeze(1), dim=3)  # [bs, 17, x]

    _, ind = torch.max(final_sim, dim=2)

    sim_match = []
    for i in range(batch_size):
        org_mat = attr_feat[i]  # [attr_num, C]
        sim_mat = attr_sim_weight[i]  # [attr_num, C]
        shuffle_mat = []

        for j in range(ind.shape[1]):
            new = org_mat[j] + sim_mat[ind[i][j]]  # [C]
            new = new.unsqueeze(0)
            shuffle_mat.append(new)

        bs_mat = torch.cat(shuffle_mat, dim=0)
        sim_match.append(bs_mat)
    alignment_feat = torch.stack(sim_match, dim=0)
    return alignment_feat

from .clip import clip
def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)
    return model


def build_vit_backbone(num_class, cfg):
    """
    Create a ResNet instance from config.
    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    model = build_transformer(num_class, cfg)

    return model




