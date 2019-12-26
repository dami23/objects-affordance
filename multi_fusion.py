import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)
import numpy as np

out_dim = 512
fused_dim = 1024

class Attention_weight(nn.Module):
    def __init__(self, embedding_size, attention_size, vis_size, tex_size):
        super(Attention_weight, self).__init__()

        self.embedding_size = embedding_size
        self.attention_size = attention_size

        self.bias = torch.nn.Parameter(torch.randn(1)).cuda()
        self.H = torch.nn.Parameter(torch.randn(self.attention_size))

        self.vis_size = vis_size
        self.tex_size = tex_size

        self.vis_first_embedding = nn.Embedding(self.vis_size, 1).cuda()
        self.vis_linear = nn.Linear(self.vis_size, self.embedding_size).cuda()

        self.tex_first_embedding = nn.Embedding(self.tex_size, 1).cuda()
        self.tex_linear = nn.Linear(self.tex_size, self.embedding_size).cuda()

        self.attention_linear_drop = nn.Dropout(0.5).cuda()
        self.attention_linear1 = nn.Linear(65536, 512)

    def forward(self, input_v, input_t):
        vis_emd = self.vis_first_embedding(input_v.cuda())
        vis_emd = self.vis_linear(vis_emd.view(1, self.vis_size).float().cuda())

        tex_emd = self.tex_first_embedding(input_t.cuda())
        tex_emd = self.tex_linear(tex_emd.view(1, self.tex_size).float().cuda())

        wij_arr = vis_emd.t() * tex_emd
        wij_arr = wij_arr.view(1, self.embedding_size * self.embedding_size)

        interaction_layer = self.attention_linear_drop(wij_arr.data.cpu())
        attention_tmp = self.attention_linear1(interaction_layer)
        attention_tmp = getattr(torch, 'tanh')(attention_tmp)
        attention_tmp = attention_tmp * self.H
        attention_weight = torch.nn.Softmax(dim=1)(attention_tmp)
        attention_weight = attention_weight.view(1, 1, 512)

        return attention_weight

class Embeddings_Fusion(nn.Module):
    def __init__(self,):
        super(Embeddings_Fusion, self).__init__()

    def forward(self, atten_weight, input_v, input_t):
        if input_v.dim() != input_t.dim() and input_v.dim() != 3:
            raise ValueError

        batch_size = input_v.size(0)
        weight_height = input_v.size(1)
        dim_hv = input_v.size(2)
        dim_ht = input_t.size(2)
        dim_att = atten_weight.size(2)

        if not atten_weight.is_contiguous():
            atten_weight = atten_weight.contiguous()
        if not input_v.is_contiguous():
            input_v = input_v.contiguous()
        if not input_t.is_contiguous():
            input_t = input_t.contiguous()

        atten_weight = atten_weight.view(1 * 1, dim_att)
        x_v = input_v.view(batch_size * weight_height, dim_hv)
        x_t = input_t.view(batch_size * weight_height, dim_ht)

        one_tensor = torch.ones(batch_size * weight_height, dim_att)
        x_vv = torch.mul((one_tensor - atten_weight.data.cpu()), x_v.data.cpu())
        x_tt = torch.mul(atten_weight.data.cpu(), x_t.data.cpu())
        # x_mm = torch.mul(x_v, x_t)
        x_mm = torch.cat((x_vv, x_tt), 1)
        x_mm = x_mm.view(batch_size, weight_height, fused_dim)

        return x_mm

class MultiFusion(nn.Module):
    def __init__(self):
        super(MultiFusion, self).__init__()

        # embedding features preprocess
        self.conv_v_pro = nn.Conv2d(2048, out_dim, 1, 1)
        self.linear_t_pro = nn.Linear(4096, out_dim).cuda()

    def multi_fusion(self, embed_v, encod_tex):
        batch_size = embed_v.size(0)
        width = embed_v.size(2)
        height = embed_v.size(3)

        # Process visual before fusion
        x_v = embed_v / (embed_v.norm(p=2, dim=1, keepdim=True).expand_as(embed_v) + 1e-8)
        x_v = x_v.view(batch_size, 512, width * height)
        x_v = x_v.transpose(1,2)

        # Process texture before fusion
        x_t = encod_tex / (encod_tex.norm(p=2, dim=1, keepdim=True).expand_as(encod_tex) + 1e-8)
        x_t = F.dropout(x_t.cuda(), p=0.5)
        x_t = self.linear_t_pro(x_t.cuda())
        x_t = getattr(torch, 'tanh')(x_t)
        x_t = x_t.view(batch_size, 1, out_dim)
        x_t = x_t.expand(batch_size, width * height, out_dim)

        # feature embedding and compute attention weights
        vis_np =  embed_v.data.cpu().numpy()
        x = np.nonzero(vis_np)
        vis_nonzero = torch.from_numpy(vis_np[x]).long().cuda()

        tex_np = encod_tex.data.cpu().numpy()
        y = np.nonzero(tex_np)
        tex_nonzero = torch.from_numpy(tex_np[y]).long().cuda()

        attention_weights = Attention_weight(256, 512, x[0].shape[0], y[0].shape[0])
        att_weight = attention_weights(vis_nonzero, tex_nonzero)

        # multimodal fusion
        x_att = self.embeds_fusion(att_weight.cuda(), x_v.cuda(), x_t.cuda())
        # x_att = getattr(torch, 'tanh')(x_att)

        # Process attention vectors
        # x_att = F.dropout(x_att, p=0.5, training=self.training)
        x_att = x_att.view(batch_size, width, height, fused_dim)
        x_att = x_att.transpose(2, 3).transpose(1, 2)

        return x_att

    def forward(self, input_v, input_t):
        if input_v.dim() != 4 and input_t.dim() != 3:
            raise ValueError

        fusion_vec = self.multi_fusion(input_v.cuda(), input_t.cuda())

        return fusion_vec

class FusionNet(MultiFusion):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.embeds_fusion = Embeddings_Fusion()

    def fusion(self, embed_vis, encod_tex):
        return self.embeds_fusion(embed_vis.cuda(), encod_tex.cuda())
