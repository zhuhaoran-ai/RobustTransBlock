import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
import numpy as np
from kan import KAN
from Timesblock import TimesBlock

class Model(nn.Module):
    """
    Vanilla Transformer
    with O(L^2) complexity
    Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len=configs.seq_len
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        self.output_attention = configs.output_attention
        # Embedding
        self.enc_embedding2 = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.enc_embedding = DataEmbedding(configs.enc_in, 16, configs.embed, configs.freq,
                                           configs.dropout)
        self.model = nn.ModuleList([TimesBlock(configs)  # e_layers   默认为2, help='num of encoder layers'  有多个Timesblock     两个Timesblock
                                    for _ in range(configs.d_layers)])
        # Encoder
        self.d_model=configs.d_model
        self.layer_norm = nn.LayerNorm(16)#括号中的参数为通道数
        self.layer=configs.d_layers
        self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)#（96,192）
        self.projection = nn.Linear(
                16, configs.c_out)#由512变为7   是（48,52,512）乘以维度为512,7的W（权重）
        self.projection1 = nn.Linear(
                 self.seq_len+self.label_len,self.pred_len + self.seq_len)#由512变为7   是（48,52,512）乘以维度为512,7的W（权重）
        self.projection2 = nn.Linear(
               16, configs.d_model)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(2)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                               configs.dropout)
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AttentionLayer(
                            FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=False),
                            configs.d_model, configs.n_heads),
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(1)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
            )
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def forecast1(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Embedding
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc =x_enc/ stdev
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
           0, 2, 1)  # align temporal dimension
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))#进行多次timesblock
        enc_out = self.projection2(enc_out)
        
        #enc_out = self.projection(enc_out)              
        enc_out=enc_out[:, -self.seq_len:, :]
        #enc_out = self.projection1(enc_out)
        #enc_out = self.projection(enc_out)
        #enc_out = self.enc_embedding(enc_out, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        #dec_out = self.projection(enc_out)
        
        
        
        #print(enc_out.shape)
        
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        #print(dec_out.shape)
        dec_out = self.projection1(dec_out.permute(0, 2, 1)).permute(
           0, 2, 1)
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out
    def forecast2(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Embedding
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
           0, 2, 1)  # align temporal dimension
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))#进行多次timesblock
        #enc_out = self.projection2(enc_out)
        
        #enc_out = self.projection(enc_out)              
        #enc_out=enc_out[:, -self.seq_len:, :]
        #enc_out = self.projection1(enc_out)
        #enc_out = self.projection(enc_out)
        #enc_out = self.enc_embedding(enc_out, x_mark_enc)
        #enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        dec_out = self.projection(enc_out)
        
        
        
        #print(enc_out.shape)
        
        #dec_out = self.dec_embedding(x_dec, x_mark_dec)
        #dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        #print(dec_out.shape)
        #dec_out = self.projection1(dec_out.permute(0, 2, 1)).permute(
        #   0, 2, 1)
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out
    def forecast3(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Embedding
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        enc_out = self.enc_embedding2(x_enc, x_mark_enc)
        #enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
         #  0, 2, 1)  # align temporal dimension
        #for i in range(self.layer):
        #    enc_out = self.layer_norm(self.model[i](enc_out))#进行多次timesblock
        #enc_out = self.projection2(enc_out)
        
        #enc_out = self.projection(enc_out)              
        #enc_out=enc_out[:, -self.seq_len:, :]
        #enc_out = self.projection1(enc_out)
        #enc_out = self.projection(enc_out)
        #enc_out = self.enc_embedding(enc_out, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        #dec_out = self.projection(enc_out)
        
        
        
        #print(enc_out.shape)
        
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        #print(dec_out.shape)
        dec_out = self.projection1(dec_out.permute(0, 2, 1)).permute(
           0, 2, 1)
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out
    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out)
        return dec_out

    def anomaly_detection(self, x_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast1(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
