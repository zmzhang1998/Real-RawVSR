# Copyright (c) 2021 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn as nn
import models_DBSR.dbsr.encoders as dbsr_encoders
import models_DBSR.dbsr.decoders as dbsr_decoders
import models_DBSR.dbsr.merging as dbsr_merging
from models_DBSR.alignment.pwcnet import PWCNet



class DBSRNet(nn.Module):
    """ Deep Burst Super-Resolution model"""
    def __init__(self, encoder, merging, decoder):
        super().__init__()

        self.encoder = encoder      # Encodes input images and performs alignment
        self.merging = merging      # Merges the input embeddings to obtain a single feature map
        self.decoder = decoder      # Decodes the merged embeddings to generate HR RGB image

    def forward(self, im):
        out_enc = self.encoder(im)
        out_merge = self.merging(out_enc)
        out_dec = self.decoder(out_merge)

        return out_dec['pred'], {'offsets': out_enc['offsets'], 'fusion_weights': out_merge['fusion_weights']}


def dbsrnet_cvpr2021(enc_init_dim=64, enc_num_res_blocks=9, enc_out_dim=512,
                     dec_init_conv_dim=64, dec_num_pre_res_blocks=5, dec_post_conv_dim=32, dec_num_post_res_blocks=4,
                     upsample_factor=8, activation='relu', train_alignmentnet=False,
                     offset_feat_dim=64,
                     weight_pred_proj_dim=64,
                     num_offset_feat_extractor_res=1,
                     num_weight_predictor_res=1,
                     offset_modulo=1.0,
                     use_offset=True,
                     ref_offset_noise=0.0,
                     softmax=True,
                     use_base_frame=True,
                     icnrinit=True,
                     gauss_blur_sd=1.0,
                     gauss_ksz=3,
                     ):
    # backbone
    alignment_net = PWCNet(load_pretrained=True,
                           weights_path='models_DBSR/pwcnet-network-default.pth')

    encoder = dbsr_encoders.ResEncoderWarpAlignnet(enc_init_dim, enc_num_res_blocks, enc_out_dim,
                                                   alignment_net,
                                                   activation=activation,
                                                   train_alignmentnet=train_alignmentnet)

    merging = dbsr_merging.WeightedSum(enc_out_dim, weight_pred_proj_dim, offset_feat_dim,
                                       num_offset_feat_extractor_res=num_offset_feat_extractor_res,
                                       num_weight_predictor_res=num_weight_predictor_res,
                                       offset_modulo=offset_modulo,
                                       use_offset=use_offset,
                                       ref_offset_noise=ref_offset_noise,
                                       softmax=softmax, use_base_frame=use_base_frame)

    decoder = dbsr_decoders.ResPixShuffleConv(enc_out_dim, dec_init_conv_dim, dec_num_pre_res_blocks,
                                              dec_post_conv_dim, dec_num_post_res_blocks,
                                              upsample_factor=upsample_factor, activation=activation,
                                              gauss_blur_sd=gauss_blur_sd, icnrinit=icnrinit,
                                              gauss_ksz=gauss_ksz)

    net = DBSRNet(encoder=encoder, merging=merging, decoder=decoder)
    return net
