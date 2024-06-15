import numpy as np
import torch
import torch.nn as nn
import math
from diff_models_withAtten import diff_open




class CSDI_base(nn.Module):
    def __init__(self, target_dim, config, device):
        super().__init__()
        self.device = device
        self.target_dim = target_dim
        self.mse = nn.MSELoss()
        self.threshold = 0.5
        self.noisy_rate = 1e0 # 生成噪声项的幅度
        self.w = 0.2
        # 从配置中获取时间嵌入维度 (timeemb)，该维度用于生成时间嵌入。
        self.emb_time_dim = config["model"]["timeemb"]
        # 从配置中获取特征嵌入维度 (featureemb)，该维度用于生成特征嵌入。
        self.emb_feature_dim = config["model"]["featureemb"]


        # 计算总的嵌入维度
        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim

        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        # 配置扩散模型
        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 1
        self.diffmodel = diff_open(config_diff, input_dim)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat) # 累乘函数
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1)

    def time_embedding(self, pos, d_model=128):
        '''
        self, pos, d_model=128
        生成时间嵌入，通过对时间进行编码而得到向量表示
        返回生成的时间嵌入矩阵 pe,其中每一行代表一个时间步的嵌入向量,形状为 (B, L, d_model)
        '''
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
    
    def get_feature_pattern(self, observed_region, channel = 32):
        observed_region = observed_region.unsqueeze(-1)



        return observed_region
    
    def get_side_info(self, observed_tp, observed_data,observed_region):
        B, L = observed_data.shape
        _, E = observed_region.shape
        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb),生成时间嵌入

        # Feature Embedding
        feature_embed = self.get_feature_pattern(observed_region, channel = 32)  # (K,emb,1)
        feature_embeding = feature_embed.expand(B,E,L).permute(0,2,1)

        side_info = torch.cat([time_embed, feature_embeding], dim=-1) # B, L, emb_time_dim + emb_feature_dim


        return side_info

    def calc_loss_valid(
        self, observed_data, side_info, is_train
    ):
        '''
        计算在模型验证过程中的损失, 取多个时间步并计算均值
        '''
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, side_info, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
        self, observed_data, side_info, is_train, set_t=-1
    ):
        '''
        计算模型在给定时间步 t 下的损失
        '''
        B, L = observed_data.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise

        total_input = self.set_input_to_diffmodel(noisy_data)

        predicted = self.diffmodel(total_input, side_info, t)  # (B,L)

        loss = self.mse(noise, predicted)
                
        return loss

    def set_input_to_diffmodel(self, noisy_data):
        total_input = noisy_data  # (B,L)

        return total_input

    def impute(self, observed_data, side_info, n_samples):
        '''
        对观测数据进行多次插补，生成多个可能的插补样本
        返回包含生成的插补样本的张量
        '''
        B, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, L).to(self.device)
        side_info_uncond = side_info.clone()
        side_info_uncond[:,:,self.emb_time_dim] = 0

        for i in range(n_samples):

            current_sample = torch.randn_like(observed_data)

            for t in range(self.num_steps - 1, -1, -1):
                diff_input = current_sample
                predicted = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device))

                side_info[:, :, self.emb_time_dim:] = 0
                predicted_uncond = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device))
                predict_mean = (1 + self.w) * predicted - self.w * predicted_uncond
                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predict_mean)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise

            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_region,
            observed_tp,
            idex_test
        ) = self.process_data(batch)



        side_info = self.get_side_info(observed_tp, observed_data, observed_region)
        thresh = np.random.rand()
        if thresh < self.threshold:
            side_info[:,:,self.emb_time_dim:] = 0


        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, side_info, is_train)

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_region,
            observed_tp,
            idex_test
        ) = self.process_data(batch)

        with torch.no_grad():
            side_info = self.get_side_info(observed_tp, observed_data, observed_region)

            samples = self.impute(observed_data, side_info, n_samples)

        return samples, observed_data, observed_tp

class OpenDiff(CSDI_base):
    def __init__(self, config, device, target_dim=1):
        super(OpenDiff, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_region = batch["region_emb"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        idex_test = batch["idex_test"].to(self.device).int()



        return (
            observed_data,
            observed_region,
            observed_tp,
            idex_test
        )
