import os
import pandas as pd
import pickle
from models.modules import *


class Item_Embedding(nn.Module):
    def __init__(self, emb_pipline, item_num, is_hot=False,**key_words):
        super(Item_Embedding, self).__init__()
        data_statis = pd.read_pickle(os.path.join(key_words["language_embs_path"], 'data_statis.df'))

        self.item_num = item_num
        self.is_hot = is_hot

        self.construct_item_embeddings(emb_pipline, **key_words)

    # 根据不同模型生成对应的 item_embedding
    def construct_item_embeddings(self, emb_pipline, **key_words):
        if emb_pipline == "ID":
            self.init_ID_embedding(key_words["hidden_dim"], key_words["ID_embs_init_type"])
        elif emb_pipline == "SI":  # semantic initialization
            self.init_ID_embedding(key_words["hidden_dim"], "language_embeddings", **key_words)
        elif emb_pipline == "SR":  # semantic reconstruction
            self.init_ID_embedding(key_words["hidden_dim"], key_words["ID_embs_init_type"], **key_words)
            language_embs = self.load_language_embeddings(key_words["language_embs_path"],
                                                          key_words["language_model_type"],
                                                          key_words["language_embs_scale"])
            # padding_emb = np.random.rand(language_embs.shape[1])  # padding ID embedding
            # language_embs = np.vstack([language_embs, padding_emb])
            self.language_embeddings = nn.Embedding.from_pretrained(
                torch.tensor(language_embs, dtype=torch.float32),
                freeze=True,
            )
        elif emb_pipline == "Dual_view":  # Dual view modeling of LLNESR
            self.init_ID_embedding(key_words["hidden_dim"], "language_embeddings", **key_words)
            language_embs = self.load_language_embeddings(key_words["language_embs_path"],
                                                          key_words["language_model_type"],
                                                          key_words["language_embs_scale"])
            padding_emb = np.random.rand(language_embs.shape[1])  # padding ID embedding
            language_embs = np.vstack([language_embs, padding_emb])
            self.language_embeddings = nn.Embedding.from_pretrained(
                torch.tensor(language_embs, dtype=torch.float32),
                freeze=True,
                padding_idx=self.item_num
            )
        elif emb_pipline == "AP":  # Adaptive Projection -> MoRec/UniRec
            language_embs = self.load_language_embeddings(key_words["language_embs_path"],
                                                          key_words["language_model_type"],
                                                          key_words["language_embs_scale"])
            padding_emb = np.random.rand(language_embs.shape[1])  # padding ID embedding
            language_embs = np.vstack([language_embs, padding_emb])
            self.language_embeddings = nn.Embedding.from_pretrained(
                torch.tensor(language_embs, dtype=torch.float32),
                freeze=True,
                padding_idx=self.item_num
            )
        elif emb_pipline == "WAP":  # Adaptive Projection for whitened language embeddings
            key_words["item_frequency_flag"] = False
            key_words['standardization'] = True
            language_embs = self.semantic_space_decomposion(None, **key_words)
            padding_emb = np.random.rand(language_embs.shape[1])  # padding ID embedding
            language_embs = np.vstack([language_embs, padding_emb])
            self.language_embeddings = nn.Embedding.from_pretrained(
                torch.tensor(language_embs, dtype=torch.float32),
                freeze=True,
                padding_idx=self.item_num
            )
        elif emb_pipline == "AF":  # AlphaFuse
            # 裁剪的 语义嵌入
            cliped_language_embs = self.semantic_space_decomposion(key_words["hidden_dim"], **key_words)
            padding_emb = np.random.rand(cliped_language_embs.shape[1])  # padding ID embedding
            cliped_language_embs = np.vstack([cliped_language_embs, padding_emb])

            # AlphaFuse的做法是: 使用"semantic_space_decomposion"生成hidden_dim(128维)的语义信息
            self.language_embeddings = nn.Embedding.from_pretrained(
                torch.tensor(cliped_language_embs, dtype=torch.float32),
                freeze=True,
                padding_idx=self.item_num
            )
            # 并单纯使用nn.Embedding + ID_embs_init_type方式初始化 ID_embedding
            self.init_ID_embedding(self.nullity, key_words["ID_embs_init_type"])
            # self.init_ID_embedding(self.nullity, "zeros")

        elif emb_pipline == "BF":  # IBaSRec
            # 裁剪的 语义嵌入
            cliped_language_embs = self.semantic_space_decomposion(key_words["hidden_dim"], **key_words)
            padding_emb = np.random.rand(cliped_language_embs.shape[1])  # padding ID embedding
            cliped_language_embs = np.vstack([cliped_language_embs, padding_emb])
            if self.is_hot:
                self.language_embeddings = nn.Embedding.from_pretrained(
                    torch.tensor(cliped_language_embs, dtype=torch.float32),
                    freeze=False,
                    padding_idx=self.item_num
                )
            else:
                self.language_embeddings = nn.Embedding.from_pretrained(
                    torch.tensor(cliped_language_embs, dtype=torch.float32),
                    freeze=True,
                    padding_idx=self.item_num
                )
            self.language_dim = cliped_language_embs.shape[1]  # <-- add (关键)

            # self.init_ID_embedding(key_words["hidden_dim"], key_words["ID_embs_init_type"])

    def load_language_embeddings(self, directory, language_model_type, scale):
        # {dataset}_emb_matrix.pickle存储的是ndarray格式文件
        language_embs = pd.read_pickle(os.path.join(directory, language_model_type + '_emb_matrix.pickle'))

        self.item_num = len(language_embs)
        self.language_dim = len(language_embs[0])

        a = np.stack(language_embs)
        b = a * scale
        return b

    # id_embedding 初始化方式
    def init_ID_embedding(self, ID_dim, init_type, **key_words):
        # pipeline == 'SI' / semantic initialization
        if init_type == "language_embeddings":
            # 加载预处理好的语义嵌入
            language_embs = self.load_language_embeddings(key_words["language_embs_path"],
                                                          key_words["language_model_type"],
                                                          key_words["language_embs_scale"])
            # 如果 语义嵌入维度 == ID嵌入维度
            if self.language_dim == ID_dim:
                padding_emb = np.random.rand(language_embs.shape[1])  # padding ID embedding
                language_embs = np.vstack([language_embs, padding_emb])
                # language_embs = np.vstack([language_embs, padding_emb])
                self.ID_embeddings = nn.Embedding.from_pretrained(
                    torch.tensor(language_embs, dtype=torch.float32),
                    freeze=False,
                    padding_idx=self.item_num
                )
            # 通常来讲self.language_dim > ID_dim
            else:
                clipped_language_embs = self.semantic_space_decomposion(ID_dim, **key_words)

                padding_emb = np.random.rand(clipped_language_embs.shape[1])  # padding semantic embedding
                clipped_language_embs = np.vstack([clipped_language_embs, padding_emb])
                # language_embs = np.vstack([language_embs, padding_emb])
                self.ID_embeddings = nn.Embedding.from_pretrained(
                    torch.tensor(clipped_language_embs, dtype=torch.float32),
                    freeze=False,
                    padding_idx=self.item_num
                )
        else:
            # nn.Embedding 生成 id_embedding
            self.ID_embeddings = nn.Embedding(
                num_embeddings=self.item_num + 1,
                embedding_dim=ID_dim,
            )

            # 初始化id_embedding
            if init_type == "uniform":
                nn.init.uniform_(self.ID_embeddings.weight, a=0.0, b=1.0)
            # AlphaFuse使用的normal初始化方式
            elif init_type == "normal":
                nn.init.normal_(self.ID_embeddings.weight, 0, 1)
            elif init_type == "zeros":
                nn.init.zeros_(self.ID_embeddings.weight)
            elif init_type == "ortho":
                nn.init.orthogonal_(self.ID_embeddings.weight, gain=1.0)
            elif init_type == "xavier":
                nn.init.xavier_uniform_(self.ID_embeddings.weight, gain=1.0)
            elif init_type == "sparse":
                nn.init.sparse_(self.ID_embeddings.weight, 0.01, std=1)
            else:
                raise NotImplementedError("This kind of init for ID embeddings is not implemented yet.")

    # 对 语言嵌入做 SVD、裁剪、标准化
    def semantic_space_decomposion(self, clipped_dim, **key_words):
        language_embs = self.load_language_embeddings(key_words["language_embs_path"], key_words["language_model_type"],
                                                      key_words["language_embs_scale"])
        # default: False -> 不使用流行度
        if not key_words["item_frequency_flag"]:
            # The default item distribution is a uniform distribution.
            self.language_mean = np.mean(language_embs, axis=0)
            # 去中心化
            cov = np.cov(language_embs - self.language_mean, rowvar=False)
        else:
            items_pop = np.load(os.path.join(key_words["language_embs_path"], 'items_pop.npy'))
            items_freq_scale = 1.0 / items_pop.sum()
            items_freq = (items_pop * items_freq_scale).reshape(-1, 1)
            self.language_mean = np.sum(language_embs * items_freq, axis=0)
            cov = np.cov((language_embs - self.language_mean) * np.sqrt(items_freq), rowvar=False)
            # raise NotImplementedError("Custom item distribution is not implemented yet.")

        # 对斜方差矩阵进行svd处理,得到左奇异向量矩阵U和奇异值矩阵S
        U, S, _ = np.linalg.svd(cov, full_matrices=False)

        # None -> 矩阵奇异值的阈值
        if key_words["null_thres"] is not None:
            indices_null = np.where(S <= key_words["null_thres"])[0]
            self.nullity = len(indices_null)
        # null_dim 就是规定好的 null_space 的维度
        elif key_words["null_dim"] is not None:
            self.nullity = key_words["null_dim"]
        # print("The Nullity is", self.nullity)
        # self.squared_singular_values = S
        # self.language_bases = U
        if clipped_dim is None:
            clipped_dim = self.language_dim
        if key_words["cover"]:
            clipped_dim = clipped_dim - self.nullity

        # 奇异值分解 -> 截断矩阵
        Projection_matrix = U[..., :clipped_dim]

        # Whitening
        if key_words['standardization']:
            # 对E_language的奇异值矩阵进行-1/2 实现起来使用sqrt(1/S)更方便
            Diagnals = np.sqrt(1 / S)[:clipped_dim]
            Projection_matrix = Projection_matrix.dot(np.diag(Diagnals))  # V_{\lamda} into V_1
        clipped_language_embs = (language_embs - self.language_mean).dot(Projection_matrix)

        self.clipped_language_dim = clipped_language_embs.shape[1]  # <-- add

        return clipped_language_embs


class SASRec_backbone(nn.Module):
    def __init__(self, device, item_num, **key_words):
        super(SASRec_backbone, self).__init__()

        self.seq_len = key_words['max_len']
        self.item_num = item_num
        # self.item_embeddings = Item_Embedding("ID", **key_words)
        # self.item_num = item_num
        # self.seq_len = seq_len

        self.dropout = key_words["dropout_rate"]
        self.device = device
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        # self.language_dim = self.item_embeddings.language_dim
        self.hidden_dim = key_words["hidden_dim"]

        self.positional_embeddings = nn.Embedding(
            num_embeddings=self.seq_len,
            embedding_dim=self.hidden_dim
        )
        # emb_dropout is added
        self.emb_dropout = nn.Dropout(self.dropout)
        self.ln_1 = nn.LayerNorm(self.hidden_dim)
        self.ln_2 = nn.LayerNorm(self.hidden_dim)
        self.ln_3 = nn.LayerNorm(self.hidden_dim)

        # transformer block
        self.mh_attn = MultiHeadAttention(self.hidden_dim, self.hidden_dim, key_words["num_heads"], self.dropout)
        self.feed_forward = PositionwiseFeedForward(self.hidden_dim, self.hidden_dim, self.dropout)
        # self.s_fc = nn.Linear(self.hidden_size, self.item_num)
        # self.ac_func = nn.ReLU()

    # 不同模型实现不同的接口 -> AlphaFuse的实现就是row space + null space
    def embed_ID(self, x):
        # return self.item_embeddings.ID_embeddings(x)
        pass

    def return_item_emb(self, ):
        # return self.item_embeddings.ID_embeddings.weight
        pass

    def forward(self, sequences):
        """
            parameters:
                sequences: user-interaction
            return:
                logits: actually is the representation of user-behaviour
        """
        inputs_emb = self.embed_ID(sequences)
        inputs_emb += self.positional_embeddings(torch.arange(self.seq_len).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(sequences, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        logits = ff_out[:, -1].squeeze()
        return logits

    def predict(self, sequences):
        # inputs_emb = self.item_embeddings(states) * self.item_embeddings.embedding_dim ** 0.5
        state_hidden = self.forward(sequences)
        item_embs = self.return_item_emb()
        scores = torch.matmul(state_hidden, item_embs[:-1].transpose(0, 1))  # (B,|I|)
        return scores

    def calculate_ce_loss(self, sequences, target):
        seq_output = self.forward(sequences)
        item_embs = self.return_item_emb()  # (|I|,d)
        # item_embs = self.item_emb.return_embs()
        logits = torch.matmul(seq_output, item_embs[:-1].transpose(0, 1))
        loss = self.ce_loss(logits, target)
        return loss

    def calculate_bce_loss(self, sequences, target, neg_ratio, emb_type="both"):
        """
            sequence: 输入的Interaction sequence
            target: ground truth target
            neg_ratio: 每个正样本对应的负样本的个数
        """
        # negative sampling
        # 生成与正样本对应的负样本目标张量
        # 首先生成所有可能的负样本
        # sequences_set = set(sequences.view(-1).tolist())
        batch_size = target.shape[0]
        neg_samples = torch.randint(0, self.item_num, (batch_size, neg_ratio))
        expanded_target = target.view(batch_size, 1).expand(batch_size, neg_ratio).cpu()
        # expanded_sequences = sequences.view(batch_size, -1, 1).expand(batch_size, sequences.shape[1], neg_ratio).cpu()
        # mask_target = neg_samples == expanded_target
        # mask_sequences = (neg_samples.unsqueeze(1).expand(-1, sequences.shape[1], -1) == expanded_sequences).any(dim=1)
        # mask = mask_target | mask_sequences
        mask = neg_samples == expanded_target
        while mask.any():
            new_samples = torch.randint(0, self.item_num, (batch_size, neg_ratio))
            neg_samples = torch.where(mask, new_samples, neg_samples)
            mask = neg_samples == expanded_target
            # mask_target = neg_samples == expanded_target
            # mask_sequences = (neg_samples.unsqueeze(1).expand(-1, sequences.shape[1], -1) == expanded_sequences).any(dim=1)
            # mask = mask_target | mask_sequences
        target_neg = neg_samples.to(target.device)

        # pos_embs = self.item_embeddings(target)
        pos_embs = self.embed_ID(target)
        neg_embs = self.embed_ID(target_neg)

        log_feats = self.forward(sequences)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats.unsqueeze(1) * neg_embs).sum(dim=-1)

        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=self.device), torch.zeros(neg_logits.shape,
                                                                                               device=self.device)
        loss = self.bce_loss(pos_logits, pos_labels)
        loss += self.bce_loss(neg_logits, neg_labels)

        return loss

    def calculate_infonce_loss(self, sequences, target, neg_ratio, temperature, emb_type="both"):

        # negative sampling
        # 生成与正样本对应的负样本目标张量
        # 首先生成所有可能的负样本
        # sequences_set = set(sequences.view(-1).tolist())
        batch_size = target.shape[0]
        neg_samples = torch.randint(0, self.item_num, (batch_size, neg_ratio))
        expanded_target = target.view(batch_size, 1).expand(batch_size, neg_ratio).cpu()
        expanded_sequences = sequences.view(batch_size, -1, 1).expand(batch_size, sequences.shape[1], neg_ratio).cpu()
        # mask_target = neg_samples == expanded_target
        # mask_sequences = (neg_samples.unsqueeze(1).expand(-1, sequences.shape[1], -1) == expanded_sequences).any(dim=1)
        # mask = mask_target | mask_sequences
        mask = neg_samples == expanded_target
        while mask.any():
            new_samples = torch.randint(0, self.item_num, (batch_size, neg_ratio))
            neg_samples = torch.where(mask, new_samples, neg_samples)
            mask = neg_samples == expanded_target
            # mask_target = neg_samples == expanded_target
            # mask_sequences = (neg_samples.unsqueeze(1).expand(-1, sequences.shape[1], -1) == expanded_sequences).any(dim=1)
            # mask = mask_target | mask_sequences
        target_neg = neg_samples.to(target.device)

        # pos_embs = self.item_embeddings(target)
        pos_embs = self.embed_ID(target)
        neg_embs = self.embed_ID(target_neg)
        log_feats = self.forward(sequences)

        log_feats = F.normalize(log_feats, p=2, dim=-1)
        pos_embs = F.normalize(pos_embs, p=2, dim=-1)
        neg_embs = F.normalize(neg_embs, p=2, dim=-1)
        # normed_log_feats = log_feats / torch.sqrt(1e-8 + log_feats.square().sum(-1, keepdim=True))
        # normed_pos_embs = pos_embs / torch.sqrt(1e-8 + pos_embs.square().sum(-1, keepdim=True))
        # normed_neg_embs = neg_embs / torch.sqrt(1e-8 + neg_embs.square().sum(-1, keepdim=True))

        pos_logits = (log_feats * pos_embs).sum(dim=-1, keepdim=True)
        neg_logits = torch.bmm(neg_embs, log_feats.unsqueeze(-1)).squeeze(-1)

        logits = torch.cat([pos_logits, neg_logits], dim=-1)
        logits /= temperature

        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)  # (batch_size,)
        loss = F.cross_entropy(logits, labels)
        return loss


### pure ID embeddings
class SASRec(SASRec_backbone):
    def __init__(self, device, item_num, **key_words):
        super().__init__(device, item_num, **key_words)

        self.item_embeddings = Item_Embedding("ID", item_num, **key_words)

    def embed_ID(self, x):
        return self.item_embeddings.ID_embeddings(x)

    def return_item_emb(self, ):
        return self.item_embeddings.ID_embeddings.weight


class MoRec(SASRec_backbone):
    def __init__(self, device, item_num, **key_words):
        super().__init__(device, item_num, **key_words)

        self.item_embeddings = Item_Embedding("AP", item_num, **key_words)
        self.language_dim = self.item_embeddings.language_dim
        self.adapter = nn.Sequential(
            nn.Linear(self.language_dim, key_words['hidden_dim']),
            nn.GELU()
        )

    def embed_ID(self, x):
        language_embs = self.item_embeddings.language_embeddings(x)
        return self.adapter(language_embs)

    def return_item_emb(self, ):
        language_embs = self.item_embeddings.language_embeddings.weight
        return self.adapter(language_embs)


class WhitenRec(SASRec_backbone):
    def __init__(self, device, item_num, **key_words):
        super().__init__(device, item_num, **key_words)

        self.item_embeddings = Item_Embedding("WAP", item_num, **key_words)
        self.language_dim = self.item_embeddings.language_dim
        self.adapter = nn.Sequential(
            nn.Linear(self.language_dim, key_words['hidden_dim']),
            nn.GELU()
        )

    def embed_ID(self, x):
        language_embs = self.item_embeddings.language_embeddings(x)
        return self.adapter(language_embs)

    def return_item_emb(self, ):
        language_embs = self.item_embeddings.language_embeddings.weight
        return self.adapter(language_embs)


class LLMInit(SASRec_backbone):
    # 使用E_language并进行适当裁剪，使其初始化为 ID_embedding
    # id_embedding 不冻结，保持梯度更新

    def __init__(self, device, item_num, **key_words):
        super().__init__(device, item_num, **key_words)

        self.item_embeddings = Item_Embedding("SI", item_num, **key_words)
        # self.language_dim = self.item_embeddings.language_dim

    def embed_ID(self, x):
        return self.item_embeddings.ID_embeddings(x)

    def return_item_emb(self, ):
        return self.item_embeddings.ID_embeddings.weight


class RLMRec(SASRec_backbone):
    def __init__(self, device, item_num, **key_words):
        super().__init__(device, item_num, **key_words)

        self.item_embeddings = Item_Embedding("SR", item_num, **key_words)
        self.language_dim = self.item_embeddings.language_dim
        if key_words['SR_aligement_type'] == 'con':
            self.reconstructor = nn.Sequential(
                nn.Linear(self.language_dim, (self.language_dim + key_words['hidden_dim']) // 2),
                nn.LeakyReLU(),
                nn.Linear((self.language_dim + key_words['hidden_dim']) // 2, key_words['hidden_dim'])
            )
        elif key_words['SR_aligement_type'] == 'gen':
            self.reconstructor = nn.Sequential(
                nn.Linear(key_words['hidden_dim'], (self.language_dim + key_words['hidden_dim']) // 2),
                nn.LeakyReLU(),
                nn.Linear((self.language_dim + key_words['hidden_dim']) // 2, self.language_dim)
            )

    def embed_ID(self, x):
        return self.item_embeddings.ID_embeddings(x)

    def return_item_emb(self, ):
        return self.item_embeddings.ID_embeddings.weight

    def reconstruct_gen_loss(self, ):
        rec_language_embs = self.reconstructor(
            self.return_item_emb()[:-1])  # self.return_item_emb()[-1] is the padding embedding
        language_embs = self.item_embeddings.language_embeddings.weight
        rec_language_embs = F.normalize(rec_language_embs, p=2, dim=-1)
        language_embs = F.normalize(language_embs, p=2, dim=-1)
        return 1 - (rec_language_embs * language_embs).sum() / self.item_num

    def reconstruct_con_loss(self, ):
        language_embs = self.item_embeddings.language_embeddings.weight
        rec_ID_embs = self.reconstructor(language_embs)  # self.return_item_emb()[-1] is the padding embedding
        ID_embs = self.return_item_emb()[:-1]
        rec_ID_embs = F.normalize(rec_ID_embs, p=2, dim=-1)
        ID_embs = F.normalize(ID_embs, p=2, dim=-1)
        return 1 - (rec_ID_embs * ID_embs).sum() / self.item_num


class UniSRec(SASRec_backbone):
    # UniSRec的实现过程中，language_embeddings不参与梯度更新
    def __init__(self, device, item_num, **key_words):
        super().__init__(device, item_num, **key_words)

        self.item_embeddings = Item_Embedding("AP", item_num, **key_words)
        self.language_dim = self.item_embeddings.language_dim
        self.adapter = MoEAdaptorLayer(
            8,
            [self.language_dim, key_words['hidden_dim']],
            0.2
        )

    def embed_ID(self, x):
        language_embs = self.item_embeddings.language_embeddings(x)
        # UniRec均使用MoE自适应层将language_embedding -> id_embedding
        return self.adapter(language_embs)

    def return_item_emb(self, ):
        language_embs = self.item_embeddings.language_embeddings.weight
        return self.adapter(language_embs)


class LLMESR(SASRec_backbone):
    def __init__(self, device, item_num, **key_words):
        super().__init__(device, item_num, **key_words)

        self.item_embeddings = Item_Embedding("Dual_view", item_num, **key_words)
        self.language_dim = self.item_embeddings.language_dim
        self.adapter = nn.Sequential(
            nn.Linear(self.language_dim, int(self.language_dim / 2)),
            nn.Linear(int(self.language_dim / 2), key_words['hidden_dim'])
        )

        self.language2ID = Multi_CrossAttention(self.hidden_dim, self.hidden_dim, 2)
        self.ID2language = Multi_CrossAttention(self.hidden_dim, self.hidden_dim, 2)

        self.reg = Contrastive_Loss2()
        self._init_sd(**key_words)

    def _init_sd(self, **key_words):
        """
        Init Retrieval Augmented Self-Distillation resources (ndarray version).
        Expect:
          - {dataset_dir}/{dataset_name}/sim_user_100.pkl   -> np.ndarray (user_num, 100)
          - {dataset_dir}/{dataset_name}/train_data.df      -> pandas df pickle with columns: user_id, seq
          - {dataset_dir}/{dataset_name}/data_statis.df     -> pandas df pickle with user_num
        Build:
          - self.sim_user_100: (user_num, K) LongTensor
          - self.user_seq_table: (user_num, seq_len) LongTensor (pad=item_num)
        """

        # ---- switches ----
        self.lambda_sd = float(key_words.get("beta", 0.0))
        self.sd_sample_k = int(key_words.get("sd_sample_k", 1))
        self.sd_detach_teacher = bool(key_words.get("sd_detach_teacher", True))

        if self.lambda_sd <= 0:
            return

        base = key_words.get("language_embs_path", None)
        if base is None:
            raise ValueError("[L_sd] need key_words['language_embs_path'] = ../dataset/{dataname}")

        sim_path = os.path.join(base, "sim_user_100.pkl")
        train_path = os.path.join(base, "train_data.df")
        stat_path = os.path.join(base, "data_statis.df")

        for p in [sim_path, train_path, stat_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"[L_sd] missing file: {p}")

        # ---- user_num ----
        stat = pd.read_pickle(stat_path)
        if "user_num" not in stat.columns:
            raise KeyError(f"[L_sd] data_statis.df has no 'user_num'. columns={list(stat.columns)}")
        user_num = int(stat["user_num"].iloc[0])

        # ---- load sim_user_100 (ndarray) ----
        with open(sim_path, "rb") as f:
            sim_mat = pickle.load(f)  # np.ndarray (user_num, K)

        sim_mat = np.asarray(sim_mat)
        if sim_mat.ndim != 2:
            raise ValueError(f"[L_sd] sim_user_100 must be 2D, got shape={sim_mat.shape}")
        if sim_mat.shape[0] != user_num:
            raise ValueError(f"[L_sd] sim_user_100 first dim={sim_mat.shape[0]} != user_num={user_num}")

        sim = torch.as_tensor(sim_mat, dtype=torch.long)  # (user_num, K)
        K = sim.size(1)

        # ---- build user_seq_table from train_data.df ----
        pad_id = self.item_num
        user_seq = torch.full((user_num, self.seq_len), pad_id, dtype=torch.long)

        train_df = pd.read_pickle(train_path)
        if "user_id" not in train_df.columns or "seq" not in train_df.columns:
            raise KeyError(f"[L_sd] train_data.df needs columns ['user_id','seq'], got {list(train_df.columns)}")

        for _, row in train_df.iterrows():
            u = int(row["user_id"])
            if u < 0 or u >= user_num:
                continue

            seq = row["seq"]
            if not isinstance(seq, (list, tuple)):
                try:
                    seq = list(seq)
                except Exception:
                    continue

            seq = list(seq)[-self.seq_len:]
            if len(seq) == 0:
                continue

            user_seq[u, :len(seq)] = torch.tensor(seq, dtype=torch.long)

        # ---- buffers ----
        self.register_buffer("sim_user_100", sim, persistent=False)
        self.register_buffer("user_seq_table", user_seq, persistent=False)

    def get_state(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        返回用于 next-item prediction 的 user state 表征
        对 SASRec/GRU4Rec：forward(sequences) 已经是 (B, 2H)
        """
        return self.forward(sequences)

    def sd_loss(self, user_ids: torch.Tensor, sequences: torch.Tensor,
                student_state: torch.Tensor = None) -> torch.Tensor:
        if (self.lambda_sd <= 0) or (self.sim_user_100 is None) or (self.user_seq_table is None):
            return torch.tensor(0.0, device=sequences.device)

        device = sequences.device
        B = user_ids.size(0)

        pool = self.sim_user_100[user_ids]

        if pool.size(1) > 1:
            pool = torch.where(pool == user_ids.unsqueeze(1), pool[:, 1:2].expand_as(pool), pool)

        K = min(self.sd_sample_k, pool.size(1))
        ridx = torch.randint(0, pool.size(1), (B, K), device=device)
        sim_ids = torch.gather(pool.to(device), 1, ridx)  # (B, K)

        sim_seq = self.user_seq_table[sim_ids.reshape(-1)].to(device)

        with torch.no_grad():
            t_all = self.get_state(sim_seq)  # (B*K, D)
            t_all = t_all.view(B, K, -1)  # (B, K, D)
            t_mean = t_all.mean(dim=1)  # (B, D)

            t_mean = t_mean.detach()

        s = student_state if student_state is not None else self.get_state(sequences)  # (B, D)

        loss_fct = torch.nn.MSELoss()
        loss = loss_fct(s, t_mean)

        return self.lambda_sd * loss

    def embed_ID_text(self, x):
        language_embs = self.item_embeddings.language_embeddings(x)
        ID_embs = self.item_embeddings.ID_embeddings(x)
        return ID_embs, self.adapter(language_embs)

    def embed_ID(self, x):
        ID_embs, language_embs = self.embed_ID_text(x)
        return torch.cat([ID_embs, language_embs], dim=-1)

    def return_item_emb(self, ):
        ID_embs = self.item_embeddings.ID_embeddings.weight
        language_embs = self.item_embeddings.language_embeddings.weight
        language_embs = self.adapter(language_embs)
        return torch.cat([ID_embs, language_embs], dim=-1)

    def forward(self, sequences):
        inputs_id_emb, inputs_text_emb = self.embed_ID_text(sequences)
        inputs_text_emb += self.positional_embeddings(torch.arange(self.seq_len).to(self.device))
        inputs_id_emb += self.positional_embeddings(torch.arange(self.seq_len).to(self.device))

        text_seq = self.emb_dropout(inputs_text_emb)
        id_seq = self.emb_dropout(inputs_id_emb)

        cross_id_seqs = self.language2ID(text_seq, id_seq, sequences, self.item_num)
        cross_text_seqs = self.ID2language(id_seq, text_seq, sequences, self.item_num)
        cross_id_seqs = 1 * cross_id_seqs + 0 * id_seq
        cross_text_seqs = 1 * cross_text_seqs + 0 * text_seq

        # ID channel backbone.forward()
        mask = torch.ne(sequences, self.item_num).float().unsqueeze(-1).to(self.device)
        cross_id_seqs *= mask
        seq_normalized = self.ln_1(cross_id_seqs)
        mh_attn_out = self.mh_attn(seq_normalized, cross_id_seqs)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        id_logits = ff_out[:, -1].squeeze()

        # text channel backbone.forward()
        # mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        cross_text_seqs *= mask
        seq_normalized = self.ln_1(cross_text_seqs)
        mh_attn_out = self.mh_attn(seq_normalized, cross_text_seqs)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        text_logits = ff_out[:, -1].squeeze()

        log_feats = torch.cat([id_logits, text_logits], dim=-1)

        return log_feats

    # def reg_loss(self, sequences):
    #     unfold_item_id = torch.masked_select(sequences, sequences != self.item_num)
    #     id_emb, language_emb = self.embed_ID_text(unfold_item_id)
    #     reg_loss = self.reg(language_emb, id_emb)
    #     return reg_loss


class AlphaFuse(SASRec_backbone):
    def __init__(self, device, item_num, **key_words):
        super().__init__(device, item_num, **key_words)

        self.item_embeddings = Item_Embedding("AF", item_num, **key_words)
        # self.language_dim = self.item_embeddings.language_dim
        self.nullity = self.item_embeddings.nullity
        self.cover = key_words["cover"]

    def embed_ID(self, x):
        """
            x: item1, item2, item10
        """
        language_embs = self.item_embeddings.language_embeddings(x)
        # fuse_embs = language_embs.clone()
        ID_embs = self.item_embeddings.ID_embeddings(x)
        if self.cover:
            return torch.cat((language_embs, ID_embs), dim=-1)
        else:
            fuse_embs = language_embs.clone()
            fuse_embs[..., -self.nullity:] = language_embs[..., -self.nullity:] + ID_embs
        return fuse_embs

    def return_item_emb(self, ):
        language_embs = self.item_embeddings.language_embeddings.weight
        # fuse_embs = language_embs.clone()
        ID_embs = self.item_embeddings.ID_embeddings.weight
        if self.cover:
            return torch.cat((language_embs, ID_embs), dim=-1)
        else:
            fuse_embs = language_embs.clone()
            fuse_embs[..., -self.nullity:] = language_embs[..., -self.nullity:] + ID_embs
        return fuse_embs


class IBaSRec(SASRec_backbone):
    """
    Semantic-preserving fine-tuning + Behaviour residual

    e_sem(i) = e_lang0(i) + Δ(i)              (Δ is trainable, e_lang0 fixed)
    e_beh(i) = e_beh_emb(i)                  (trainable, learns CF/exposure bias)
    e(i)     = e_sem(i) + gate(i) * e_beh(i) (same hidden_dim, no capacity cut)
    """

    def __init__(self, device, item_num, **key_words):
        super().__init__(device, item_num, **key_words)

        self.device = device
        self.item_num = item_num
        self.hidden_dim = key_words["hidden_dim"]
        self.is_hot = False
        if key_words["predict_mode"] == "hot":
            self.is_hot = True

        self.item_embeddings = Item_Embedding("BF", item_num, self.is_hot, **key_words)
        # (item_num+1, hidden_dim), include padding row at index item_num

        self.lang0 = self.item_embeddings.language_embeddings

        init_type = key_words["ID_embs_init_type"]

        # ---- behaviour residual embedding ----
        # beh_emb其实就是Item_Embedding中init_embedding的实现，但是为了方便修改，单独拿出来init一下
        self.beh_emb = nn.Embedding(
            num_embeddings=self.item_num + 1,
            embedding_dim=self.hidden_dim,
            padding_idx=self.item_num
        )
        # 初始化id_embedding
        if init_type == "uniform":
            nn.init.uniform_(self.beh_emb.weight, a=0.0, b=1.0)
        elif init_type == "normal":
            # nn.init.normal_(self.beh_emb.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.beh_emb.weight, 0, 1)
        elif init_type == "zeros":
            nn.init.zeros_(self.beh_emb.weight)
        elif init_type == "ortho":
            nn.init.orthogonal_(self.beh_emb.weight, gain=1.0)
        elif init_type == "xavier":
            nn.init.xavier_uniform_(self.beh_emb.weight, gain=1.0)
        elif init_type == "sparse":
            nn.init.sparse_(self.beh_emb.weight, 0.01, std=1)
        else:
            raise NotImplementedError("This kind of init for ID embeddings is not implemented yet.")

        # ---- semantic fine-tuning residual delta ----
        self.sem_delta = nn.Embedding(
            num_embeddings=self.item_num + 1,
            embedding_dim=self.hidden_dim,
            padding_idx=self.item_num
        )
        nn.init.zeros_(self.sem_delta.weight)  # start from pure lang0

        # ---- item-wise gate: gate(i) in (0,1) ----
        self.gate = nn.Embedding(
            num_embeddings=self.item_num + 1,
            embedding_dim=1,
            padding_idx=self.item_num
        )
        # start with small behaviour contribution: sigmoid(-2) ~ 0.12 -> rely on semantic_embedding at first
        nn.init.constant_(self.gate.weight, -2.0)
        # nn.init.constant_(self.gate.weight, 1.0)


        self.beh_scale = float(key_words.get("beh_scale", 1.0))

        # ---- ablation setting ----
        self.predict_mode = key_words['predict_mode']

    # ---------- core embedding ----------
    def _get_item_components(self, ids: torch.Tensor):
        """
        ids: (...,) long
        return:
          e_sem: (..., D)
          e_beh: (..., D)
          g:     (..., 1) in (0,1)
        """
        e_lang0 = self.lang0(ids)  # fixed
        delta = self.sem_delta(ids)  # trainable
        e_sem = e_lang0 + delta

        e_beh = self.beh_emb(ids)
        g = torch.sigmoid(self.gate(ids))  # (..,1)

        return e_sem, e_beh, g, e_lang0

    def embed_ID(self, x):
        """
        x: item ids tensor (B,L) or (B,) or (B,neg)
        output: (..., hidden_dim)
        """
        if self.predict_mode == 'beh':
            return self.beh_emb(x)

        elif self.predict_mode == 'sem':
            e_lang0 = self.lang0(x)
            delta = self.sem_delta(x)
            return e_lang0 + delta

        elif self.predict_mode == 'lang':
            return self.lang0(x)

        else:
            # both和hot都走这里
            e_sem, e_beh, g, e_lang0 = self._get_item_components(x)
            e = e_sem + (self.beh_scale * g) * e_beh
            return e

    def return_item_emb(self):
        ids = torch.arange(self.item_num + 1, device=self.device, dtype=torch.long)
        return self.embed_ID(ids)

# ============ pre_ex =============

class StaticFuse(SASRec_backbone):
    """
    前置探针模型：独立实现，不继承 IBaSRec。
    只包含：
    1. 冻结的 SVD 语义 (e_lang)
    2. 可学习的 ID (e_beh)
    3. 静态系数 (alpha)
    """

    def __init__(self, device, item_num, static_alpha=0.5, **key_words):
        super().__init__(device, item_num, **key_words)
        self.device = device
        self.item_num = item_num
        self.static_alpha = static_alpha

        # 1. 复用 Item_Embedding("BF") 来加载和处理 SVD 语义
        # 这会保证我们用的是和 IBaSRec 完全一样的语义锚点
        # 这里的 item_embeddings 只用来提供 SVD 后的语义向量
        tmp_emb = Item_Embedding("BF", item_num, **key_words)

        # 提取权重并注册为 buffer (不参与训练，单纯作为常量)
        lang_weight = tmp_emb.language_embeddings.weight.detach().clone()
        self.register_buffer("lang0", lang_weight)

        # 2. 独立定义行为 ID Embedding
        self.beh_emb = nn.Embedding(
            num_embeddings=self.item_num + 1,
            embedding_dim=key_words["hidden_dim"],
            padding_idx=self.item_num
        )
        # 初始化 (保持和 IBaSRec 一致)
        nn.init.normal_(self.beh_emb.weight, 0, 1)
        with torch.no_grad():
            self.beh_emb.weight[self.item_num].zero_()

    def embed_ID(self, x):
        # 1. 获取冻结的 SVD 语义
        e_lang = self.lang0[x]

        # 2. 获取 ID
        e_beh = self.beh_emb(x)

        # 3. 静态融合
        return e_lang + self.static_alpha * e_beh

    def return_item_emb(self):
        ids = torch.arange(self.item_num + 1, device=self.device, dtype=torch.long)
        return self.embed_ID(ids)