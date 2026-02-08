import os
import pickle
import numpy as np
import pandas as pd

# from SASmodules import SASRec
from models.bert_modules import *


class Item_Embedding(nn.Module):
    def __init__(self, emb_pipline, item_num, is_hot=False, **key_words):
        super(Item_Embedding, self).__init__()
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

            # SR格式下的 language_embs 是否有问题 -> 没问题 RLMRec只用到实际item_num大小的language_emb
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
            padding_emb = np.zeros((1, language_embs.shape[1]), dtype=np.float32)  # padding ID embedding
            mask_token_emb = np.zeros((1, language_embs.shape[1]), dtype=np.float32)

            language_embs = np.vstack([language_embs, padding_emb, mask_token_emb])

            self.language_embeddings = nn.Embedding.from_pretrained(
                torch.tensor(language_embs, dtype=torch.float32),
                freeze=True,
                padding_idx=self.item_num
            )

            with torch.no_grad():
                self.language_embeddings.weight[self.item_num].zero_()  # pad
                self.language_embeddings.weight[self.item_num + 1].zero_()  # mask

        elif emb_pipline == "AP":  # Adaptive Projection -> MoRec/UniRec
            language_embs = self.load_language_embeddings(key_words["language_embs_path"],
                                                          key_words["language_model_type"],
                                                          key_words["language_embs_scale"])
            padding_emb = np.zeros((1, language_embs.shape[1]), dtype=np.float32)  # padding ID embedding
            mask_token_emb = np.zeros((1, language_embs.shape[1]), dtype=np.float32)

            language_embs = np.vstack([language_embs, padding_emb, mask_token_emb])

            self.language_embeddings = nn.Embedding.from_pretrained(
                torch.tensor(language_embs, dtype=torch.float32),
                freeze=True,
                padding_idx=self.item_num
            )

            with torch.no_grad():
                self.language_embeddings.weight[self.item_num].zero_()  # pad
                self.language_embeddings.weight[self.item_num + 1].zero_()  # mask

        elif emb_pipline == "WAP":  # Adaptive Projection for whitened language embeddings
            key_words["item_frequency_flag"] = False
            key_words['standardization'] = True
            language_embs = self.semantic_space_decomposion(None, **key_words)
            padding_emb = np.zeros((1, language_embs.shape[1]), dtype=np.float32)  # padding ID embedding
            mask_token_emb = np.zeros((1, language_embs.shape[1]), dtype=np.float32)

            language_embs = np.vstack([language_embs, padding_emb, mask_token_emb])

            self.language_embeddings = nn.Embedding.from_pretrained(
                torch.tensor(language_embs, dtype=torch.float32),
                freeze=True,
                padding_idx=self.item_num
            )

            with torch.no_grad():
                self.language_embeddings.weight[self.item_num].zero_()  # pad
                self.language_embeddings.weight[self.item_num + 1].zero_()  # mask

        elif emb_pipline == "AF":  # AlphaFuse
            # 裁剪的 语义嵌入
            cliped_language_embs = self.semantic_space_decomposion(key_words["hidden_dim"], **key_words)
            padding_emb = np.zeros((1, cliped_language_embs.shape[1]), dtype=np.float32)  # padding ID embedding
            mask_token_emb = np.zeros((1, cliped_language_embs.shape[1]), dtype=np.float32)

            cliped_language_embs = np.vstack([cliped_language_embs, padding_emb, mask_token_emb])

            # AlphaFuse的做法是: 使用"semantic_space_decomposion"生成hidden_dim(128维)的语义信息
            self.language_embeddings = nn.Embedding.from_pretrained(
                torch.tensor(cliped_language_embs, dtype=torch.float32),
                freeze=True,
                padding_idx=self.item_num
            )
            # 并单纯使用nn.Embedding + ID_embs_init_type方式初始化 ID_embedding
            self.init_ID_embedding(self.nullity, key_words["ID_embs_init_type"])
            # self.init_ID_embedding(self.nullity, "zeros")

            with torch.no_grad():
                self.language_embeddings.weight[self.item_num].zero_()  # pad
                self.language_embeddings.weight[self.item_num + 1].zero_()  # mask

        elif emb_pipline == "BF":  # IBaSRec
            # 裁剪的 语义嵌入
            cliped_language_embs = self.semantic_space_decomposion(key_words["hidden_dim"], **key_words)
            padding_emb = np.zeros((1, cliped_language_embs.shape[1]), dtype=np.float32)  # padding ID embedding
            mask_token_emb = np.zeros((1, cliped_language_embs.shape[1]), dtype=np.float32)

            cliped_language_embs = np.vstack([cliped_language_embs, padding_emb, mask_token_emb])

            # AlphaFuse的做法是: 使用"semantic_space_decomposion"生成hidden_dim(128维)的语义信息
            if self.is_hot:
                self.language_embeddings = nn.Embedding.from_pretrained(
                    torch.tensor(cliped_language_embs, dtype=torch.float32),
                    freeze=False,
                    padding_idx=self.item_num
                )
            else:
                # AlphaFuse的做法是: 使用"semantic_space_decomposion"生成hidden_dim(128维)的语义信息
                self.language_embeddings = nn.Embedding.from_pretrained(
                    torch.tensor(cliped_language_embs, dtype=torch.float32),
                    freeze=True,
                    padding_idx=self.item_num
                )

            with torch.no_grad():
                self.language_embeddings.weight[self.item_num].zero_()  # pad
                self.language_embeddings.weight[self.item_num + 1].zero_()  # mask

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
                padding_emb = np.zeros((1, language_embs.shape[1]), dtype=np.float32)  # padding ID embedding
                mask_token_emb = np.zeros((1, language_embs.shape[1]), dtype=np.float32)

                language_embs = np.vstack([language_embs, padding_emb, mask_token_emb])

                # language_embs = np.vstack([language_embs, padding_emb])
                self.ID_embeddings = nn.Embedding.from_pretrained(
                    torch.tensor(language_embs, dtype=torch.float32),
                    freeze=False,
                    padding_idx=self.item_num
                )
            # 通常来讲self.language_dim > ID_dim
            else:
                clipped_language_embs = self.semantic_space_decomposion(ID_dim, **key_words)

                padding_emb = np.zeros((1, clipped_language_embs.shape[1]), dtype=np.float32)  # padding semantic embedding
                mask_token_emb = np.zeros((1, clipped_language_embs.shape[1]), dtype=np.float32)

                clipped_language_embs = np.vstack([clipped_language_embs, padding_emb, mask_token_emb])

                # language_embs = np.vstack([language_embs, padding_emb])
                self.ID_embeddings = nn.Embedding.from_pretrained(
                    torch.tensor(clipped_language_embs, dtype=torch.float32),
                    freeze=False,
                    padding_idx=self.item_num
                )
        else:
            # Bert4Rec有两个特殊token: padding_token/mask_token
            self.ID_embeddings = nn.Embedding(
                num_embeddings=self.item_num + 2,
                embedding_dim=ID_dim,
                padding_idx=self.item_num
            )

            # 1) 先初始化整个矩阵
            if init_type == "uniform":
                nn.init.uniform_(self.ID_embeddings.weight, a=0.0, b=1.0)
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

            # 2) 再把 padding 行强制置 0（永远别让 padding 带信息）
            with torch.no_grad():
                self.ID_embeddings.weight[self.item_num].zero_()

    # 对 语言嵌入做 SVD、裁剪、标准化
    def semantic_space_decomposion(self, clipped_dim, **key_words):
        language_embs = self.load_language_embeddings(key_words["language_embs_path"], key_words["language_model_type"],
                                                      key_words["language_embs_scale"])
        # default: False -> 不使用流行度
        if not key_words["item_frequency_flag"]:
            # The default item distribution is a uniform distribution.
            self.language_mean = np.mean(language_embs, axis=0)
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


class Bert4Rec_backbone(nn.Module):
    def __init__(self, device, item_num, **key_words):
        super(Bert4Rec_backbone, self).__init__()
        self.seq_len = key_words['max_len']
        # actual item_num, Bert4Rec need 2 special tokens: padding_token and mask_token
        self.item_num = item_num

        self.dropout = key_words["dropout_rate"]
        self.device = device
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        # self.language_dim = self.item_embeddings.language_dim
        self.hidden_dim = key_words["hidden_dim"]

        # TODO: full_seq = seq + target, len(full) = self.seq_len + 1
        self.positional_embeddings = nn.Embedding(
            num_embeddings=self.seq_len + 1,
            embedding_dim=self.hidden_dim
        )
        # emb_dropout is added
        self.emb_dropout = nn.Dropout(self.dropout)
        self.ln_1 = nn.LayerNorm(self.hidden_dim)
        self.ln_2 = nn.LayerNorm(self.hidden_dim)
        self.ln_3 = nn.LayerNorm(self.hidden_dim)
        self.ln_4 = nn.LayerNorm(self.hidden_dim)

        # Bert4Rec Settings
        self.mask_ratio = key_words["mask_ratio"]
        self.mask_token_ = item_num + 1  # 真实item取值[0, item_num-1], padding_idx=item_num, so mask_token=item_num+1

        self.item_encoder = TransformerEncoder(**key_words)

    def get_bi_attention_mask(self, item_seq):
        pad_id = self.item_num
        attention_mask = (item_seq != pad_id).long()  # 非pad才可见
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def add_position_embedding(self, sequence):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.embed_ID(sequence)
        position_embeddings = self.positional_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings

        sequence_emb = self.emb_dropout(sequence_emb)

        return sequence_emb

    # 不同模型实现不同的接口 -> AlphaFuse的实现就是row space + null space
    def embed_ID(self, x):
        # return self.item_embeddings.ID_embeddings(x)
        pass

    def return_item_emb(self, ):
        # return self.item_embeddings.ID_embeddings.weight
        pass

    def forward(self, sequences):
        extended_attention_mask = self.get_bi_attention_mask(sequences)
        sequence_emb = self.add_position_embedding(sequences)
        item_encoded_layers = self.item_encoder(sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True,
                                                )

        sequence_output = item_encoded_layers[-1]

        return sequence_output

    def predict(self, sequences):
        # sequences: [B, max_len]
        B = sequences.size(0)
        mask_col = torch.full((B, 1), self.mask_token_, device=sequences.device, dtype=sequences.dtype)
        full = torch.cat([sequences, mask_col], dim=1)  # [B, max_len+1]

        seq_state_hidden = self.forward(full)  # [B, max_len+1, H]
        state_hidden = seq_state_hidden[:, -1, :]  # MASK position

        item_embs = self.return_item_emb()[:self.item_num]
        return state_hidden @ item_embs.t()


    def calculate_mlm_loss(self, input_ids: torch.Tensor, answers: torch.Tensor) -> torch.Tensor:
        """
        Fix-1: Always mask the appended answer position (prevent target leakage).
        Fix-2: Only sample extra masks from the history part (exclude last pos).
        """
        device = input_ids.device
        B, L0 = input_ids.shape
        pad_id = self.item_num
        mask_id = self.mask_token_
        L = L0 + 1

        # full_seq = [history, answer]
        full_seq = torch.cat([input_ids, answers.view(-1, 1)], dim=1)  # [B, L]

        # ---- (A) mandatory mask: last position ----
        last_pos = torch.full((B, 1), L - 1, device=device, dtype=torch.long)  # [B,1]

        # ---- (B) extra random masks from history only ----
        hist = full_seq[:, :L0]  # [B, L0]
        hist_valid = (hist != pad_id)  # only real tokens in history
        hist_len = hist_valid.sum(dim=1)  # [B]

        extra_num = (hist_len.float() * self.mask_ratio).long()  # [B]
        extra_num = torch.where(hist_len > 0, extra_num, torch.zeros_like(extra_num))
        extra_num = torch.minimum(extra_num, hist_len)  # avoid > hist_len
        max_k = int(extra_num.max().item())

        if max_k > 0:
            scores = torch.rand(B, L0, device=device).masked_fill(~hist_valid, float("-inf"))
            extra_idx = scores.argsort(dim=1, descending=True)[:, :max_k]  # [B,max_k]
            ar = torch.arange(max_k, device=device).unsqueeze(0).expand(B, max_k)
            extra_mask_pos = ar < extra_num.unsqueeze(1)  # [B,max_k]
        else:
            extra_idx = torch.empty(B, 0, device=device, dtype=torch.long)
            extra_mask_pos = torch.empty(B, 0, device=device, dtype=torch.bool)

        # ---- (C) combine indices: [last_pos] + [extra_idx] ----
        masked_index = torch.cat([last_pos, extra_idx], dim=1)  # [B, 1+max_k]
        mask_pos = torch.cat(
            [torch.ones(B, 1, device=device, dtype=torch.bool), extra_mask_pos],
            dim=1
        )  # [B, 1+max_k]

        max_all = masked_index.size(1)
        if max_all == 0:
            return torch.tensor(0.0, device=device)

        # labels
        pos_items = full_seq.gather(dim=1, index=masked_index)  # [B, 1+max_k]
        pos_items = pos_items.masked_fill(~mask_pos, 0)

        # masked input
        masked_full_seq = full_seq.clone()
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, max_all)
        masked_full_seq[batch_idx[mask_pos], masked_index[mask_pos]] = mask_id

        # forward
        seq_output = self.forward(masked_full_seq)  # [B, L, H]
        H = seq_output.size(-1)

        # gather masked outputs
        gather_idx = masked_index.unsqueeze(-1).expand(-1, -1, H)  # [B, 1+max_k, H]
        masked_output = seq_output.gather(dim=1, index=gather_idx)  # [B, 1+max_k, H]

        # logits over real items only
        item_emb = self.return_item_emb()[:self.item_num]  # [V, H]
        logits = masked_output @ item_emb.t()  # [B, 1+max_k, V]

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        per_pos_loss = loss_fct(logits.reshape(-1, self.item_num), pos_items.reshape(-1))
        targets = mask_pos.float().reshape(-1)
        loss = (per_pos_loss * targets).sum() / targets.sum().clamp(min=1.0)
        return loss


### pure ID embeddings
class Bert4Rec(Bert4Rec_backbone):
    def __init__(self, device, item_num, **key_words):
        super().__init__(device, item_num, **key_words)

        self.item_embeddings = Item_Embedding("ID", item_num, **key_words)

    def embed_ID(self, x):
        return self.item_embeddings.ID_embeddings(x)

    def return_item_emb(self, ):
        return self.item_embeddings.ID_embeddings.weight


class MoRec(Bert4Rec_backbone):
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


class WhitenRec(Bert4Rec_backbone):
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


class LLMInit(Bert4Rec_backbone):
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


class RLMRec(Bert4Rec_backbone):
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
            self.return_item_emb()[:self.item_num])  # self.return_item_emb()[-1] is the padding embedding
        language_embs = self.item_embeddings.language_embeddings.weight
        rec_language_embs = F.normalize(rec_language_embs, p=2, dim=-1)
        language_embs = F.normalize(language_embs, p=2, dim=-1)
        return 1 - (rec_language_embs * language_embs).sum() / self.item_num

    def reconstruct_con_loss(self, ):
        language_embs = self.item_embeddings.language_embeddings.weight
        rec_ID_embs = self.reconstructor(language_embs)  # self.return_item_emb()[-1] is the padding embedding
        ID_embs = self.return_item_emb()[:self.item_num]
        rec_ID_embs = F.normalize(rec_ID_embs, p=2, dim=-1)
        ID_embs = F.normalize(ID_embs, p=2, dim=-1)
        return 1 - (rec_ID_embs * ID_embs).sum() / self.item_num


class UniSRec(Bert4Rec_backbone):
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


class LLMESR(Bert4Rec_backbone):
    def __init__(self, device, item_num, **key_words):
        super().__init__(device, item_num, **key_words)

        self.item_embeddings = Item_Embedding("Dual_view", item_num, **key_words)
        # Dual_view 的Init方式是，language_embedding 使用全量的 language_dim
        # ID_embedding 【原论文使用的是PCA】 则使用svd + whiten 得到的 hidden_dim
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
        Bert4Rec: 用 [seq, MASK] 的最后位置表征做 next-item prediction
        sequences: (B, L) 纯历史序列（不含 mask）
        return: (B, 2H)
        """
        device = sequences.device
        B = sequences.size(0)
        mask_col = torch.full((B, 1), self.mask_token_, dtype=torch.long, device=device)
        full = torch.cat([sequences, mask_col], dim=1)  # (B, L+1)
        out = self.forward(full)  # (B, L+1, 2H)  :contentReference[oaicite:6]{index=6}
        return out[:, -1, :]

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
        # shape -> [item_num+2, language_dim]
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
        device = sequences.device
        B, L = sequences.size()

        # 1) dual-view embeddings (both -> hidden_dim)
        inputs_id_emb, inputs_text_emb = self.embed_ID_text(sequences)  # [B,L,H], [B,L,H]

        # 2) position embedding (use actual length L, not self.seq_len+1 hardcode)
        pos_ids = torch.arange(L, dtype=torch.long, device=device).unsqueeze(0).expand(B, L)
        pos_emb = self.positional_embeddings(pos_ids)  # [B,L,H]

        inputs_id_emb = inputs_id_emb + pos_emb
        inputs_text_emb = inputs_text_emb + pos_emb

        id_seq = self.emb_dropout(inputs_id_emb)
        text_seq = self.emb_dropout(inputs_text_emb)

        # 3) cross-attention (your module masks padding inside via log_seqs==pad_id)
        # Multi_CrossAttention: first arg -> Q, second arg -> K,V  :contentReference[oaicite:4]{index=4}
        cross_id_seqs = self.language2ID(text_seq, id_seq, sequences, self.item_num)  # [B,L,H]
        cross_text_seqs = self.ID2language(id_seq, text_seq, sequences, self.item_num)  # [B,L,H]

        cross_id_seqs = cross_id_seqs + 0.0 * id_seq
        cross_text_seqs = cross_text_seqs + 0.0 * text_seq

        # 4) Bert4Rec backbone encoder (bidirectional attention mask)
        extended_attention_mask = self.get_bi_attention_mask(sequences)  # :contentReference[oaicite:5]{index=5}

        id_layers = self.item_encoder(
            cross_id_seqs, extended_attention_mask, output_all_encoded_layers=False
        )
        text_layers = self.item_encoder(
            cross_text_seqs, extended_attention_mask, output_all_encoded_layers=False
        )

        id_out = id_layers[-1]  # [B,L,H]
        text_out = text_layers[-1]  # [B,L,H]

        seq_out = torch.cat([id_out, text_out], dim=-1)  # [B,L,2H]
        return seq_out

    def reg_loss(self, sequences):
        unfold_item_id = torch.masked_select(sequences, sequences != self.item_num)
        language_emb, id_emb = self.embed_ID_text(unfold_item_id)
        reg_loss = self.reg(language_emb, id_emb)
        return reg_loss


class AlphaFuse(Bert4Rec_backbone):
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


class IBaSRec(Bert4Rec_backbone):
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
        self.beh_emb = nn.Embedding(
            num_embeddings=self.item_num + 2,
            embedding_dim=self.hidden_dim,
            padding_idx=self.item_num
        )
        # 初始化id_embedding
        if init_type == "uniform":
            nn.init.uniform_(self.beh_emb.weight, a=0.0, b=1.0)
        elif init_type == "normal":
            nn.init.normal_(self.beh_emb.weight, 0,1)
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
            num_embeddings=self.item_num + 2,
            embedding_dim=self.hidden_dim,
            padding_idx=self.item_num
        )
        nn.init.zeros_(self.sem_delta.weight)  # start from pure lang0

        # ---- item-wise gate: gate(i) in (0,1) ----
        self.gate = nn.Embedding(
            num_embeddings=self.item_num + 2,
            embedding_dim=1,
            padding_idx=self.item_num
        )
        # start with small behaviour contribution: sigmoid(-2) ~ 0.12 -> rely on semantic_embedding at first
        nn.init.constant_(self.gate.weight, -2.0)

        with torch.no_grad():
            # semantic priors for special tokens = 0
            self.lang0.weight[self.item_num].zero_()  # pad
            self.lang0.weight[self.mask_token_].zero_()  # mask

            # pad token should never contribute behaviour
            self.beh_emb.weight[self.item_num].zero_()
            self.gate.weight[self.item_num].fill_(-10.0)  # sigmoid ~ 0

            # mask token needs a strong, learnable embedding for MLM
            self.gate.weight[self.mask_token_].fill_(-20.0)

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
            e_sem, e_beh, g, e_lang0 = self._get_item_components(x)
            e = e_sem + (self.beh_scale * g) * e_beh
            return e

    def return_item_emb(self):
        ids = torch.arange(self.item_num + 1, device=self.device, dtype=torch.long)
        return self.embed_ID(ids)  # (item_num+1, D)

    # ---------- debugging helpers ----------
    @torch.no_grad()
    def predict_sem(self, sequences):
        B = sequences.size(0)

        mask_col = torch.full(
            (B, 1),
            self.mask_token_,
            device=sequences.device,
            dtype=sequences.dtype
        )
        full = torch.cat([sequences, mask_col], dim=1)  # [B, L+1]

        seq_hidden = self.forward(full)  # [B, L+1, H]
        state = seq_hidden[:, -1, :]

        ids = torch.arange(self.item_num, device=self.device)
        e_sem, _, _, _ = self._get_item_components(ids)  # [I, H]

        return state @ e_sem.t()  # [B, I]

    @torch.no_grad()
    def predict_lang(self, sequences):
        B = sequences.size(0)

        mask_col = torch.full(
            (B, 1),
            self.mask_token_,
            device=sequences.device,
            dtype=sequences.dtype
        )
        full = torch.cat([sequences, mask_col], dim=1)  # [B, L+1]

        seq_hidden = self.forward(full)  # [B, L+1, H]
        state = seq_hidden[:, -1, :]  # [B, H]

        ids = torch.arange(self.item_num, device=self.device)
        _, _, _, e_lang0 = self._get_item_components(ids)  # [I, H]

        return state @ e_lang0.t()  # [B, I]

    @torch.no_grad()
    def predict_beh(self, sequences):
        B = sequences.size(0)

        mask_col = torch.full(
            (B, 1),
            self.mask_token_,
            device=sequences.device,
            dtype=sequences.dtype
        )
        full = torch.cat([sequences, mask_col], dim=1)

        seq_hidden = self.forward(full)
        state = seq_hidden[:, -1, :]

        ids = torch.arange(self.item_num, device=self.device)
        _, e_beh, _, _ = self._get_item_components(ids)

        return state @ e_beh.t()

# ============ pre_ex =============

class StaticFuse(Bert4Rec_backbone):
    def __init__(self, device, item_num, static_alpha=0.5, **key_words):
        super().__init__(device, item_num, **key_words)
        self.device = device
        self.item_num = item_num
        self.static_alpha = static_alpha

        # 复用 BF 的 SVD 处理逻辑
        tmp_emb = Item_Embedding("BF", item_num, **key_words)
        lang_weight = tmp_emb.language_embeddings.weight.detach().clone()
        self.register_buffer("lang0", lang_weight)

        # 行为 ID (Bert4Rec 需要 +2: padding 和 mask)
        self.beh_emb = nn.Embedding(
            num_embeddings=self.item_num + 2,
            embedding_dim=key_words["hidden_dim"],
            padding_idx=self.item_num
        )
        nn.init.normal_(self.beh_emb.weight, 0, 1)
        with torch.no_grad():
            self.beh_emb.weight[self.item_num].zero_()

    def embed_ID(self, x):
        e_lang = self.lang0[x]
        e_beh = self.beh_emb(x)
        return e_lang + self.static_alpha * e_beh

    def return_item_emb(self):
        ids = torch.arange(self.item_num + 1, device=self.device, dtype=torch.long)
        return self.embed_ID(ids)
