"""network.py — 多模态离散时间生存模型

模型架构：
  - StaticEncoder: 2 层前馈网络，将 107 维静态特征映射到 128 维嵌入
  - TimeSeriesEncoder: 双向 GRU，编码 22 通道 × 72 时步的 ICU 时序数据
  - ClinicalBERTEncoder: 冻结的 Bio_ClinicalBERT + 线性投影层，编码放射科报告
  - CrossAttention: 成对跨模态注意力（Static↔TS, Static↔Text, TS↔Text）
  - MultiModalSurvival: 融合三模态 + landmark 时间嵌入，输出 K 个区间的风险概率
"""
from __future__ import annotations

import torch
import torch.nn as nn

from utils.options import DEFAULT_ATTENTION_HEADS, DEFAULT_BERT_MODEL


class StaticEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TimeSeriesEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.seq_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.summary_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    @staticmethod
    def _length_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        steps = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        return steps < lengths.unsqueeze(1)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, max_len, _ = x.shape
        seq_output = x.new_zeros((batch_size, max_len, self.hidden_dim))
        summary = x.new_zeros((batch_size, self.hidden_dim))

        if lengths is not None:
            lengths = lengths.to(x.device)
            valid_mask = lengths > 0
            token_mask = self._length_mask(lengths.clamp(min=0), max_len)
            if valid_mask.any():
                packed = nn.utils.rnn.pack_padded_sequence(
                    x[valid_mask],
                    lengths[valid_mask].cpu(),
                    batch_first=True,
                    enforce_sorted=False,
                )
                packed_output, hidden = self.gru(packed)
                unpacked, _ = nn.utils.rnn.pad_packed_sequence(
                    packed_output,
                    batch_first=True,
                    total_length=max_len,
                )
                seq_output[valid_mask] = self.seq_proj(unpacked)
                last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
                summary[valid_mask] = self.summary_proj(last_hidden)
            return seq_output, summary, token_mask

        seq_output, hidden = self.gru(x)
        token_mask = torch.ones((batch_size, max_len), dtype=torch.bool, device=x.device)
        seq_output = self.seq_proj(seq_output)
        summary = self.summary_proj(torch.cat([hidden[-2], hidden[-1]], dim=1))
        return seq_output, summary, token_mask


class ClinicalBERTEncoder(nn.Module):
    def __init__(
        self,
        output_dim: int = 128,
        dropout: float = 0.3,
        freeze_bert: bool = True,
        model_name: str = DEFAULT_BERT_MODEL,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError("transformers is required for ClinicalBERT text encoding.") from exc

        self.bert = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        bert_dim = int(self.bert.config.hidden_size)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.proj = nn.Sequential(
            nn.Linear(bert_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence = self.proj(outputs.last_hidden_state)
        return sequence, attention_mask.bool()


class CrossAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = DEFAULT_ATTENTION_HEADS, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
        kv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = query.unsqueeze(1)
        if kv_mask is not None:
            kv_mask = kv_mask.bool()
            empty_rows = ~kv_mask.any(dim=1)
            if empty_rows.any():
                kv_mask = kv_mask.clone()
                kv_mask[empty_rows, 0] = True
                kv = kv.clone()
                kv[empty_rows, 0] = 0.0
            key_padding_mask = ~kv_mask
        else:
            key_padding_mask = None
        attended, _ = self.attn(query, kv, kv, key_padding_mask=key_padding_mask)
        return self.norm(query + self.dropout(attended)).squeeze(1)


class MultiModalSurvival(nn.Module):
    def __init__(
        self,
        static_dim: int,
        ts_input_dim: int,
        hidden_dim: int = 128,
        num_time_bins: int = 16,
        use_static: bool = True,
        use_ts: bool = True,
        use_text: bool = True,
        bert_model_name: str = DEFAULT_BERT_MODEL,
        local_files_only: bool = False,
        freeze_bert: bool = True,
        attention_heads: int = DEFAULT_ATTENTION_HEADS,
        use_cross_attention: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_static = use_static
        self.use_ts = use_ts
        self.use_text = use_text
        self.use_cross_attention = use_cross_attention
        self.text_mode = "content"

        self.static_enc = StaticEncoder(static_dim, hidden_dim=hidden_dim * 2, output_dim=hidden_dim) if use_static else None
        self.ts_enc = TimeSeriesEncoder(ts_input_dim, hidden_dim=hidden_dim) if use_ts else None
        self.text_enc = (
            ClinicalBERTEncoder(
                output_dim=hidden_dim,
                dropout=0.3,
                freeze_bert=freeze_bert,
                model_name=bert_model_name,
                local_files_only=local_files_only,
            )
            if use_text
            else None
        )

        self.landmark_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
        )

        if use_cross_attention:
            self.ca_static_ts = CrossAttention(hidden_dim, attention_heads, dropout=0.1) if use_static and use_ts else None
            self.ca_static_text = CrossAttention(hidden_dim, attention_heads, dropout=0.1) if use_static and use_text else None
            self.ca_ts_text = CrossAttention(hidden_dim, attention_heads, dropout=0.1) if use_ts and use_text else None
        else:
            # Simple concatenation fusion — no cross-attention modules
            self.ca_static_ts = None
            self.ca_static_text = None
            self.ca_ts_text = None

        num_vectors = 1  # landmark always present
        if use_static:
            num_vectors += 1
        if use_ts:
            num_vectors += 1
        if use_text:
            num_vectors += 1
        if use_cross_attention:
            if use_static and use_ts:
                num_vectors += 1
            if use_static and use_text:
                num_vectors += 1
            if use_ts and use_text:
                num_vectors += 1
        fusion_in = num_vectors * hidden_dim + (1 if use_text else 0)

        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim * 4),
            nn.LayerNorm(hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        self.hazard_head = nn.Linear(hidden_dim, num_time_bins)

    @staticmethod
    def _masked_mean(seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.float().unsqueeze(-1)
        denom = weights.sum(dim=1).clamp(min=1.0)
        return (seq * weights).sum(dim=1) / denom

    def forward(
        self,
        x_static: torch.Tensor,
        x_ts: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
        landmark_h_norm: torch.Tensor | None = None,
        ts_lengths: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fused_parts = []
        static_summary = None
        if self.use_static:
            static_summary = self.static_enc(x_static)
            fused_parts.append(static_summary)

        ts_summary = None
        if self.use_ts:
            if x_ts is None:
                raise ValueError("x_ts is required when use_ts=True.")
            ts_seq, ts_summary, ts_token_mask = self.ts_enc(x_ts, ts_lengths)
            fused_parts.append(ts_summary)
            if self.use_cross_attention and self.use_static and self.ca_static_ts is not None:
                fused_parts.append(self.ca_static_ts(static_summary, ts_seq, ts_token_mask))

        if self.use_text:
            if text_mask is None or input_ids is None or attention_mask is None:
                raise ValueError("text_mask, input_ids, and attention_mask are required when use_text=True.")
            sample_text_mask = text_mask.float().view(-1)
            batch_size, seq_len = attention_mask.shape
            text_seq = x_static.new_zeros((batch_size, seq_len, self.hidden_dim))
            text_token_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=attention_mask.device)

            if self.text_mode != "mask_only":
                active_rows = sample_text_mask > 0
                if active_rows.any():
                    active_text_seq, active_text_token_mask = self.text_enc(
                        input_ids[active_rows],
                        attention_mask[active_rows],
                    )
                    text_seq[active_rows] = active_text_seq
                    text_token_mask[active_rows] = active_text_token_mask
            text_summary = self._masked_mean(text_seq, text_token_mask)
            fused_parts.append(text_summary)
            if self.use_cross_attention and self.use_static and self.ca_static_text is not None:
                fused_parts.append(self.ca_static_text(static_summary, text_seq, text_token_mask))
            if self.use_cross_attention and self.use_ts and ts_summary is not None and self.ca_ts_text is not None:
                fused_parts.append(self.ca_ts_text(ts_summary, text_seq, text_token_mask))

        if landmark_h_norm is None:
            raise ValueError("landmark_h_norm is required.")
        fused_parts.append(self.landmark_proj(landmark_h_norm))

        fusion_input = torch.cat(fused_parts, dim=1)
        if self.use_text and text_mask is not None:
            fusion_input = torch.cat([fusion_input, text_mask.float()], dim=1)

        fused = self.fusion(fusion_input)
        hazards = torch.sigmoid(self.hazard_head(fused))
        survs = torch.cumprod(1.0 - hazards, dim=1)
        return hazards, survs
