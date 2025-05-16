import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import List, Optional
import numpy as np
import xgboost as xgb

class ProbRegEnsemble:
    """
    XGBRegressor C개를 묶어 XGBClassifier와 동일한 predict_proba() 인터페이스 제공
    """
    def __init__(self, regs: List[xgb.XGBRegressor]):
        self.regs = regs                    # 각 클래스별 회귀기

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        preds = np.column_stack([r.predict(X) for r in self.regs])   # (N, C)
        preds = np.clip(preds, 1e-12, None)                          # 음수 방지
        preds /= preds.sum(axis=1, keepdims=True)                    # 확률화
        return preds


class ResidualBlock(nn.Module):
    """
    두 개의 Linear → SiLU → (선택적) Dropout → Skip Connection
    입력·출력 차원 동일 (hidden_dim).
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.0, use_batchnorm: bool = True):
        super().__init__()

        layers: list[nn.Module] = [
            nn.Linear(hidden_dim, hidden_dim)
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.SiLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))

        self.block = nn.Sequential(*layers)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))  # Skip connection



class ResNetMLP(nn.Module):
    """
    ResNet‑style MLP.

    Args:
        input_dim     : 입력 차원
        hidden_dim    : Residual block 내부 차원 (고정)
        num_blocks    : ResidualBlock 개수
        output_dim    : 출력 차원
        dropout       : 드롭아웃 확률
        use_batchnorm : BatchNorm 사용 여부
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        output_dim: int = 4,
        dropout: float = 0.0,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout, use_batchnorm) for _ in range(num_blocks)]
        )

        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.res_blocks(x)
        return self.head(x)
    

# -------------------------------------------------
# 1. CNN + LSTM
# -------------------------------------------------
class CNNLSTM(nn.Module):
    """
    Conv1d → SiLU → LSTM → FC
      * same_padding=True  : 커널이 짝수여도 출력 길이를 입력과 동일하게 유지
      * lengths(Optional): 실제 시퀀스 길이를 전달하면 PackedSequence 처리
    """
    def __init__(
        self,
        feature_dim: int,
        num_filters: int,
        kernel_size: int,
        lstm_hidden: int,
        output_dim: int,
        lstm_layers: int = 4,
        bidirectional: bool = False,
        dropout: float = 0.0,
        same_padding: bool = True,          # ← NEW
    ) -> None:
        super().__init__()

        pad = (kernel_size - 1) // 2 if same_padding else kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels=feature_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=pad,
        )
        self.SiLU = nn.SiLU()

        self.lstm = nn.LSTM(
            input_size=num_filters,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.fc = nn.Linear(lstm_out_dim, output_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None):
        # x: (B, seq_len, feat)  →  (B, feat, seq_len)
        x = self.SiLU(self.conv(x.transpose(1, 2)))      # (B, num_filters, L)
        x = x.transpose(1, 2)                            # (B, L, num_filters)

        if lengths is not None:
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            output, (h_n, _) = self.lstm(packed)
            last_hidden = h_n[-1]                        # (B, lstm_hidden*)
        else:
            output, _ = self.lstm(x)
            last_hidden = output[:, -1, :]               # 마지막 시점

        return self.fc(last_hidden)


# -------------------------------------------------
# 2. LSTM (순수)
# -------------------------------------------------
class LSTMClassifier(nn.Module):
    """
    순수 LSTM 기반 분류기 (PackedSequence 지원).
    """
    def __init__(
        self,
        feature_dim: int,
        lstm_hidden: int,
        output_dim: int,
        lstm_layers: int = 4,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.fc = nn.Linear(lstm_out_dim, output_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None):
        # x: (B, seq_len, feature_dim)
        if lengths is not None:
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, (h_n, _) = self.lstm(packed)
            last_hidden = h_n[-1]                        # (B, lstm_out_dim)
        else:
            output, _ = self.lstm(x)
            last_hidden = output[:, -1, :]               # (B, lstm_out_dim)

        return self.fc(last_hidden)


class CNNMLP(nn.Module):
    """
    Conv1d → SiLU → Global‑MaxPool → MLP
    입력: (batch, seq_len, feat_dim)
    """
    def __init__(self, feat_dim, num_filters=128,
                 kernel_size=3, hidden_dim=256,
                 output_dim=4, dropout=0.2):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(feat_dim, num_filters,
                              kernel_size, padding=pad)
        self.SiLU = nn.SiLU()
        self.pool = nn.AdaptiveMaxPool1d(1)      # (B, C, 1)
        self.head = nn.Sequential(
            nn.Flatten(),                        # (B, C)
            nn.Linear(num_filters, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):         # x: (B, L, F)
        x = self.conv(x.transpose(1, 2))         # (B, C, L)
        x = self.SiLU(x)
        x = self.pool(x)                         # (B, C, 1)
        return self.head(x)
    

class AEClassifier(nn.Module):
    """AutoEncoder + 4‑class classifier"""
    def __init__(self, input_dim, bottleneck_dim=128, hidden_dim=512,
                 dropout=0.1, num_classes=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        logits = self.cls_head(z)
        return recon, logits


class VAEClassifier(nn.Module):
    """Variational AE + 4‑class classifier"""
    def __init__(self, input_dim, bottleneck_dim=128, hidden_dim=512,
                 dropout=0.1, num_classes=4):
        super().__init__()
        # encoder → mu, logvar
        self.enc_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.mu = nn.Linear(hidden_dim, bottleneck_dim)
        self.logvar = nn.Linear(hidden_dim, bottleneck_dim)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )
        # classifier
        self.cls_head = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.enc_fc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        logits = self.cls_head(z)
        return recon, logits, mu, logvar

# -------------------------------------------------
# 5. MLP (개선형)
# -------------------------------------------------
class MLP(nn.Module):
    """
    유연한 깊이의 완전연결 네트워크.

    Args:
        input_dim: 입력 벡터 차원.
        hidden_dims: 예) [256, 128] 형태의 리스트.
        output_dim: 출력 차원 (클래스 수 등).
        dropout: 층 사이 드롭아웃 확률.
        use_batchnorm: BatchNorm1d 사용 여부.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | tuple[int, ...],
        output_dim: int,
        dropout: float = 0.0,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerMLP(nn.Module):
    """
    Transformer Encoder → (CLS/mean) Pooling → MLP 분류기

    입력 형상
        x: (batch, seq_len, feature_dim)

    Args
        feature_dim   : Transformer 입력 차원 (= model_dim)
        n_heads       : Multi‑Head Attention 머리 수
        n_layers      : Encoder layer 수
        ff_dim        : Transformer 내부 FFN hidden 차원
        dropout       : 드롭아웃 확률
        use_cls_token : True 면 CLS 토큰, False 면 mean‑pool
        mlp_hidden    : 분류기 MLP hidden 차원
        num_classes   : 출력 클래스 수
    """
    def __init__(
        self,
        feature_dim: int,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.1,
        use_cls_token: bool = True,
        mlp_hidden: int = 256,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.use_cls_token = use_cls_token

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, feature_dim))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,          # (B, L, D) 형식 사용
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.mlp_head = nn.Sequential(
            nn.Linear(feature_dim, mlp_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_classes),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        if self.use_cls_token:
            nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        if self.use_cls_token:
            cls_tok = self.cls_token.expand(x.size(0), -1, -1)  # (B, 1, D)
            x = torch.cat([cls_tok, x], dim=1)                  # (B, L+1, D)

        h = self.encoder(x)                                     # (B, L(+1), D)

        if self.use_cls_token:
            pooled = h[:, 0]                                    # CLS 토큰
        else:
            pooled = h.mean(dim=1)                              # Mean‑pool

        return self.mlp_head(pooled)