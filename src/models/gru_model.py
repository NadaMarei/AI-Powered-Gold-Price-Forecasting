"""
Novel GRU-based Architecture for Gold Price Forecasting
========================================================

This module implements a publication-grade GRU architecture with:
1. Volatility-adaptive gating mechanism
2. Temporal attention layer
3. Skip connections between layers
4. Feature-wise transformation layers

Mathematical Formulation
------------------------

Standard GRU equations:
    z_t = σ(W_z · [h_{t-1}, x_t])           # Update gate
    r_t = σ(W_r · [h_{t-1}, x_t])           # Reset gate
    h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t])   # Candidate hidden state
    h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t  # Final hidden state

Our Volatility-Adaptive Modification:
    v_t = volatility indicator at time t
    α_t = sigmoid(W_v · v_t + b_v)          # Volatility adaptation factor
    
    Modified gates with volatility conditioning:
    z_t = σ(W_z · [h_{t-1}, x_t] + α_t · W_zv · v_t)
    r_t = σ(W_r · [h_{t-1}, x_t] + α_t · W_rv · v_t)
    
    This allows the model to adapt its memory retention and reset behavior
    based on current market volatility regime.

Skip Connections:
    h_l^{out} = h_l + W_{skip} · h_{l-2}   (for l >= 2)
    
    Enables gradient flow and combines information across layers.

Temporal Attention:
    e_t = v^T · tanh(W_h · h_t + b)        # Attention scores
    α_t = softmax(e_t)                      # Attention weights
    c = Σ α_t · h_t                         # Context vector
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Dict, List
import math


class FeatureTransformLayer(nn.Module):
    """
    Feature-wise transformation layer for heterogeneous input scales.
    
    Applies learnable per-feature scaling and shifting, followed by
    group normalization to handle different feature types (prices, 
    volumes, percentages, etc.).
    """
    
    def __init__(self, num_features: int, num_groups: int = 8):
        """
        Initialize feature transform layer.
        
        Args:
            num_features: Number of input features
            num_groups: Number of groups for group normalization
        """
        super().__init__()
        
        self.num_features = num_features
        
        # Learnable per-feature scaling
        self.feature_scale = nn.Parameter(torch.ones(1, 1, num_features))
        self.feature_shift = nn.Parameter(torch.zeros(1, 1, num_features))
        
        # Group normalization (works on feature dimension)
        # Adjust num_groups if num_features is smaller
        actual_groups = min(num_groups, num_features)
        # Ensure num_features is divisible by num_groups
        while num_features % actual_groups != 0:
            actual_groups -= 1
        self.group_norm = nn.GroupNorm(actual_groups, num_features)
        
        # Non-linear transformation
        self.activation = nn.GELU()
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Transform features.
        
        Args:
            x: Input tensor [batch, seq_len, features]
            
        Returns:
            Transformed tensor [batch, seq_len, features]
        """
        # Apply learnable scaling and shifting
        x = x * self.feature_scale + self.feature_shift
        
        # Reshape for group norm: [batch * seq_len, features] -> [batch * seq_len, features, 1]
        batch_size, seq_len, num_features = x.shape
        x = x.view(batch_size * seq_len, num_features)
        x = self.group_norm(x)
        x = x.view(batch_size, seq_len, num_features)
        
        # Non-linear activation
        x = self.activation(x)
        
        return x


class VolatilityAdaptiveGRUCell(nn.Module):
    """
    Custom GRU cell with volatility-adaptive gating mechanism.
    
    Modifies the standard GRU gates to incorporate market volatility
    information, allowing the model to adapt its behavior in different
    market regimes.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        volatility_size: int = 1,
        dropout: float = 0.0
    ):
        """
        Initialize volatility-adaptive GRU cell.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            volatility_size: Size of volatility indicator
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.volatility_size = volatility_size
        
        # Standard GRU weights
        self.weight_ih = nn.Linear(input_size, 3 * hidden_size, bias=True)
        self.weight_hh = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        
        # Volatility conditioning weights
        self.weight_vol = nn.Linear(volatility_size, 2 * hidden_size, bias=True)
        self.volatility_gate = nn.Linear(volatility_size, 1, bias=True)
        
        # Variational dropout
        self.dropout = dropout
        self.input_dropout = nn.Dropout(dropout)
        self.hidden_dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name or param.dim() < 2:
                nn.init.zeros_(param)
    
    def forward(
        self,
        input: Tensor,
        hidden: Tensor,
        volatility: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass of volatility-adaptive GRU cell.
        
        Args:
            input: Input tensor [batch, input_size]
            hidden: Hidden state [batch, hidden_size]
            volatility: Volatility indicator [batch, volatility_size]
            
        Returns:
            New hidden state [batch, hidden_size]
        """
        # Apply dropout
        if self.training and self.dropout > 0:
            input = self.input_dropout(input)
            hidden = self.hidden_dropout(hidden)
        
        # Compute standard GRU gates
        gi = self.weight_ih(input)  # [batch, 3 * hidden]
        gh = self.weight_hh(hidden)  # [batch, 3 * hidden]
        
        i_r, i_z, i_n = gi.chunk(3, dim=1)
        h_r, h_z, h_n = gh.chunk(3, dim=1)
        
        # Compute volatility adaptation
        if volatility is not None:
            vol_adaptation = torch.sigmoid(self.volatility_gate(volatility))  # [batch, 1]
            vol_gates = self.weight_vol(volatility)  # [batch, 2 * hidden]
            v_r, v_z = vol_gates.chunk(2, dim=1)
            
            # Modified gates with volatility conditioning
            reset_gate = torch.sigmoid(i_r + h_r + vol_adaptation * v_r)
            update_gate = torch.sigmoid(i_z + h_z + vol_adaptation * v_z)
        else:
            reset_gate = torch.sigmoid(i_r + h_r)
            update_gate = torch.sigmoid(i_z + h_z)
        
        # Candidate hidden state
        new_gate = torch.tanh(i_n + reset_gate * h_n)
        
        # Final hidden state
        new_hidden = (1 - update_gate) * hidden + update_gate * new_gate
        
        return new_hidden


class VolatilityAdaptiveGRU(nn.Module):
    """
    Multi-layer bidirectional GRU with volatility-adaptive gating.
    
    Implements skip connections between layers for improved gradient flow
    and information propagation across different abstraction levels.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 3,
        bidirectional: bool = True,
        dropout: float = 0.2,
        use_skip_connections: bool = True,
        volatility_size: int = 1
    ):
        """
        Initialize multi-layer volatility-adaptive GRU.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state per direction
            num_layers: Number of GRU layers
            bidirectional: Whether to use bidirectional GRU
            dropout: Dropout probability
            use_skip_connections: Whether to use skip connections
            volatility_size: Size of volatility indicator
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.use_skip_connections = use_skip_connections
        
        # Forward GRU cells
        self.forward_cells = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size * self.num_directions
            cell = VolatilityAdaptiveGRUCell(
                layer_input_size, hidden_size, volatility_size, dropout
            )
            self.forward_cells.append(cell)
        
        # Backward GRU cells (for bidirectional)
        if bidirectional:
            self.backward_cells = nn.ModuleList()
            for i in range(num_layers):
                layer_input_size = input_size if i == 0 else hidden_size * self.num_directions
                cell = VolatilityAdaptiveGRUCell(
                    layer_input_size, hidden_size, volatility_size, dropout
                )
                self.backward_cells.append(cell)
        
        # Skip connection projections
        if use_skip_connections and num_layers > 2:
            self.skip_projections = nn.ModuleList()
            for i in range(2, num_layers):
                # Project from layer i-2 to layer i
                proj = nn.Linear(hidden_size * self.num_directions, hidden_size * self.num_directions)
                self.skip_projections.append(proj)
        
        # Layer normalization for each layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size * self.num_directions)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        input: Tensor,
        volatility: Optional[Tensor] = None,
        hidden: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through all GRU layers.
        
        Args:
            input: Input tensor [batch, seq_len, input_size]
            volatility: Volatility indicator [batch, volatility_size]
            hidden: Initial hidden state [num_layers * num_directions, batch, hidden_size]
            
        Returns:
            Tuple of:
                - Output tensor [batch, seq_len, hidden_size * num_directions]
                - Final hidden state [num_layers * num_directions, batch, hidden_size]
        """
        batch_size, seq_len, _ = input.shape
        
        # Initialize hidden states
        if hidden is None:
            hidden = self._init_hidden(batch_size, input.device)
        
        # Store layer outputs for skip connections
        layer_outputs = []
        
        current_input = input
        final_hidden = []
        
        for layer_idx in range(self.num_layers):
            # Forward direction
            forward_output = []
            h_forward = hidden[layer_idx * self.num_directions]
            
            for t in range(seq_len):
                h_forward = self.forward_cells[layer_idx](
                    current_input[:, t, :], h_forward, volatility
                )
                forward_output.append(h_forward)
            
            forward_output = torch.stack(forward_output, dim=1)  # [batch, seq_len, hidden]
            final_hidden.append(h_forward)
            
            if self.bidirectional:
                # Backward direction
                backward_output = []
                h_backward = hidden[layer_idx * self.num_directions + 1]
                
                for t in reversed(range(seq_len)):
                    h_backward = self.backward_cells[layer_idx](
                        current_input[:, t, :], h_backward, volatility
                    )
                    backward_output.insert(0, h_backward)
                
                backward_output = torch.stack(backward_output, dim=1)  # [batch, seq_len, hidden]
                final_hidden.append(h_backward)
                
                # Concatenate forward and backward
                layer_output = torch.cat([forward_output, backward_output], dim=2)
            else:
                layer_output = forward_output
            
            # Apply skip connection
            if self.use_skip_connections and layer_idx >= 2:
                skip_idx = layer_idx - 2
                skip_connection = self.skip_projections[skip_idx](layer_outputs[layer_idx - 2])
                layer_output = layer_output + skip_connection
            
            # Apply layer normalization and dropout
            layer_output = self.layer_norms[layer_idx](layer_output)
            if layer_idx < self.num_layers - 1:
                layer_output = self.dropout(layer_output)
            
            layer_outputs.append(layer_output)
            current_input = layer_output
        
        # Stack final hidden states
        final_hidden = torch.stack(final_hidden, dim=0)
        
        return current_input, final_hidden
    
    def _init_hidden(self, batch_size: int, device: torch.device) -> Tensor:
        """Initialize hidden states to zeros."""
        return torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size,
            device=device
        )


class TemporalAttention(nn.Module):
    """
    Multi-head temporal attention mechanism for sequence weighting.
    
    Learns to focus on important time steps in the sequence for
    making predictions, with multiple attention heads for capturing
    different temporal patterns.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize temporal attention.
        
        Args:
            hidden_size: Size of hidden representations
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        # Multi-head projections
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        
        # Attention weights for interpretability
        self.attention_weights: Optional[Tensor] = None
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, hidden_states: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Apply multi-head temporal attention.
        
        Args:
            hidden_states: Hidden states [batch, seq_len, hidden_size]
            mask: Optional attention mask [batch, seq_len]
            
        Returns:
            Tuple of:
                - Context vector [batch, hidden_size]
                - Attention weights [batch, num_heads, seq_len]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Use last hidden state as query
        query = self.query(hidden_states[:, -1:, :])  # [batch, 1, hidden]
        key = self.key(hidden_states)  # [batch, seq_len, hidden]
        value = self.value(hidden_states)  # [batch, seq_len, hidden]
        
        # Reshape for multi-head attention
        query = query.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / scale  # [batch, heads, 1, seq_len]
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Store for interpretability
        self.attention_weights = attention_probs.squeeze(2)  # [batch, heads, seq_len]
        
        # Apply attention to values
        context = torch.matmul(attention_probs, value)  # [batch, heads, 1, head_dim]
        context = context.transpose(1, 2).contiguous().view(batch_size, 1, self.hidden_size)
        context = context.squeeze(1)  # [batch, hidden_size]
        
        # Output projection with residual
        output = self.output_proj(context)
        output = self.layer_norm(output + hidden_states[:, -1, :])
        
        return output, self.attention_weights
    
    def get_attention_weights(self) -> Optional[Tensor]:
        """Get stored attention weights for interpretability."""
        return self.attention_weights


class GoldPriceForecastingModel(nn.Module):
    """
    Complete gold price forecasting model with all innovations.
    
    Architecture:
    1. Feature Transform Layer - Handle heterogeneous input scales
    2. Volatility-Adaptive Bidirectional GRU - 3 layers with skip connections
    3. Temporal Attention - Weight important time steps
    4. Output Network - Dense layers for final prediction
    """
    
    def __init__(
        self,
        num_features: int,
        hidden_size: int = 128,
        num_layers: int = 3,
        num_attention_heads: int = 4,
        dropout: float = 0.2,
        use_attention: bool = True,
        use_skip_connections: bool = True,
        use_volatility_gating: bool = True,
        use_feature_transform: bool = True,
        volatility_idx: Optional[int] = None
    ):
        """
        Initialize the complete forecasting model.
        
        Args:
            num_features: Number of input features
            hidden_size: GRU hidden size per direction
            num_layers: Number of GRU layers
            num_attention_heads: Number of attention heads
            dropout: Dropout probability
            use_attention: Whether to use temporal attention
            use_skip_connections: Whether to use skip connections in GRU
            use_volatility_gating: Whether to use volatility-adaptive gating
            use_feature_transform: Whether to use feature transformation layer
            volatility_idx: Index of volatility feature for regime detection
        """
        super().__init__()
        
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.use_volatility_gating = use_volatility_gating
        self.volatility_idx = volatility_idx
        
        # Feature transform layer
        self.use_feature_transform = use_feature_transform
        if use_feature_transform:
            self.feature_transform = FeatureTransformLayer(num_features)
        
        # GRU encoder
        gru_input_size = num_features
        self.gru = VolatilityAdaptiveGRU(
            input_size=gru_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout,
            use_skip_connections=use_skip_connections,
            volatility_size=1 if use_volatility_gating else 0
        )
        
        # Temporal attention
        gru_output_size = hidden_size * 2  # Bidirectional
        if use_attention:
            self.attention = TemporalAttention(
                hidden_size=gru_output_size,
                num_heads=num_attention_heads,
                dropout=dropout
            )
        
        # Output network
        self.output_network = nn.Sequential(
            nn.Linear(gru_output_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Initialize output layer
        self._init_output_layer()
        
    def _init_output_layer(self):
        """Initialize output layer with small weights for stable training."""
        for module in self.output_network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: Tensor,
        volatility: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input sequences [batch, seq_len, num_features]
            volatility: Optional volatility indicator [batch, 1]
            
        Returns:
            Dictionary with:
                - predictions: Model predictions [batch, 1]
                - attention_weights: Attention weights if using attention
                - hidden_states: Final hidden states
        """
        batch_size, seq_len, _ = x.shape
        
        # Extract volatility from input if not provided
        if volatility is None and self.use_volatility_gating and self.volatility_idx is not None:
            volatility = x[:, -1, self.volatility_idx:self.volatility_idx + 1]
        
        # Feature transformation
        if self.use_feature_transform:
            x = self.feature_transform(x)
        
        # GRU encoding
        gru_output, hidden_states = self.gru(
            x,
            volatility=volatility if self.use_volatility_gating else None
        )
        
        # Apply attention or use last hidden state
        if self.use_attention:
            context, attention_weights = self.attention(gru_output)
        else:
            context = gru_output[:, -1, :]
            attention_weights = None
        
        # Generate prediction
        predictions = self.output_network(context)
        
        return {
            'predictions': predictions,
            'attention_weights': attention_weights,
            'hidden_states': hidden_states,
            'gru_output': gru_output
        }
    
    def predict(self, x: Tensor, volatility: Optional[Tensor] = None) -> Tensor:
        """
        Generate predictions (convenience method).
        
        Args:
            x: Input sequences
            volatility: Optional volatility indicator
            
        Returns:
            Predictions tensor
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x, volatility)
        return output['predictions']
    
    def get_attention_weights(self) -> Optional[Tensor]:
        """Get attention weights from last forward pass."""
        if self.use_attention:
            return self.attention.get_attention_weights()
        return None
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_summary(self) -> Dict:
        """Get model architecture summary."""
        return {
            'num_features': self.num_features,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'use_attention': self.use_attention,
            'use_volatility_gating': self.use_volatility_gating,
            'total_parameters': self.count_parameters(),
            'architecture': str(self)
        }
