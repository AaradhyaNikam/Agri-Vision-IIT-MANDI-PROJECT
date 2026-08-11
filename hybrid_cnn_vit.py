import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights

class PositionalEncoding(nn.Module):
    """
    Injects information about the relative or absolute position of the tokens 
    in the sequence into the input representations. Since we lose spatial layout 
    when flattening CNN feature maps into a sequence, this is crucial for the 
    Transformer to understand "where" each feature patch came from.
    """
    def __init__(self, embed_dim, max_len=1000):
        super(PositionalEncoding, self).__init__()
        # Create a learnable positional embedding. 
        # (1, max_len, embed_dim) will allow broadcasting over the batch dimension.
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, embed_dim))

    def forward(self, x):
        # x shape: (Batch_size, Sequence_length, Embedding_dimension)
        seq_len = x.size(1)
        # Add the positional encoding to the input sequence up to the current sequence length
        return x + self.pos_embedding[:, :seq_len, :]

class HybridCNNViT(nn.Module):
    """
    Hybrid Convolutional Neural Network - Vision Transformer (CNN-ViT) model.
    
    This architecture combines the local feature extraction capabilities of a 
    lightweight CNN (EfficientNet-B0) with the global context understanding of a 
    Vision Transformer. Ideal for fine-grained image classification like crop diseases.
    """
    def __init__(self, num_classes=38, embed_dim=256, num_heads=8, num_layers=4, hidden_dim=512, dropout=0.1):
        super(HybridCNNViT, self).__init__()
        
        # ==========================================
        # 1. Local Feature Extraction (CNN Backbone)
        # ==========================================
        # Using a pre-trained EfficientNet-B0 as the backbone for lightweight feature extraction.
        # Loaded with the default pre-trained weights on ImageNet.
        efficientnet = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        
        # We strip the final classifier of EfficientNet to get the raw convolutional feature maps.
        # The output of the `features` part has 1280 channels by default for EfficientNet-B0.
        self.backbone = efficientnet.features
        
        # We need a 1x1 convolution to project the CNN channels (1280) down to the 
        # Transformer's embedding dimension (e.g., 256) to save compute.
        cnn_out_channels = 1280 
        self.conv_proj = nn.Conv2d(cnn_out_channels, embed_dim, kernel_size=1)
        
        # ==========================================
        # 2. Global Context (Vision Transformer)
        # ==========================================
        # Assuming an input image of 224x224, EfficientNet-B0 outputs a 7x7 spatial feature map.
        # Sequence length will be 7 * 7 = 49 patches. We use max_len=100 to accommodate slightly larger inputs.
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=100)
        
        # A classification token (CLS) that will interact with all other patch tokens via self-attention 
        # and aggregate information from the entire image for the final prediction.
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Define a single Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True # Expects input shape as (batch, seq, feature) rather than (seq, batch, feature)
        )
        
        # Stack multiple Transformer Encoder Layers to form the full Transformer block
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # ==========================================
        # 3. Classification Head
        # ==========================================
        # Final dense layer that takes the CLS token and outputs predictions for the 
        # 38 classes of the PlantVillage dataset.
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        """
        Forward pass defining the flow of tensors from the CNN into the Transformer.
        """
        batch_size = x.size(0)
        
        # Step 1: Pass image through CNN backbone to extract local spatial features.
        # Input shape: (B, 3, H, W) -> Output shape: (B, 1280, H', W')
        features = self.backbone(x)
        
        # Step 2: Project channel dimension to match Transformer embedding dimension.
        # Shape: (B, 1280, H', W') -> (B, embed_dim, H', W')
        projected_features = self.conv_proj(features)
        
        # Step 3: Flatten the spatial dimensions to create a sequence of "tokens/patches".
        # Spatial dimensions H' and W' are collapsed into a single sequence length dimension.
        # Shape: (B, embed_dim, H', W') -> (B, embed_dim, H'*W')
        seq_features = projected_features.flatten(2)
        
        # Step 4: Transpose to match Transformer input requirements (batch_size, sequence_length, embed_dim).
        # Shape: (B, embed_dim, H'*W') -> (B, H'*W', embed_dim)
        seq_features = seq_features.transpose(1, 2)
        
        # Step 5: Add Positional Encoding so the Transformer knows the spatial layout of the patches.
        seq_features = self.pos_encoder(seq_features)
        
        # Step 6: Prepend the CLS token to the sequence.
        # Expand the single CLS token to match the batch size.
        # CLS token shape: (1, 1, embed_dim) -> (B, 1, embed_dim)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        # Concatenate CLS token with the feature sequence along the sequence dimension.
        # Shape: (B, 1 + H'*W', embed_dim)
        seq_features = torch.cat((cls_tokens, seq_features), dim=1)
        
        # Step 7: Pass the sequence through the Transformer to capture global context (Self-Attention).
        # Shape remains: (B, sequence_length, embed_dim)
        transformer_out = self.transformer(seq_features)
        
        # Step 8: Extract the processed CLS token (the first token in the sequence).
        # This token has aggregated the global context from the entire image via attention.
        # Shape: (B, embed_dim)
        cls_out = transformer_out[:, 0, :]
        
        # Step 9: Pass the CLS token through the classification head to get class logits.
        # Output shape: (B, num_classes)
        logits = self.classifier(cls_out)
        
        return logits
        
    def freeze_cnn_backbone(self):
        """
        Freezes the weights of the pre-trained CNN backbone.
        
        Call this before starting the initial training phase to ensure that only the 
        newly added projection layer, Transformer, and classification head are 
        updated. This prevents the pre-trained features from being destroyed by 
        large initial gradients coming from the untrained transformer.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        print("CNN backbone weights have been frozen. Only Transformer and Head will be trained.")
        
    def unfreeze_cnn_backbone(self):
        """
        Unfreezes the weights of the CNN backbone for fine-tuning.
        
        After the Transformer head has converged, you can call this to fine-tune 
        the entire architecture end-to-end with a smaller learning rate.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True
            
        print("CNN backbone weights have been unfrozen for end-to-end fine-tuning.")

# Quick test snippet to verify shapes if run directly
if __name__ == "__main__":
    model = HybridCNNViT(num_classes=38)
    
    # Example to freeze the backbone for initial training
    model.freeze_cnn_backbone()
    
    # Dummy input representing a batch of 8 RGB images of size 224x224
    dummy_input = torch.randn(8, 3, 224, 224)
    
    # Forward pass
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # Should be (8, 38)
