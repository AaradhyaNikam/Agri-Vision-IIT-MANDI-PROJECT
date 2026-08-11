import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Load your pre-trained baseline Keras model
base_model = models.load_model('models/model.keras')

# Freeze pre-trained weights to preserve learned features
base_model.trainable = False

# 2. Extract the spatial feature map directly from the MobileNetV2 layer
# The error log confirms this layer is named 'mobilenetv2_1.00_224'
# Output shape will be (Batch, 7, 7, 1280)
feature_extractor_output = base_model.get_layer('mobilenetv2_1.00_224').output 

# 3. Reshape 2D spatial feature map (7, 7, 1280) into a 1D sequence of tokens (49, 1280)
# 7 * 7 = 49 sequence patches
c_dim = feature_extractor_output.shape[-1]
seq_features = layers.Reshape((49, c_dim))(feature_extractor_output)

# 4. Vision Transformer Block: Multi-Head Self-Attention
attention_output = layers.MultiHeadAttention(num_heads=8, key_dim=256)(
    query=seq_features, 
    value=seq_features
)

# Residual connection & Layer Normalization
attention_output = layers.LayerNormalization()(seq_features + attention_output)

# 5. Global Pooling & Final Classification Head (38 PlantVillage classes)
global_tokens = layers.GlobalAveragePooling1D()(attention_output)
dropout_out = layers.Dropout(0.2)(global_tokens)
final_predictions = layers.Dense(38, activation='softmax')(dropout_out)

# 6. Construct the Hybrid Keras Model
# We input to the base_model, but output from our new Transformer pipeline
hybrid_model = models.Model(inputs=base_model.input, outputs=final_predictions)

# 7. Compile the Hybrid Model
hybrid_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Save the text summary to the images folder with UTF-8 encoding
summary_path = 'images/hybrid_model_summary.txt'
with open(summary_path, 'w', encoding='utf-8') as f:
    hybrid_model.summary(print_fn=lambda x: f.write(x + '\n'))

print(f"\nModel summary table successfully saved to {summary_path}")