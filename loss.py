import tensorflow as tf
from tensorflow.keras import backend as K

class AdvancedCompositeLoss(tf.keras.losses.Loss):
    def __init__(self, margin_contrastive=1.0, margin_triplet=1.0, lambda_reg=0.01, kl_target=None, alpha_focal=0.25, gamma_focal=2.0, **kwargs):
        super(AdvancedCompositeLoss, self).__init__(**kwargs)
        self.margin_contrastive = margin_contrastive
        self.margin_triplet = margin_triplet
        self.lambda_reg = lambda_reg
        self.kl_target = kl_target
        self.alpha_focal = alpha_focal
        self.gamma_focal = gamma_focal
    
    def call(self, y_true, y_pred):
        # Contrastive Loss with Adaptive Margin
        margin_adaptive = self.margin_contrastive * (1 + K.exp(-y_pred))  # Adaptive margin based on predicted distance
        contrastive = K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin_adaptive - y_pred, 0)), axis=-1)
        
        # Triplet Loss with Hard-Negative Mining
        positive_pairs = K.mean(y_true * K.square(y_pred), axis=-1)
        negative_pairs = K.mean((1 - y_true) * K.square(K.maximum(self.margin_triplet - y_pred, 0)), axis=-1)
        triplet_loss = K.maximum(positive_pairs - negative_pairs + self.margin_triplet, 0)
        
        # L2 Regularization applied to specific layers
        regularization_loss = self.lambda_reg * (K.sum(K.square(K.concatenate([K.get_value(layer.kernel) for layer in self.model.layers if hasattr(layer, 'kernel')], axis=0))) 
                                                + K.sum(K.square(K.concatenate([K.get_value(layer.bias) for layer in self.model.layers if hasattr(layer, 'bias')], axis=0))))
        
        # KL Divergence with Target Probability Distribution
        if self.kl_target is not None:
            kl_divergence = K.sum(self.kl_target * K.log(K.clip(self.kl_target / (y_pred + K.epsilon()), K.epsilon(), None)) - self.kl_target + y_pred, axis=-1)
        else:
            kl_divergence = 0
        
        # Focal Loss
        focal_loss = -self.alpha_focal * K.pow(1 - y_pred, self.gamma_focal) * K.log(K.clip(y_pred, K.epsilon(), 1 - K.epsilon()))
        
        # Cosine Similarity Loss
        y_pred_normalized = K.l2_normalize(y_pred, axis=-1)
        cosine_similarity_loss = K.mean(1 - K.sum(y_true * y_pred_normalized, axis=-1))
        
        return contrastive + triplet_loss + regularization_loss + kl_divergence + focal_loss + cosine_similarity_loss

    def get_config(self):
        return {
            'margin_contrastive': self.margin_contrastive,
            'margin_triplet': self.margin_triplet,
            'lambda_reg': self.lambda_reg,
            'kl_target': self.kl_target,
            'alpha_focal': self.alpha_focal,
            'gamma_focal': self.gamma_focal
        }
