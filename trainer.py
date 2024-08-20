import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras import layers, Model, Input, regularizers, backend as K
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Concatenate
from tensorflow.keras.utils import Sequence
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from kerastuner import RandomSearch

class DataAugmentation(Sequence):
    def __init__(self, pairs, labels, batch_size=32, shuffle=True):
        self.pairs = pairs
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.pairs))
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.floor(len(self.pairs) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_pairs = self.pairs[indices]
        batch_labels = self.labels[indices]
        augmented_pairs = self._augment_data(batch_pairs)
        return [augmented_pairs[:, 0], augmented_pairs[:, 1]], batch_labels

    def _augment_data(self, data):
        return data

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def create_graph_siamese_network(input_shape):
    input_node = Input(shape=input_shape)
    input_adj = Input(shape=(None, None))
    x = Conv1D(64, kernel_size=3, activation='relu')(input_node)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    encoded = Model(inputs=input_node, outputs=x)
    return encoded

def create_attention_module(input_shape):
    def attention_module(inputs):
        x = Dense(64, activation='relu')(inputs)
        x = Dense(1, activation='sigmoid')(x)
        return x
    return attention_module

def create_inception_module(input_shape):
    def inception_module(inputs):
        branch_a = Conv1D(32, kernel_size=1, activation='relu')(inputs)
        branch_a = Conv1D(64, kernel_size=3, padding='same', activation='relu')(branch_a)
        branch_b = Conv1D(64, kernel_size=1, activation='relu')(inputs)
        branch_b = Conv1D(128, kernel_size=5, padding='same', activation='relu')(branch_b)
        branches = Concatenate(axis=-1)([branch_a, branch_b])
        return branches
    return inception_module

def build_model(hp):
    input_shape = (50,)
    input_node = Input(shape=input_shape)
    input_adj = Input(shape=(None, None))
    graph_siamese_network = create_graph_siamese_network(input_shape)
    encoded_a = graph_siamese_network(input_node)
    encoded_b = graph_siamese_network(input_node)
    attention_a = create_attention_module(input_shape)(encoded_a)
    attention_b = create_attention_module(input_shape)(encoded_b)
    inception_a = create_inception_module(input_shape)(attention_a)
    inception_b = create_inception_module(input_shape)(attention_b)
    distance = K.sum(K.square(inception_a - inception_b), axis=1, keepdims=True)
    similarity = layers.Activation('sigmoid')(distance)
    model = Model(inputs=[input_node, input_adj], outputs=similarity)
    model.compile(
        loss=AdvancedCompositeLoss(
            margin_contrastive=hp.Float('margin_contrastive', 0.5, 1.5, step=0.1),
            margin_triplet=hp.Float('margin_triplet', 0.5, 1.5, step=0.1),
            lambda_reg=hp.Float('lambda_reg', 0.001, 0.1, step=0.001),
            alpha_focal=hp.Float('alpha_focal', 0.1, 0.5, step=0.1),
            gamma_focal=hp.Float('gamma_focal', 1.0, 5.0, step=0.5)
        ),
        optimizer=hp.Choice('optimizer', ['adam', 'sgd']),
        metrics=['accuracy']
    )
    return model

class LearningRateScheduler(Callback):
    def __init__(self, initial_lr=1e-3, decay_rate=0.9):
        super().__init__()
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate

    def on_epoch_end(self, epoch, logs=None):
        new_lr = self.initial_lr * (self.decay_rate ** epoch)
        K.set_value(self.model.optimizer.lr, new_lr)
        print(f'\nEpoch {epoch+1}: Learning rate is {new_lr:.5f}')

def main():
    input_file_path = 'dataset.h5'
    output_file_path = 'filtered_and_reduced_data.tif'
    metadata = {
        'x_min': 0, 
        'y_max': 0, 
        'pixel_size_x': 1, 
        'pixel_size_y': 1, 
        'epsg_code': 4326
    }
    
    processor = HyperspectralDataProcessor(input_file_path, use_gpu=True)
    processor.open_dataset()
    
    data2D = processor.reshape_to_2D()
    encoding_dim = 50
    reduced_data_autoencoder = processor.apply_autoencoder(data2D, encoding_dim)
    
    n_components_pca = 10
    pca_reduced_data = processor.apply_pca(data2D, n_components_pca)
    
    sigma_color = 0.1
    sigma_spatial = 15
    filtered_data_bilateral = processor.apply_bilateral_filtering(
        reduced_data_autoencoder.reshape(n_components_pca, -1).reshape(n_components_pca, 100, 100),
        sigma_color, sigma_spatial
    )
    
    num_endmembers = 10
    endmembers = processor.vertex_component_analysis(data2D, num_endmembers)
    
    n_components_dict = 15
    dictionary_components = processor.apply_dictionary_learning(data2D, n_components_dict)
    
    processor.save_as_geotiff(filtered_data_bilateral, output_file_path, metadata)
    
    labels = np.random.randint(0, 2, reduced_data_autoencoder.shape[0])
    pairs, pair_labels = create_pairs(reduced_data_autoencoder, labels)
    
    scaler = StandardScaler()
    pairs = scaler.fit_transform(pairs.reshape(-1, pairs.shape[-1])).reshape(pairs.shape)
    
    train_pairs, val_pairs, train_labels, val_labels = train_test_split(pairs, pair_labels, test_size=0.2, random_state=42)
    
    input_shape = (train_pairs.shape[2],)
    
    tuner = RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=20,
        executions_per_trial=1,
        directory='my_dir',
        project_name='advanced_siamese'
    )
    tuner.search(train_pairs[:, 0], train_labels, validation_data=(val_pairs[:, 0], val_labels))
    
    best_model = tuner.get_best_models(num_models=1)[0]
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_scheduler = LearningRateScheduler(initial_lr=1e-3, decay_rate=0.9)
    
    train_generator = DataAugmentation(train_pairs, train_labels, batch_size=32)
    val_generator = DataAugmentation(val_pairs, val_labels, batch_size=32, shuffle=False)
    
    best_model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=20,
        callbacks=[early_stopping, lr_scheduler]
    )
    
    test_pairs, test_labels = create_pairs(reduced_data_autoencoder, labels)
    test_pairs = scaler.transform(test_pairs.reshape(-1, test_pairs.shape[-1])).reshape(test_pairs.shape)
    
    predictions = best_model.predict([test_pairs[:, 0], input_adj])
    predicted_labels = (predictions > 0.5).astype(int)
    
    processor.close_dataset()
    print(f"Predictions: {predicted_labels}")

if __name__ == '__main__':
    main()
