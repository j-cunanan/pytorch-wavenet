import time
import torch
import tensorflow as tf  # For tensorboard logging
from wavenet_model import WaveNetModel, load_latest_model_from
from audio_data import WavenetDataset
from wavenet_training import WavenetTrainer, generate_audio
from model_logging import TensorboardLogger
from scipy.io import wavfile

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Model initialization
model = WaveNetModel(
    layers=10,
    blocks=3,
    dilation_channels=32,
    residual_channels=32,
    skip_channels=1024,
    end_channels=512,
    output_length=16,
    bias=True
).to(device)

# Uncomment to load a previously saved model
# model = load_latest_model_from('snapshots')
# model.to(device)
# or
# model.load_state_dict(torch.load('snapshots/some_model'))

print('Model architecture:', model)
print('Receptive field:', model.receptive_field)
print('Parameter count:', model.parameter_count())

# Dataset initialization
data = WavenetDataset(
    dataset_file='train_samples/bach_chaconne/dataset.npz',
    item_length=model.receptive_field + model.output_length - 1,
    target_length=model.output_length,
    file_location='train_samples/bach_chaconne',
    test_stride=500
)
print(f'The dataset has {len(data)} items')

def generate_and_log_samples(step):
    sample_length = 32000
    
    # Load the latest model for generation
    gen_model = load_latest_model_from('snapshots', device='cpu')
    print("Starting audio generation...")
    
    with torch.no_grad():
        # Generate samples with temperature 0.5
        print("Generating samples with temperature 0.5...")
        samples = generate_audio(
            gen_model,
            length=sample_length,
            temperatures=[0.5]
        )
        tf_samples = tf.convert_to_tensor(samples, dtype=tf.float32)
        logger.audio_summary('temperature_0.5', tf_samples, step, sr=16000)

        # Generate samples with temperature 1.0
        print("Generating samples with temperature 1.0...")
        samples = generate_audio(
            gen_model,
            length=sample_length,
            temperatures=[1.]
        )
        tf_samples = tf.convert_to_tensor(samples, dtype=tf.float32)
        logger.audio_summary('temperature_1.0', tf_samples, step, sr=16000)
    print("Audio generation completed")

# Initialize logger
logger = TensorboardLogger(
    log_interval=200,
    validation_interval=400,
    generate_interval=800,
    generate_function=generate_and_log_samples,
    log_dir="logs/chaconne_model"
)

# Initialize trainer
trainer = WavenetTrainer(
    model=model,
    dataset=data,
    lr=0.0001,
    weight_decay=0.0,
    snapshot_path='snapshots',
    snapshot_name='chaconne_model',
    snapshot_interval=1000,
    logger=logger
)

# Start training
print('Starting training...')
trainer.train(
    batch_size=16,
    epochs=10,
    continue_training_at_step=0
)