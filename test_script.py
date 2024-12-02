import time
import torch
import tensorflow as tf
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
    layers=8,
    blocks=4,
    dilation_channels=16,
    residual_channels=16,
    skip_channels=16,
    output_length=8,
    bias=True
).to(device)

# Dataset initialization
data = WavenetDataset(
    dataset_file='train_samples/saber/dataset.npz',
    item_length=model.receptive_field + model.output_length - 1,
    target_length=model.output_length,
    file_location='train_samples/saber',
    test_stride=20
)

print(f'The dataset has {len(data)} items')
print(f'Model architecture: {model}')
print(f'Receptive field: {model.receptive_field}')
print(f'Parameter count: {model.parameter_count()}')

def generate_and_log_samples(step):
    sample_length = 4000
    
    # Load the latest model for generation
    gen_model = load_latest_model_from('snapshots', device='cpu')
    print("Starting audio generation...")
    
    with torch.no_grad():
        # Generate samples with temperature 0
        print("Generating samples with temperature 0...")
        samples = generate_audio(
            gen_model,
            length=sample_length,
            temperatures=[0]
        )
        tf_samples = tf.convert_to_tensor(samples, dtype=tf.float32)
        logger.log_audio('temperature_0', tf_samples, step, sr=16000)

        # Generate samples with temperature 0.5
        print("Generating samples with temperature 0.5...")
        samples = generate_audio(
            gen_model,
            length=sample_length,
            temperatures=[0.5]
        )
        tf_samples = tf.convert_to_tensor(samples, dtype=tf.float32)
        logger.log_audio('temperature_0.5', tf_samples, step, sr=16000)
    
    print("Audio generation completed")

# Initialize logger
logger = TensorboardLogger(
    log_interval=200,
    validation_interval=200,
    generate_interval=500,
    generate_function=generate_and_log_samples,
    log_dir="logs"
)

# Initialize trainer
trainer = WavenetTrainer(
    model=model,
    dataset=data,
    lr=0.0001,
    weight_decay=0.1,
    logger=logger,
    snapshot_path='snapshots',
    snapshot_name='saber_model',
    snapshot_interval=500
)

# Training
print('Starting training...')
start_time = time.time()

try:
    trainer.train(
        batch_size=8,
        epochs=20
    )
except KeyboardInterrupt:
    print("Training interrupted by user")
finally:
    end_time = time.time()
    duration = end_time - start_time
    hours = duration // 3600
    minutes = (duration % 3600) // 60
    seconds = duration % 60
    print(f'Training took {hours:.0f}h {minutes:.0f}m {seconds:.1f}s')