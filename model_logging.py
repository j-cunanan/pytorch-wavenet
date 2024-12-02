import tensorflow as tf
import torch
import threading


class Logger:
    def __init__(self,
                 log_interval=50,
                 validation_interval=200,
                 generate_interval=500,
                 trainer=None,
                 generate_function=None):
        self.trainer = trainer
        self.log_interval = log_interval
        self.validation_interval = validation_interval
        self.generate_interval = generate_interval
        self.accumulated_loss = 0
        self.generate_function = generate_function
        if self.generate_function is not None:
            self.generate_thread = threading.Thread(target=self.generate_function)
            self.generate_thread.daemon = True

    def log(self, current_step, current_loss):
        self.accumulated_loss += current_loss
        if current_step % self.log_interval == 0:
            self.log_loss(current_step)
            self.accumulated_loss = 0
        if current_step % self.validation_interval == 0:
            self.validate(current_step)
        if current_step % self.generate_interval == 0:
            self.generate(current_step)

    def log_loss(self, current_step):
        avg_loss = self.accumulated_loss / self.log_interval
        print(f"loss at step {current_step}: {avg_loss}")

    def validate(self, current_step):
        avg_loss, avg_accuracy = self.trainer.validate()
        print(f"validation loss: {avg_loss}")
        print(f"validation accuracy: {avg_accuracy * 100}%")

    def generate(self, current_step):
        if self.generate_function is None:
            return

        if self.generate_thread.is_alive():
            print("Last generate is still running, skipping this one")
        else:
            self.generate_thread = threading.Thread(
                target=self.generate_function,
                args=[current_step]
            )
            self.generate_thread.daemon = True
            self.generate_thread.start()


class TensorboardLogger(Logger):
    def __init__(self,
                 log_interval=50,
                 validation_interval=200,
                 generate_interval=500,
                 trainer=None,
                 generate_function=None,
                 log_dir='logs'):
        super().__init__(log_interval, validation_interval, generate_interval, trainer, generate_function)
        self.writer = tf.summary.create_file_writer(log_dir)

    def log_loss(self, current_step):
        # loss
        avg_loss = self.accumulated_loss / self.log_interval
        with self.writer.as_default():
            tf.summary.scalar('loss', avg_loss, step=current_step)

            # parameter histograms
            for tag, value in self.trainer.model.named_parameters():
                tag = tag.replace('.', '/')
                # Detach tensor and move to CPU before converting to numpy
                with torch.no_grad():
                    tensor_value = value.detach().cpu().numpy()
                    if value.grad is not None:
                        grad_value = value.grad.detach().cpu().numpy()
                    else:
                        grad_value = None
                
                tf.summary.histogram(tag, tensor_value, step=current_step)
                if grad_value is not None:
                    tf.summary.histogram(f"{tag}/grad", grad_value, step=current_step)

    def validate(self, current_step):
        avg_loss, avg_accuracy = self.trainer.validate()
        with self.writer.as_default():
            tf.summary.scalar('validation loss', avg_loss, step=current_step)
            tf.summary.scalar('validation accuracy', avg_accuracy, step=current_step)

    def log_audio(self, tag, sample, step, sr=16000):
        with self.writer.as_default():
            tf.summary.audio(tag, sample, sample_rate=sr, step=step, max_outputs=4)

    def tensor_summary(self, tag, tensor, step):
        with torch.no_grad():
            tensor_value = tensor.detach().cpu().numpy()
        with self.writer.as_default():
            tf.summary.scalar(tag, tensor_value, step=step)