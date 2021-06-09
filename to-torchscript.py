import torch
import torchvision

from dh_segment_torch.training import Trainer

params = {
  "color_labels": {"label_json_file": '/media/data/deepLearning/dhsegment/test/color_labels.json'}, # Color labels produced before
  "train_dataset": {
    "type": "image_csv", # Image csv dataset
    "csv_filename": "/media/data/deepLearning/dhsegment/test/train.csv",
    "base_dir": "/media/data/deepLearning/dhsegment/test",
    "repeat_dataset": 4, # Repeat 4 times the data since we have little
    "compose": {"transforms": [{"type": "fixed_size_resize", "output_size": 1e6}]} # Resize to a fixed size, could add other transformations.
  },
  "val_dataset": {
    "type": "image_csv", # Validation dataset
    "csv_filename": "/media/data/deepLearning/dhsegment/test/val.csv",
    "base_dir": "/media/data/deepLearning/dhsegment/test",
    "compose": {"transforms": [{"type": "fixed_size_resize", "output_size": 1e6}]}
  },
  "model": { # Model definition, original dhSegment
    "encoder": "resnet50",
    "decoder": {
      "decoder_channels": [512, 256, 128, 64, 32],
      "max_channels": 512
    }
  },
  "metrics": [['miou', 'iou'], ['iou', {"type": 'iou', "average": None}], 'precision'], # Metrics to compute
  "optimizer": {"lr": 1e-4}, # Learning rate
  "lr_scheduler": {"type": "exponential", "gamma": 0.9995}, # Exponential decreasing learning rate
  "val_metric": "+miou", # Metric to observe to consider a model better than another, the + indicates that we want to maximize
  "early_stopping": {"patience": 4}, # Number of validation steps without increase to tolerate, stops if reached
  "model_out_dir": "/media/data/deepLearning/dhsegment/test-model", # Path to model output
  "num_epochs": 100, # Number of epochs for training
  "evaluate_every_epoch": 5, # Number of epochs between each validation of the model
  "batch_size": 4, # Batch size (to be changed if the allocated GPU has little memory)
  "num_data_workers": 0,
  "track_train_metrics": False,
  "loggers": [
    {"type": 'tensorboard', "log_dir": "/media/data/deepLearning/dhsegment/test-log", "log_every": 5, "log_images_every": 10}, # Tensorboard logging
  ],
  "device": "cpu"
}

device_name = "cpu"
trainer = Trainer.from_params(params)

device = torch.device(device_name)

print(trainer.device)

trainer_checkpoint = '/media/data/deepLearning/dhsegment/wip-1000-blocks-models/train_trainer_checkpoint_iter=13500.pth'
model_checkpoint = '/media/data/deepLearning/dhsegment/wip-1000-blocks-models/train_model_checkpoint_iter=13500.pth'

if 'trainer_checkpoint' in vars() or 'trainer_checkpoint' in globals():
  print("Loading previous model")
  state_dict = {}

  state_dict = torch.load(trainer_checkpoint, map_location=device)

  state_dict["model"] = torch.load(model_checkpoint, map_location=device)
  state_dict["device"] = device_name

  from collections.abc import Mapping

  def state_to_device(state, device):
    for key, param in state.items():
      if isinstance(param, torch.Tensor):
        param.data = param.data.to(device)
        if param._grad is not None:
          param._grad.data = param._grad.data.to(device)
      elif isinstance(param, Mapping):
        state_to_device(param, device)

  state_to_device(state_dict, device)

  trainer.load_state_dict(state_dict)

  optimizer = trainer.optimizer
  state_to_device(optimizer.state,device)

trainer.device = device
trainer.model.to(device)

# Switch the model to eval model
trainer.model.eval()

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 1000, 1000, dtype=torch.float32)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(trainer.model, example, strict=False)

# Save the TorchScript model
traced_script_module.save("/media/data/deepLearning/dhsegment/wip-1000-blocks-models/dhsegment_torchscript_model.pt")