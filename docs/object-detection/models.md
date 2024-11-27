# Model Wrappers

`jatic_ri` provides access to several common object-detection models using MAITE-compliant wrappers.

## `torchvision` Wrappers

The `torchvision` wrapper provides access to pre-trained object detection models from the `torchvision` package. Users have two main options for leveraging this wrapper:

### 1. Using Pre-Trained Weights from `torchvision`
This is the simplest option:
- Users only need to specify the name of the model they wish to use.
- Optionally, they can specify the device (cpu or cuda) on which they want to run the model.

### 2. Using Custom Pre-Trained Weights
This option is recommended for advanced users:
- Users can provide a pickle file containing pre-trained weights via the `pickle_path` keyword argument. The pickle file is expected
to contain only the pre-trained weights (i.e., the `state_dict`, and created using something similar to `torch.save(model.state_dict())`).
- Additionally, they must supply a configuration file via the `config_path` keyword argument. The configuration file should include:
    - A dictionary of class labels corresponding to the model's output categories, stored under the key `index2label`.
    - (Optional) The total number of class labels (`num_classes`), if different from the torchvision defaults.
- Users can customize the key for class labels if `index2label` is not used.


### Additional Notes

- **Pickle File Assumptions**:
    - The pickle file is expected to contain only the pre-trained weights (i.e., the `state_dict`).
    - If the pickle file includes additional information (e.g., model architecture), this will cause an error.

- **Image Dimensions**:
    - It is assumed that all images passed to the model for prediction have the same height and width.
    - If this assumption does not hold, the wrapper will raise an error. Please contact the RI team if your use case involves images with unequal dimensions.
