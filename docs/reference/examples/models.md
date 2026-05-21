# Model Wrappers

`checkmaite` provides access to several common object-detection models using MAITE-compliant wrappers.

## `torchvision` Wrappers

The `torchvision` wrapper provides access to pre-trained object detection models from the `torchvision` package. Users have two main options for leveraging this wrapper:

### 1. Using Pre-Trained Weights from `torchvision`

This is the simplest option:

- Users only need to specify the name of the model they wish to use.
- Optionally, they can specify the device (cpu or cuda) on which they want to run the model.

### 2. Using Custom Pre-Trained Weights

This option is recommended for advanced users:

- Users can provide a pickle file containing pre-trained weights via the `pickle_path` keyword argument. The pickle file is expected to contain only the pre-trained weights (i.e., the `state_dict`, created using something similar to `torch.save(model.state_dict())`).
- Additionally, they must supply a configuration file via the `config_path` keyword argument. This is checkmaite wrapper metadata, not a torchvision-native config file.
- The configuration file must include class labels under the key `index2label` by default. Users can customize this key with the wrapper's `index2label_key` argument.
- The optional `num_classes` field should be included when the custom weights use a different number of classes than the torchvision default.

Example config file:

```json
{
  "index2label": {
    "0": "background",
    "1": "person",
    "2": "vehicle"
  },
  "num_classes": 3
}
```

`index2label` may also be provided as a list:

```json
{
  "index2label": ["background", "person", "vehicle"],
  "num_classes": 3
}
```

Dictionary keys are converted to integers when the wrapper loads the config.

### Additional Notes

#### Pickle File Assumptions

- The pickle file is expected to contain only the pre-trained weights (i.e., the `state_dict`).
- If the pickle file includes additional information (e.g., model architecture), this will cause an error.

#### Image Dimensions

- It is assumed that all images passed to the model for prediction have the same height and width.
- If this assumption does not hold, the wrapper will raise an error. Please contact the checkmaite team if your use case involves images with unequal dimensions.
