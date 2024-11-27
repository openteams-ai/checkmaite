# Details on the test images for `survivor` and `reallabel`

## `survivor`

`survivor` distinguishes (using metric scores) between three types of images in a dataset:
    - easy images: images that a large number of models agree on and have high confidence in
    - hard images: images that a large number of models agree on and have low confidence in
    - 'on-the-bubble' images: images where there is little consenus among models

The basic premise is that on-the-bubble images are most useful for understanding and evaluating models.

The test image has been generated using synthetic metric data that should result in `survivor` identifying
2 easy images, 2 hard images and 2 on-the-bubble images.

## `reallabel`

`reallabel` distinguishes (using bounding box intersection algorithms from model predictions)
between three types of labels in a ground-truth dataset:
    - likely wrong: labels in a ground-truth dataset that are likely incorrect
    - likely missed: labels that are not in a ground-truth dataset and have likely been missed
    - likely correct: labels that in a ground-truth dataset and are likely correct

The basic premise is that if there is disagreement between ground-truth and a large number of
(confident) model predictions, then this likely reflects an error in the ground-truth.

The test image has been generated using synthetic inference data for an image (a kitchen) in the test
suite. The synthetic inference data was created so as to have high agreement with the ground-truth
for the fridge ('likely correct') and then disagreement for the dining table ('likely wrong') and
the oven ('likely missed').
