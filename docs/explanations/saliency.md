# Saliency and explainability

A computer vision model takes an image and returns a prediction, such as a
class label or a set of boxes. The prediction comes from the whole image at
once: nothing in the output says which parts of the input produced it. Yet
test and evaluation (T&E) often needs exactly that. Why did the model call
this camel a horse? What in a kitchen scene made it predict "beach ball"? Would
it still detect the dog if the second dog were not in the frame?

**Saliency algorithms** answer questions like these. A saliency algorithm
estimates which regions of an input most influenced a model's prediction. Its
result is a **saliency map**: a heatmap laid over the image, where warmer
regions mattered more to the prediction being explained.

![D-RISE saliency map for one detection](../assets/saliency_map.png)

*A saliency map for one bicycle detection (red box). The warm region shows the
pixels the detector's score for that box depended on most.*

## White-box and black-box explanation

An explanation method (XAI) needs some way to interact with the model it
explains (the AI). There are two broad approaches.

**White-box** methods open the model and read its internal state while it
makes a prediction. Grad-CAM, for example, combines a network's internal
feature maps with the gradients of its output.

- They are efficient, typically one forward and backward pass per
  explanation, and they use what the model actually computed.
- They are tied to a model architecture. An implementation for one network
  may not work for another, may require modifying the model to expose its
  internals, and must be maintained as the model changes.

**Black-box** methods never look inside the model. Instead, they ask the model
a series of related questions: they run it on many altered copies of the input
and compare the answers with the original prediction.

- They are model-agnostic. Anything with the same inputs and outputs can be
  explained, including models that cannot be exposed for security or
  contractual reasons, and the explanation keeps working as the model evolves.
- They are expensive, because the model runs many times per explanation. They
  also observe the model's behavior only indirectly: they show how outputs
  respond to changes in the input, not how the model computed them.

Because T&E usually treats the model under test as an opaque system, black-box
methods are a natural fit for it.

## How black-box saliency works

Black-box saliency for images works in two steps:

1. **Image perturbation.** Generate many modified copies of the input, for
   example by occluding random regions with masks.
2. **Heatmap generation.** Run the model on every copy, measure how its output
   changed relative to the original, and turn those changes into a per-pixel
   saliency score.

Regions whose occlusion consistently lowers the model's confidence in the
explained prediction are scored as salient. Keeping the two steps separate
lets a perturbation strategy be combined with different scoring methods.

**RISE** (Randomized Input Sampling for Explanation) is a widely used example
for image classification. It occludes the input with many random masks and
weights each mask by the model's confidence in the class being explained. The
RISE authors used up to 8,000 masked copies per image, which shows the cost of
the approach. Variants extend it:

- **MC-RISE** (Multi-Color RISE) uses colored fills rather than a single
  occlusion color, which can show how color contributes to a prediction.
- **D-RISE** applies the same idea to object detectors. Instead of one class
  score, it scores how well each masked copy reproduces the detection being
  explained, using both its box and its class.

## Limitations

- **Cost.** Every image needs one model run per mask, often several hundred.
  Fewer masks are faster but give noisier maps. To keep runs affordable, run
  on a sample of images (for example the model's errors), lower the mask
  count, and for object detection set `saliency_max_side` to the model's
  input size.
- **Settings matter.** Mask count, mask size, and occlusion probability all
  change the result. Compare saliency maps only when they were produced with
  the same settings.
- **Behavior, not mechanism.** A saliency map shows which inputs the
  prediction was sensitive to. It does not show how the model reasons, and it
  cannot relate its findings to anything inside the model.
- **One prediction at a time.** A map explains a single prediction for a
  single image. Conclusions about a model need maps across many images.
- **Plausible is not faithful.** A map can look convincing without reflecting
  what the model used, and studies have found that saliency maps help people
  predict or debug models less than expected. Black-box methods such as RISE
  and D-RISE only use the model's outputs, which avoids some of these
  problems, but their maps are still coarse estimates. Use saliency to screen
  for gross shortcuts, such as a detector that relies on the background or a
  watermark, and to choose which predictions to inspect. Do not treat a
  plausible map as evidence that the model is right for the right reasons.

## In CheckMAITE

The [XAITK tutorial](../tool-usage/xaitk_tutorial.ipynb) shows the
`XaitkExplainable` capability, which produces saliency maps with the black-box
algorithms in [xaitk-saliency](https://github.com/XAITK/xaitk-saliency):

- Image classification uses RISE by default, with 50 masks, and also accepts
  other xaitk-saliency classifier algorithms such as MC-RISE.
- Object detection uses D-RISE by default, with 20 masks.

The default mask counts are small so that runs finish quickly. Increase them
when you need stable maps; the tutorial shows how mask count and size change
the result.

### Saliency generators

The `saliency_generator` field of the config takes any xaitk-saliency
generator by its plugin type name. These are tested in CheckMAITE:

Image classification:

- `rise.RISEStack`, the default.
- `mc_rise.MCRISEStack`, which gives one map per fill color and class.
- `occlusion_based.PerturbationOcclusion`, which pairs any perturber, such as
  `SlidingWindow`, with a scorer, such as `OcclusionScoring`.

Object detection:

- `drise.DRISEStack`, the default.
- `drise.RandomGridStack`, D-RISE scoring with a random grid of fixed-size
  cells.
- `occlusion_based.PerturbationOcclusion`, which pairs any perturber with
  `DRISEScoring`.

The type names are relative to
`xaitk_saliency.impls.gen_image_classifier_blackbox_sal` and
`xaitk_saliency.impls.gen_object_detector_blackbox_sal`. The
`SlidingWindowStack` classifier generator is not available, because
xaitk-saliency does not register it as a plugin.

### Saliency scores

Each run also scores its saliency maps, so runs can be compared and
thresholded without reading every map. The scores are screening flags, not
measures of model quality: use them to find the maps worth opening (a missed
peak, low coverage) or to spot a change between runs, and read those maps
before drawing conclusions. The scores are in the run outputs
(`score_summary` holds the per-run count, mean, median, standard deviation,
minimum and maximum), in the Markdown report, and in the
`xaitk_explainable` analytics table.

- **Entropy** (`saliency_entropy`) is the xaitk-saliency `Entropy` metric, in
  bits. The map is min-max scaled to [0, 1] first. Low entropy means the
  saliency is concentrated in a few pixels; high entropy means it is spread
  out. `normalized_saliency_entropy` divides it by log2 of the pixel count,
  which puts it in [0, 1] and makes images of different sizes comparable.
  Smooth maps tend to score close to 1, so look at differences between runs
  with the same generator settings rather than at the absolute value.
- **Ground-truth saliency coverage** (object detection only,
  `gt_saliency_coverage`) is the share of a detection's above-average
  saliency that falls inside its matched ground-truth box. It uses
  xaitk-saliency's `compute_saliency_coverage` on the map minus its mean,
  clipped at 0, with the box as the mask. The mean is removed because a
  D-RISE map averages the detection's score over the masks that leave each
  pixel visible, so every pixel sits near the average score; only the excess
  shows what the detection relied on. A value near 1 means the detector
  relied on the object; a low value means it relied on the surroundings.
- **Peak in box** (`gt_saliency_peak_in_box`) is the pointing game: whether
  the map's highest pixel lies inside the matched ground-truth box. It does
  not depend on the box's size, and its mean over a run is the hit rate. A
  hit rate near 1 means the explanations sit on the objects; the misses are
  the detections worth opening. `gt_saliency_peak_distance` shows how far
  outside a miss is: the distance in pixels, at the working resolution, from
  the map's highest pixel to the box, and 0 for a hit. On objects smaller
  than a D-RISE grid cell, a miss a few pixels outside the box is usually a
  near-tie at the box edge, so check the distance before treating a miss as
  a real failure.

A detection is matched to the ground-truth box it overlaps most, if that IoU
is at least `gt_match_iou_threshold` (0.5 by default). Matching ignores class
labels, so a detection with the wrong label can still be matched to the
object it covers, and several detections can match the same box. Detections
with no match have no coverage score; the matched IoU is stored as
`matched_gt_iou`.

For object detection, each image explains up to `dets_limit` detections
(10 by default), highest score first. The model runs once per perturbed
image however many detections are explained, so the limit mostly sets how
many maps are stored and reported. Set `min_score` to explain only
detections the model is confident about, such as its operating threshold.
This matters for models that return every candidate box, such as ONNX
models, which modelmaite returns without a score threshold or NMS. Images
with no detections left get no saliency maps and no records.

Saliency generation holds every mask and every masked image in memory at
once, so memory grows with the number of masks times the image's pixel
count. For example, 500 D-RISE masks on a 1738 by 1610 image peaked at about
23 GB. For object detection, set `saliency_max_side` to run at a smaller
size, ideally the model's own input size: memory falls with the square of
the scale, and the outputs are at that size, with `image_scale` recording
the factor. Otherwise use fewer masks.

For image classification, each image is scored on the map of its
ground-truth class, or of the predicted class when the image has no
ground-truth class (an all-zero target). MC-RISE gives one score per fill
color.

## Further reading

- [Concepts of Saliency and Explainability in
  AI](https://xaitk-saliency.readthedocs.io/en/latest/xaitk_explanation.html),
  the xaitk-saliency explanation that this page follows.
- Petsiuk, V., Das, A., & Saenko, K. (2018). RISE: Randomized input sampling
  for explanation of black-box models. [arXiv:1806.07421](https://arxiv.org/abs/1806.07421)
- Petsiuk, V., et al. (2021). Black-box explanation of object detectors via
  saliency maps (D-RISE). [arXiv:2006.03204](https://arxiv.org/abs/2006.03204)
- Adebayo, J., et al. (2018). Sanity checks for saliency maps.
  [arXiv:1810.03292](https://arxiv.org/abs/1810.03292)
- Tomsett, R., et al. (2020). Sanity checks for saliency metrics.
  [arXiv:1912.01451](https://arxiv.org/abs/1912.01451)
- Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep
  networks via gradient-based localization.
  [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)
