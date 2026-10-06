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

- **Cost.** Every saliency map needs many model runs. Fewer masks are faster
  but give noisier maps.
- **Settings matter.** Mask count, mask size, and occlusion probability all
  change the result. Compare saliency maps only when they were produced with
  the same settings.
- **Behavior, not mechanism.** A saliency map shows which inputs the
  prediction was sensitive to. It does not show how the model reasons, and it
  cannot relate its findings to anything inside the model.
- **One prediction at a time.** A map explains a single prediction for a
  single image. Conclusions about a model need maps across many images.

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

## Further reading

- [Concepts of Saliency and Explainability in
  AI](https://xaitk-saliency.readthedocs.io/en/latest/xaitk_explanation.html),
  the xaitk-saliency explanation that this page follows.
- Petsiuk, V., Das, A., & Saenko, K. (2018). RISE: Randomized input sampling
  for explanation of black-box models. [arXiv:1806.07421](https://arxiv.org/abs/1806.07421)
- Petsiuk, V., et al. (2021). Black-box explanation of object detectors via
  saliency maps (D-RISE). [arXiv:2006.03204](https://arxiv.org/abs/2006.03204)
- Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep
  networks via gradient-based localization.
  [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)
