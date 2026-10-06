# Distribution shift: drift and out-of-distribution detection

A model is trained and validated on one dataset, then deployed into conditions
that keep changing: seasons turn, sensors age, missions and targets evolve.
When the data a deployed model sees differs statistically from its training
data, the decision boundaries it learned no longer fit, and performance
degrades. This gap between the training distribution and the operational
distribution is **distribution shift**. For systems fielded for years, it
should be expected, not treated as a rare edge case.

Two complementary checks address it:

- **Drift detection** works at the population level. It asks whether a batch
  of incoming data, as a whole, is statistically consistent with a reference
  set such as the training data.
- **Out-of-distribution (OOD) detection** works at the instance level. It asks
  whether an individual sample is unlike anything in the reference set.

Neither is enough alone. A batch can pass a drift test while containing a few
genuinely anomalous samples whose effect is too diluted to detect, and
flagging individual outliers does not show that a whole population has moved.

## Kinds of shift

Which part of the data has changed determines what can be done about it. For a
model that predicts labels from images, there are three kinds of shift:

- **Covariate shift**: the distribution of inputs changes, but the
  relationship between an input and its label does not. A vehicle detector
  trained mostly on clear daytime imagery and deployed in heavy fog faces
  covariate shift: a vehicle is still a vehicle, but the images now fall where
  training data was sparse.
- **Label shift**: the class balance changes, but each class still looks the
  same. A model trained with targets and non-targets in equal numbers, then
  deployed where targets are rare, will over-predict the class that used to be
  common.
- **Concept drift**: the inputs look the same, but the correct label for them
  has changed. Examples include a new vehicle variant that resembles an
  existing class but must be classified differently, or a change in labeling
  standards.

Real shift often combines all three. The distinction is useful because it
identifies what changed, and therefore what response fits.

## Detecting drift

Drift detectors are fitted on a reference set, then applied to new batches.
Each test returns a p-value and a drift flag. Images are rarely compared pixel
by pixel; instead, a feature extractor turns each image into an **embedding**,
a numeric vector, and the tests compare embeddings.

Tests differ in which kinds of change they detect best:

- **Kolmogorov-Smirnov (KS)** compares each feature separately, using the
  largest gap between the two cumulative distributions. It is the standard
  baseline and is most sensitive to shifts in the bulk of a distribution, less
  so to changes in the tails.
- **Cramér-von Mises (CVM)** also compares each feature separately, but sums
  the gaps across the whole distribution. This makes it better than KS at
  detecting changes in spread and subtle shifts, such as gradual sensor
  degradation.
- **Maximum Mean Discrepancy (MMD)** compares all features jointly. It can
  detect shift in the relationships between features, such as noise that has
  become correlated across channels, which per-feature tests miss. It is
  suited to high-dimensional embeddings, at a higher computational cost.

Per-feature tests make one test per embedding dimension, so their p-values are
corrected for multiple testing before a single drift decision is made.

## Detecting out-of-distribution samples

A common OOD method needs no training beyond the feature extractor. It indexes
the reference embeddings, then scores each new sample by its mean distance to
its *k* nearest reference neighbors. A sample far from all of them lies in a
sparse region of the embedding space and is flagged when its score exceeds a
threshold calibrated on the reference set.

This method is fast and easy to update, and its score has a direct meaning.
It is only as good as the embedding, though: if the feature extractor does not
capture the way an anomaly differs, the distance will not reveal it.

## Limitations

- **The reference set must be representative.** If the reference data is
  itself biased or misses part of the operational range, every test is
  calibrated against the wrong baseline.
- **A p-value is not a verdict.** Very large batches make tests flag
  differences too small to matter; small batches lack the power to detect real
  shift. Whether a detected shift matters needs judgment about its size and
  nature.
- **Per-feature tests miss correlations.** KS and CVM cannot see shift that
  appears only in how features relate; multivariate tests such as MMD can, but
  are costlier and harder to interpret.
- **Distances weaken in high dimensions.** As embedding dimensionality grows,
  distances between points become more uniform and nearest-neighbor scores
  less informative. Reducing dimensionality first helps.
- **Drift is not degradation.** A drift test shows that the data changed, not
  that the model got worse. Confirming a performance impact needs labeled
  operational data.

## In CheckMAITE

The [DataEval shift tutorial](../tool-usage/dataeval_shift_tutorial.ipynb)
shows the `DataevalShift` capability, which compares an operational dataset
with a reference dataset using [DataEval](https://github.com/aria-ml/dataeval):

- Both datasets are embedded with the same feature extractor, then reduced
  with PCA fitted on the reference set, to 256 dimensions by default.
- Drift is tested with MMD, CVM, and KS on those embeddings. Each reports
  whether drift was detected, with its distance, p-value, and threshold.
- OOD samples are flagged with a *k*-nearest-neighbor detector on the same
  embeddings, using cosine distance and a threshold at the 99th percentile of
  the reference set's own distances.

DataEval provides further drift and OOD methods that CheckMAITE does not
currently run, including tail-sensitive tests, domain classifiers,
reconstruction-based and uncertainty-based detectors, and label parity. The
DataEval page below describes them.

## Further reading

- [Distribution
  Shift](https://dataeval.readthedocs.io/en/latest/concepts/DistributionShift.html),
  the DataEval explanation that this page follows.
- Rabanser, S., Günnemann, S., & Lipton, Z. (2019). Failing loudly: An
  empirical study of methods for detecting dataset shift. *NeurIPS*.
  [arXiv:1810.11953](https://arxiv.org/abs/1810.11953)
- Gretton, A., et al. (2012). A kernel two-sample test. *Journal of Machine
  Learning Research*, 13, 723–773.
  [paper](https://jmlr.csail.mit.edu/papers/v13/gretton12a.html)
- Lipton, Z., Wang, Y. X., & Smola, A. (2018). Detecting and correcting for
  label shift with black box predictors. *ICML*.
  [arXiv:1802.03916](https://arxiv.org/abs/1802.03916)
- Kuan, J., & Mueller, J. (2022). Back to the basics: Revisiting
  out-of-distribution detection baselines.
  [arXiv:2207.03061](https://arxiv.org/abs/2207.03061)
