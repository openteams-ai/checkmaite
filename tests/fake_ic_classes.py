from typing import Any, Iterable, Sequence, Tuple
from torch import Tensor
from maite._internals.protocols import ArrayLike
import maite.protocols.image_classification as ic

DEFAULT_IC_MODEL_PREDICTIONS: Sequence[Tensor] = [
    Tensor([-0.2832, -0.4654, -0.5866,  3.3975, -0.4545,  0.0570, -0.3015, -0.4808, -0.1757, -0.5573]),
    Tensor([-0.1634, -0.1646, -0.6448, -0.3842, -0.5051, -0.4712, -0.4654, -0.6231, 3.4097, -0.6745]),
    Tensor([-0.0775,  0.1461, -0.5379, -0.4177, -0.5669, -0.6365, -0.4918, -0.6450, 3.2838, -0.5834]),
    Tensor([ 2.2617, -0.7592,  1.2758, -0.3185, -0.1752, -0.6627, -0.4864, -0.5339, -0.1966, -0.5145]),
    Tensor([-0.3537, -0.2256, -0.1936, -0.3826, -0.3499, -0.5073,  3.3957, -0.5077, -0.2492, -0.5025]),
    Tensor([-0.7112, -0.6057, -0.5041,  0.1275, -0.3270,  0.1434,  3.3354, -0.4935, -0.4712, -0.6506]),
    Tensor([-0.3078,  3.4402, -0.2747, -0.6082, -0.5243, -0.6291, -0.3965, -0.3440, -0.2732,  0.2556]),
    Tensor([-0.2937, -0.4419, -0.4337, -0.2989, -0.3909, -0.3279,  3.3799, -0.5901, -0.2065, -0.4537]),
    Tensor([-0.5285, -0.4450, -0.6761,  3.3978, -0.3468,  0.0582, -0.2942, -0.3642, -0.1514, -0.5457]),
    Tensor([-0.5058,  3.3764, -0.3019, -0.5000, -0.5454, -0.6148, -0.3679, -0.4290, -0.2001,  0.3801]),
    Tensor([ 2.9787, -0.6276,  0.8979, -0.4128, -0.6522, -0.2636, -0.6498, -0.5331, -0.3570, -0.3990]),
    Tensor([-0.3237,  0.0699, -0.4151, -0.4021, -0.1599, -0.3998, -0.3533, -0.5239, -0.4047,  3.3992]),
    Tensor([-0.5112, -0.7362, -0.4982,  0.1223, -0.3884,  3.1434, -0.4271, -0.3184, -0.2962, -0.4724]),
    Tensor([-0.3638, -0.2724, -0.2569, -0.3294, -0.0768, -0.2099, -0.3563,  3.5357, -0.3638, -0.4512]), # 7 predicted, 8 is correct target
    Tensor([-0.3090, -0.0466, -0.3871, -0.2501, -0.1755, -0.4787, -0.3690, -0.5002, -0.3567,  3.3851]),
    Tensor([ 0.0523, -0.5057, -0.5043, -0.4131, -0.4282, -0.5697, -0.4123, -0.7042, 3.3588, -0.7597]),
    Tensor([-0.4867, -0.6812, -0.2931,  0.2918, -0.5057,  3.1666, -0.4555, -0.3837, -0.3678, -0.6348]),
    Tensor([-0.2538, -0.1592, -0.3918, -0.3946, -0.1470, -0.3973, -0.3574,  3.4823, -0.2857, -0.2486]), # 7 predicted, 8 is correct target
    Tensor([-0.2821, -0.3694, -0.5154, -0.2898, -0.3963, -0.5335, -0.4082, -0.7029, 3.4225, -0.6853]),
    Tensor([-0.4688, -0.3213, -0.0638, -0.3976, -0.2782, -0.3688,  3.4971, -0.5379, -0.4190, -0.5718])
 ]

DEFAULT_IC_DATASET_TARGETS: Sequence[Tensor] = [
    Tensor([0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]),
    Tensor([0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]),
    Tensor([0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]),
    Tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
    Tensor([0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]),
    Tensor([0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]),
    Tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]),
    Tensor([0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]),
    Tensor([0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]),
    Tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]),
    Tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
    Tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]),
    Tensor([0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]),
    Tensor([0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]), # altered 7 to 8
    Tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]),
    Tensor([0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]),
    Tensor([0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]),
    Tensor([0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]), # altered 7 to 8
    Tensor([0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]),
    Tensor([0., 0., 0., 0., 0., 0., 1., 0., 0., 0.])
]


class RestartingIterator(Iterable):
    """This custom iterator will restart (not raise StopIteration) after reaching last item of the iterable."""

    def __init__(self, iterable):
        self.iterable = iterable
        self.iterator = iter(iterable)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.iterable)
            return next(self.iterator)


class FakeICModel(ic.Model):
    """This MAITE-compliant fake (stub) IC model can be instantiated with canned responses to its __call__ function for testing.
    A restarting iterator class is used so that a fake model can handle an indefinite number of calls of arbitrary batch sizes.

    IMPORTANT - if the default stub values do not meet your testing requirements, consult the RI Team (in GitLab)
    before implementing a custom instance of this class.  As much as possible, we would like to have as many test cases as possible use
    the default attributes/fixture, so expanding the defaults to cover additional scenarios is likely preferable to creating different fake
    data for different scenarios.

    Attributes:
        prediction_data (Sequence[ArrayLike]): A Sequence of responses to provide, sequentially and repeating, as the model is called.
        The __call__ method will batch these responses into a return batch of the correct size.

            The default value should suffice for most (if not all) test cases.  It is a list of 20 predictions of a 10-class multiclass classification
            task.  The data has been configured so that 18 of the 20 predictions are correct when run against the default fake IC dataset.  If a different
            response pattern is needed, the prediction_data argument can be used.
            
            For example:
            
                prediction_data = [ torch.Tensor([0,1,0,0]) ] will return a prediction representing class 2 of 4 for every target passed to the model.
            
                prediction_data = [ torch.Tensor([1,0,0]), torch.Tensor([0,1,0]), torch.Tensor([0,0,1])] will sequentially rotate between picking classes 
                with index 0, 1, 2 of three possible classes.

            A fake model can also be pre-loaded with the desired responses to a known dataset order.  (This is how the default value is implemented
            vis-a-vis the default values for the FakeICDataset.)
    
    NOTE: real implementations of MAITE Models may include an index2label property (mapping class '3' to 'chair', e.g.).  TBD if that's necessary for our fake suite.
    """
    def __init__(self, prediction_data: Sequence[ArrayLike] = DEFAULT_IC_MODEL_PREDICTIONS):
        self.prediction_iter: RestartingIterator = RestartingIterator(prediction_data)

    def __call__(self, input: Sequence[ArrayLike]) -> Sequence[ArrayLike]:
        return [ next(self.prediction_iter) for _ in range(len(input))]
    

class FakeICDataset(ic.Dataset):
    """
    This MAITE-compliant fake IC dataset can be instantiated with any MAITE-compliant representations of (1)images, (2) corresponding class ground truths, and
    (3) corresponding metadata for an image classification dataset. 

    IMPORTANT - if the default fake values do not meet your testing requirements, consult the RI Team (in GitLab) before implementing a custom instance of
    this class.  As much as possible, we would like to have as many test cases as possible use the default attributes/fixture, so expanding the defaults to
    cover additional scenarios is likely preferable to creating different fake data for different scenarios.
    
    The user can make pre-seed their own dataset with any MAITE-compliant data.

    Attributes:
        images (Sequence[ArrayLike]): A Sequence (e.g. list) of images represented in an ArrayLike format (e.g. a torch Tensor)
        targets (Sequence[ArrayLike]): A Sequence of ground truth targets, typically as a 1D array of one-hot encodings, shape (C,).
        metadata (Sequence[dict[str,Any]]): A dictionary of metadata associated with each image.
    """
    def __init__(self, 
                 images: Sequence[ArrayLike] = list([ Tensor(1,1,1) for _ in range(20) ]),
                 targets: Sequence[ArrayLike] = DEFAULT_IC_DATASET_TARGETS,
                 metadata: Sequence[dict[str, Any]] = [ {"id": i} for i in range(20)]
                ):
        self.images: Sequence[ArrayLike] = images
        self.targets: Sequence[ArrayLike] = targets
        self.metadata: Sequence[dict[str, Any]] = metadata

    def __getitem__(self, ind:int) -> Tuple[ArrayLike, ArrayLike, dict[str,Any]]:
      return (self.images[ind],self.targets[ind],self.metadata[ind])

    def __len__(self) -> int:
      return len(self.images)

# Next task... fake metric
