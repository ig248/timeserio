import abc

from timeserio.batches import utils as batch_utils


class BatchGenerator(batch_utils.Sequence, abc.ABC):
    pass
