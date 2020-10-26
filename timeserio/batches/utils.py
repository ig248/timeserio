from abc import abstractmethod

from timeserio import externals


def ceiling_division(dividend, divisor):
    if not dividend:
        return 0
    return dividend // divisor + (1 if dividend % divisor else 0)


class Sequence(object):
    """Copy of https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence

    Defined here to avoid importing tf/keras unnecesseraly.
    """
    @abstractmethod
    def __getitem__(self, index):
        """Gets batch at position `index`.
        Arguments:
            index: position of the batch in the Sequence.
        Returns:
            A batch
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        """Number of batch in the Sequence.
        Returns:
            The number of batches in the Sequence.
        """
        raise NotImplementedError

    def on_epoch_end(self):
        """Method called at the end of every epoch."""
        pass

    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            yield item


def to_keras_gen(sequence):
    """Convert to keras.utils.Sequence"""
    keras_seq = externals.keras.utils.Sequence
    timeserio_seq = Sequence
    if isinstance(sequence, timeserio_seq):

        class KerasSequence(keras_seq):
            _sequence = sequence

            def __getitem__(inner_self, index):
                return inner_self._sequence[index]

            def __len__(inner_self):
                return len(inner_self._sequence)

            def on_epoch_end(inner_self):
                return inner_self._sequence.on_epoch_end()

            def __iter__(inner_self):
                return iter(inner_self._sequence)

        return KerasSequence()

    return sequence
