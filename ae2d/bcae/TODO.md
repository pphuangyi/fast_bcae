To compare fair and square with 3d BCAE:
1. add focal classification loss
1. add transform
1. regression loss only on signals
1. learning rate balancing
1. clf. threshold sliding bar experiment

Also for 3d BCAE
1. Check precision and recall

Experiments
1. SwiFT-type of compressor
1. Gaussian filter and its approx. inverse as tranpose convolution.
    The idea behind this experiement is as follows. There are two
    problems with the current datay:
    1. The zero-suppression cause the data to have "cliff"
        around 64 which is hard for neural networks to handle.
    1. The center location of blob is not necessarily signified
        with a bigger ADC value, but could be so with a Gaussian
        filter.
    With a Gaussian filter, the center location could be signified
    and the cliff could also be leveled to some extent.
    If the Gaussia filtered data would be easier to compress with
    the center location better preserved than with the current
    algorithms, we can already call it a success. And if we can
    also approximiatedly inverse the filter, we may have a better
    compresion algorithm on the raw data.
