import numpy as np

def cut_by_size_in2_classes(a_waves, b_waves, wave_size):
    # Cut waves from raw a- and b_waves pulses of wave_size and label them
    nr_ya = len(ya)//wave_size
    nr_yb = len(yb)//wave_size

    nr_waves = nr_ya + nr_yb

    waves = np.zeros((nr_waves, wave_size))
    labels = np.zeros((nr_waves,), dtype=int)

    for i in range(nr_waves):
        start = i*wave_size
        stop = start + wave_size
        if i < nr_b:
            labels[i] = 1
            for j in range(start,stop):
                index = j - start
                waves[i][index] = b_waves[j]
        else:
            start = i*wave_size - nr_b*wave_size
            stop = start + wave_size
            labels[i] = 0
            for j in range(start,stop):
                index = j - start
                waves[i][index] = a_waves[j]

    return waves, labels


def cut_by_size_in3_classes(a_waves, b_waves, i_waves, wave_size):
    # Cut waves from raw a- and b_waves pulses of wave_size and label them
    nr_a = len(ya)//wave_size
    nr_b = len(yb)//wave_size
    nr_i = len(yi)//wave_size

    nr_waves = nr_ya + nr_yb + nr_yi

    waves = np.zeros((nr_waves, wave_size))
    labels = np.zeros((nr_waves,), dtype=int)

    for i in range(nr_waves):
        start = i*wave_size
        stop = start + wave_size

        if i < nr_a:
            labels[i] = 0
            for j in range(start,stop):
                index = j - start
                waves[i][index] = a_waves[j]

        elif (i > nr_a) and (i < nr_a + nr_b):
            start = wave_size*(i - nr_a)
            stop = start + wave_size
            labels[i] = 1
            for j in range(start,stop):
                index = j - start
                waves[i][index] = b_waves[j]

        else:
            start = wave_size*(i - nr_a - nr_b)
            stop = start + wave_size
            labels[i] = 2
            for j in range(start,stop):
                index = j - start
                waves[i][index] = i_waves[j]

    return waves, labels

def create_dataset(a_waves=None, b_waves=None, i_waves=None, wave_size=None, use_smote=False):
    # Sweeps through raw a_waves, b_waves, cuts pulses of size=wave_size
    # Returns waves vector of shape ((len(a_waves)+len(b_waves))/wave_size, wave_size)
    # and a labels vector of shape  ((len(a_waves)+len(b_waves))/wave_size, )

    nr_a = len(a_waves)//wave_size
    nr_b = len(b_waves)//wave_size

    if i_waves.any() == False:
        nr_i = 0
    else:
        nr_i = len(i_waves)//wave_size
        print('i_present')

    nr_waves = nr_a + nr_b + nr_i
    #print('Nr of waves: {}'.format(nr_waves))

    waves = np.zeros((nr_waves, wave_size))
    labels = np.zeros((nr_waves,), dtype=int)

    # Cutting function
    if i_waves.any() == False:
        waves, labels = cut_by_size_in2_classes(a_waves, b_waves, wave_size)
    else:
        waves, labels = cut_by_size_in3_classes(a_waves, b_waves, i_waves, wave_size)

    # Oversampling, IF classes are imbalanced
    if use_smote==True:
        from imblearn.over_sampling import SMOTE
        # Perform minority oversampling using SMOTE
        sm = SMOTE(sampling_strategy='auto',random_state=2) # state was 1
        waves, labels = sm.fit_resample(waves, labels)

    assert waves.shape[0] == nr_waves
    assert waves.shape[0] == labels.shape[0]

    return waves, labels


# Function used to plot 9 images in a 3x3 grid, and writing the true and
# predicted classes below each image
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

# Examples:
# Get the first images from the test-set.
images = data.x_test[0:9]

# Get the true classes for those images.
cls_true = data.y_test_cls[0:9]

# Plot the images and labels using our helper-function above.
plot_images(images=images, cls_true=cls_true)



def plot_conv_weights(weights, input_channel=0):
    # Assume weights are TensorFlow ops for 4-dim variables
    # e.g. weights_conv1 or weights_conv2.

    # Retrieve the values of the weight-variables from TensorFlow.
    # A feed-dict is not necessary because nothing is calculated.
    w = weights#session.run(weights)

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Number of filters used in the conv. layer.
    num_filters = w.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i<num_filters:
            # Get the weights for the i'th filter of the input channel.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = w[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def plot_conv_layer(layer, image):
    # Assume layer is a TensorFlow op that outputs a 4-dim tensor
    # which is the output of a convolutional layer,
    # e.g. layer_conv1 or layer_conv2.

    # Create a feed-dict containing just one image.
    # Note that we don't need to feed y_true because it is
    # not used in this calculation.
    feed_dict = {x: [image]}

    # Calculate and retrieve the output values of the layer
    # when inputting that image.
    values = session.run(layer, feed_dict=feed_dict)

    # Number of filters used in the conv. layer.
    num_filters = values.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot the output images of all the filters.
    for i, ax in enumerate(axes.flat):
        # Only plot the images for valid filters.
        if i<num_filters:
            # Get the output image of using the i'th filter.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = values[0, :, :, i]

            # Plot image.
            ax.imshow(img, interpolation='nearest', cmap='binary')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


class PeriodicPlotter:
  def __init__(self, sec, xlabel='', ylabel='', scale=None):
    from IPython import display as ipythondisplay
    import matplotlib.pyplot as plt
    import time

    self.xlabel = xlabel
    self.ylabel = ylabel
    self.sec = sec
    self.scale = scale

    self.tic = time.time()

  def plot(self, data):
    if time.time() - self.tic > self.sec:
      plt.cla()

      if self.scale is None:
        plt.plot(data)
      elif self.scale == 'semilogx':
        plt.semilogx(data)
      elif self.scale == 'semilogy':
        plt.semilogy(data)
      elif self.scale == 'loglog':
        plt.loglog(data)
      else:
        raise ValueError("unrecognized parameter scale {}".format(self.scale))

      plt.xlabel(self.xlabel); plt.ylabel(self.ylabel)
      ipythondisplay.clear_output(wait=True)
      ipythondisplay.display(plt.gcf())

self.tic = time.time()
