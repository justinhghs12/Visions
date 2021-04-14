import numpy as np

class Conv2D(object):
    """2D convolutional layer.

    Arguments:
        in_channel: number of input channel
        out_channel: number of output channel
        kernel_size (tuple): the shape of the kernel. It is a tuple = (
            kernel_height, kernel_width).
        strides (int or tuple): the strides of the convolution operation.
            padding (int or tuple): number of zero paddings.
        weights_init (obj):  an object instantiated using any initializer class
                in the "initializer" module.
        bias_init (obj):  an object instantiated using any initializer class
                in the "initializer" module.
        name (str): the name of the layer.

    Attributes:
        W (np.array): the weights of the layer. A 4D array of shape (
            out_channels, in_channels, kernel_height, kernel_width).
        b (np.array): the biases of the layer. A 1D array of shape (
            out_channels).
        kernel_size (tuple): the shape of the filter. It is a tuple = (
            out_channels, in_channels, kernel_height, kernel_width).
        strides (tuple): the strides of the convolution operation. A tuple = (
            height_stride, width_stride).
        padding (tuple): the number of zero paddings along the height and
            width. A tuple = (height_padding, width_padding).
        name (str): the name of the layer.

    """
    def __init__(
            self, kernel_size, stride, padding, name="Conv2D"):
        self.W = np.random.randn(*kernel_size)
        self.b = np.random.randn(kernel_size[0], 1)
        self.kernel_size = kernel_size
        self.stride = (stride, stride) if type(stride) == int else stride
        self.padding = (padding, padding) if type(padding) == int else padding
        self.name = name

    def __repr__(self):
        return "{}({}, {}, {})".format(
            self.name, self.kernel_size, self.stride, self.padding
        )
    
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """Compute the layer output.

        Args:
            x (np.array): the input of the layer. A 3D array of shape (
                in_channels, in_heights, in_weights).

        Returns:
            The output of the layer. A 3D array of shape (out_channels,
                out_heights, out_weights).

        """                
        p, s = self.padding, self.stride
        x_padded = np.pad(
            x, ((0, 0), (p[0], p[0]), (p[1], p[1])), mode='constant'
        )
        
        # check dimensions
        assert (x.shape[1] - self.W.shape[2] + 2 * p[0]) / s[0] + 1 > 0, \
                'Height doesn\'t work'
        assert (x.shape[2] - self.W.shape[3] + 2 * p[1]) / s[1] + 1 > 0, \
                'Width doesn\'t work'
        # TODO: Put your code below
        ret_list = []
        count = 0
        for out_channel in self.W:
        
            convolved_list = []
            for channel in range(len(x)):
                
                #conv_per_channel = np.zeros((x.shape[1]//s[0] + 1, x.shape[2]//s[1] + 1))
                conv_per_channel = np.zeros((((x.shape[1] + 2 * p[0])//s[0] - 1), (x.shape[2] + 2 * p[1])//s[1] - 1))
                current_kernel = out_channel[channel]                
                current_window = np.zeros((self.kernel_size[2], self.kernel_size[3]))
                
                count_y = 0
                for row in range(0, len(x_padded[channel]) - self.kernel_size[2] + 1, s[0]): 
                    count_x = 0
                    for weight in range(0, len(x_padded[channel][row]) - self.kernel_size[3] + 1, s[1]):
                        
                        for u in range(self.kernel_size[2]):
                            for v in range(self.kernel_size[3]):
                                current_window[u][v] = x_padded[channel][row + u][weight + v]
                                                
                        conv_per_channel[count_y][count_x] = np.multiply(current_window, current_kernel).sum()
                        
                        count_x += 1
                    count_y += 1
                    
                convolved_list.append(conv_per_channel)
            
            total = np.array(convolved_list[0])
            
            for slide in convolved_list[1:]:
                total = np.add(total, np.array(slide))
            
            add_outcomes = np.array(total) + self.b[count]
                        
            count += 1
            
            ret_list.append(add_outcomes)  

        return np.array(ret_list)

class MaxPool2D(object):
    def __init__(self, kernel_size, stride, padding, name="MaxPool2D"):
        self.kernel_size = kernel_size
        self.stride = (stride, stride) if type(stride) == int else stride
        self.padding = (padding, padding) if type(padding) == int else padding
        self.name = name

    def __repr__(self):
        return "{}({}, {}, {})".format(
        self.name, self.kernel_size, self.stride, self.padding
    )

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """Compute the layer output.
        
        Arguments:
            x {[np.array]} -- the input of the layer. A 3D array of shape (
                              in_channels, in_heights, in_weights).
        Returns:
            The output of the layer. A 3D array of shape (out_channels,
                out_heights, out_weights).
        """
        p, s = self.padding, self.stride
        x_padded = np.pad(
            x, ((0, 0), (p[0], p[0]), (p[1], p[1])), mode='constant'
        )

        # check dimensions
        assert (x.shape[1] - self.kernel_size[0] + 2 * p[0]) / s[0] + 1 > 0, \
                'Height doesn\'t work'
        assert (x.shape[2] - self.kernel_size[1] + 2 * p[1]) / s[1] + 1 > 0, \
                'Width doesn\'t work'

        # TODO: Put your code below
        
        ret_list = []
        
        for channel in range(x.shape[0]):
            
            max_per_channel = np.zeros((x.shape[1]//s[0], x.shape[2]//s[1]))
            count_y = 0
            for row in range(0, len(x_padded[channel]) - self.kernel_size[0] + 1, s[0]):
                count_x = 0
                for weight in range(0, len(x_padded[channel][row]) - self.kernel_size[1] + 1, s[1]):
                    
                    current_max = -999_999_999
                    
                    for u in range(self.kernel_size[0]):
                        for v in range(self.kernel_size[1]):   
                            
                            if current_max < x_padded[channel][row + u][weight + v]:
                                current_max = x_padded[channel][row + u][weight + v]
                            
                    max_per_channel[count_y][count_x] = current_max
                    
                    count_x += 1
                count_y += 1
                
            ret_list.append(max_per_channel)
                    
        return np.array(ret_list)

class AvgPool2D(object):
    def __init__(self, kernel_size, stride, padding, name="MaxPool2D"):
        self.kernel_size = kernel_size
        self.stride = (stride, stride) if type(stride) == int else stride
        self.padding = (padding, padding) if type(padding) == int else padding
        self.name = name

    def __repr__(self):
        return "{}({}, {}, {})".format(
        self.name, self.kernel_size, self.stride, self.padding
    )

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """Compute the layer output.
        
        Arguments:
            x {[np.array]} -- the input of the layer. A 3D array of shape (
                              in_channels, in_heights, in_weights).
        Returns:
            The output of the layer. A 3D array of shape (out_channels,
                out_heights, out_weights).
        """
        
        p, s = self.padding, self.stride
        x_padded = np.pad(
            x, ((0, 0), (p[0], p[0]), (p[1], p[1])), mode='constant'
        )

        # check dimensions
        assert (x.shape[1] - self.kernel_size[0] + 2 * p[0]) / s[0] + 1 > 0, \
                'Height doesn\'t work'
        assert (x.shape[2] - self.kernel_size[1] + 2 * p[1]) / s[1] + 1 > 0, \
                'Width doesn\'t work'
        
        # TODO: Put your code below
        ret_list = []
        for channel in range(x.shape[0]):
            avg_per_channel = np.zeros((x.shape[1]//s[0], x.shape[2]//s[1]))
            
            count_y = 0
            for row in range(0, len(x_padded[channel]) - self.kernel_size[0] + 1, s[0]):
                count_x = 0
                for weight in range(0, len(x_padded[channel][row]) - self.kernel_size[1] + 1, s[1]):
                    
                    avg = 0
                    
                    for u in range(self.kernel_size[0]):
                        for v in range(self.kernel_size[1]):                    
                            avg += (x_padded[channel][row + u][weight + v]/(self.kernel_size[0] * self.kernel_size[1]))
                            #print(avg)      
                    avg_per_channel[count_y][count_x] = avg
                    
                    count_x += 1
                count_y += 1
                
            ret_list.append(avg_per_channel)
                    
        return np.array(ret_list)
      