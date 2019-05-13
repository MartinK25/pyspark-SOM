import numpy as np
from pyspark import Row
from scipy.special import rel_entr

def jsd(p_distb, q_distb):#, base=None):
    """Jensen Shannon Distance

    Args:
        p_distb (array): first vector (discrete distribution)
        q_distb (array): second vector

    Returns:
        Jensen Shannon Distance

    """
    p = np.asarray(p_distb)#makes almost no difference to leave this out
    q = np.asarray(q_distb)
    m = (p + q) / 2.0
    left = rel_entr(p, m)
    right = rel_entr(q, m)
    js = np.sum(left, axis=0) + np.sum(right, axis=0)
    return np.sqrt(js / 2.0)

def fast_norm(x):
    """Calculates euclidean norm for input vector x"""
    return np.sqrt(np.dot(x, x.T))

def gaussian(w, h, bmu, sigma):
    """Neighborhood function: e^(-("dist to bmu")^2 / 2*pi*sigma^2)

    Args:
        w (int): width of SOM
        h (int): height of SOM
        c (array): index of bmu in SOM(h, w)
        sigma (float): sigma

    Returns:
        Neighborhood influence values for all nodes in SOM.

    """
    d = 2*np.pi*sigma*sigma
    neigx = np.arange(w) #returns [0, .. , w-1]
    neigy = np.arange(h)
    ax = np.exp(-np.power(neigx-bmu[0], 2)/d)
    ay = np.exp(-np.power(neigy-bmu[1], 2)/d)
    return np.outer(ax, ay)

def bmu(x, codebook, w, h):
    """Finds the best matching unit in the SOM. Unit with the least distance to input x.

    Args:
        x (array): input vector.
        codebook (np array): SOM net
        w (int): width of SOM
        h (int): height of SOM

    Returns:
        Returns the index of the bmu.

    """

    activation_map = np.zeros((h, w))

    eucl=False # Use Euclidean distance (or other distance metrics)
    if(eucl):
        diff = np.subtract(x, codebook)
        it = np.nditer(activation_map, flags=['multi_index']) #flags=['multi_index','buffered']) no improvement
        while not it.finished:
            activation_map[it.multi_index] = fast_norm(diff[it.multi_index])
            it.iternext()
    else:
        distm = jsd
        it = np.nditer(activation_map, flags=['multi_index']) #flags=['multi_index','buffered']) no improvement
        while not it.finished:
            activation_map[it.multi_index] = distm(x,codebook[it.multi_index])
            it.iternext()

    return np.unravel_index(activation_map.argmin(), activation_map.shape)


class SOM:
    """Class to create a new SOM

    This is a SOM batch implementation. Which means in every epoch all observations are used for training
    on the same net. The observations are distributed to different workers on spark. Each worker computes
    the update for its subset of training data. Then the updates of all workers are collected in the driver
    and the net is adjusted with the updates.

    Note:
        Distance function: Euclidean distance

    Args:
        w (int): width of SOM
        h (int): height of SOM
        num (int): number of dimensions/features
        sigma (float): Sigma (for size of influence of neighborhood fct)
        lr (float): Learning rate

    """
    def __init__(self, w, h, num, sigma=0.2, lr=0.1):
        self.w = w
        self.h = h
        self.num = num
        self.sigma = sigma
        self.lr = lr
        self._decay_func = lambda x, t, max_iter: x/(1+float(t)/max_iter)

    def initialize(self, data, fromDs):
        """Initializes the net (self.codebook) either with randomly created data or randomly with data from the input

        Args:
            data (RDD): Input dataset.
            fromDs (boolean): Should the random sample come from the input dataset.
        """
        if(fromDs):
            self.codebook = np.array([[np.zeros((self.num)) for i in range(self.w)] for j in range(self.h)])
            for j in range(self.h):
                for i in range(self.w):
                    self.codebook[j][i] = np.array(data.takeSample(True,1,42))
        else:
            np.random.seed(42)
            self.codebook = np.random.random((self.h, self.w, self.num))


    def quantization_error(self, data):
        """Calculates the quantization error with the current codebook:
        sum(euclidean norm(diff each element to its bmu)) / observations in dataset

        Args:
            data (RDD): dataset to calculated the error with.
        """
        error = 0
        for x in data:
            error += fast_norm(x-self.codebook[bmu(x, self.codebook, self.w, self.h)])
        return error/len(data)

    def train(self, data, epochs, partitions=12, calcQ = True):
        """Trains the SOM in batch mode

        Args:
            data (RDD): dataset.
            epochs (int): number of epochs
            partitions (int): deprecated (number of partitions for repartitioning the data after each iteration - useless overhead?)
            calcQ (boolean): Calculate the quantification error in each round or not.
        """

        dataRDD = data.cache() #cache the data (keep in ram)

        for t in range(epochs):

            start_ep = timeit.default_timer()

            sigma = self._decay_func(self.sigma, t, epochs)
            lr = self._decay_func(self.lr, t, epochs)
            codebookBC = spark.sparkContext.broadcast(self.codebook)

            if(calcQ):
                print("Epoch: {0:}, sigma: {1:.3f}, lr: {2:.3f}, error: {3:.3f}, Time needed to broadcast new net: {4:.2f} s".format(t, sigma, lr, self.quantization_error(dataRDD.collect()), timeit.default_timer() - start_ep ))
            else:
                print("iter: {0:}, sigma: {1:.3f}, lr: {2:.3f}, Time needed to broadcast new net: {3:.2f}".format(t, sigma, lr, 0, timeit.default_timer() - start_ep))#self.quantization_error(dataRDD.collect())))

            online=False
            if(online==True):
                def train_partition_wrapper(w, h, num_features): #wrapper function to prevent serialization of whole class

                    def train_partition(partition_data):
                        localCodebook = codebookBC.value #get broadcasted codebook

                        for elem in partition_data:
                            (w_h, w_w) = bmu(elem, localCodebook, w, h) #find bmu
                            g = gaussian(w, h, (w_h, w_w), sigma) * lr #neighborhood function * learning rate
                            it = np.nditer(g, flags=['multi_index'])
                            while not it.finished:
                                #update every node of local codebook by influcene * distance
                                localCodebook[it.multi_index] += g[it.multi_index]*(elem - localCodebook[it.multi_index])
                                it.iternext()
                        #return [localCodebook]
                        yield Row(local=localCodebook)
                    return train_partition


                #apply update function to all partitions
                resultCodebookRDD = dataRDD.mapPartitions(train_partition_wrapper(self.w, self.h, self.num))

                start_it = timeit.default_timer()

                #working version:
                #sum the updated local codebooks from each partitions and divide with number of partitions to create new net/codebook
                epoch_sum = np.array(np.zeros([self.w, self.h, self.num]))
                for row in resultCodebookRDD.collect():
                    epoch_sum += row['local']
                self.codebook = epoch_sum / float(partitions)

            else:

                def train_partition_wrapper(w, h, num_features): #wrapper function to prevent serialization of whole class

                    def train_partition(partition_data):
                        localCodebook = codebookBC.value #get broadcasted codebook
                        localNominator = np.array(np.zeros([w, h, num_features]))
                        localDenominator = np.array(np.zeros([w, h, 1]))
                        localZero = np.array(np.zeros([w, h, num_features]))

                        for elem in partition_data:
                            (w_h, w_w) = bmu(elem, localCodebook, w, h) #find bmu
                            g = gaussian(w, h, (w_h, w_w), sigma) * lr #neighborhood function * learning rate
                            it = np.nditer(g, flags=['multi_index'])
                            while not it.finished:
                                #update every node of local codebook by influcene * distance
                                localDenominator[it.multi_index] += g[it.multi_index]
                                localNominator[it.multi_index] += g[it.multi_index]*(elem - localZero[it.multi_index])
                                it.iternext()
                        #return [localCodebook]
                        yield Row(localDen=localDenominator,localNom=localNominator)
                    return train_partition


                #apply update function to all partitions
                resultCodebookRDD = dataRDD.mapPartitions(train_partition_wrapper(self.w, self.h, self.num))

                start_it = timeit.default_timer()

                #sum the updated local codebooks from each partitions and divide with number of partitions to create new net/codebook
                epoch_sum_den = np.array(np.zeros([self.w, self.h, 1]))
                epoch_sum_nom = np.array(np.zeros([self.w, self.h, self.num]))
                for row in resultCodebookRDD.collect():
                    epoch_sum_den += row['localDen']
                    epoch_sum_nom += row['localNom']
                self.codebook = epoch_sum_nom / epoch_sum_den



            print("Calculation of batch & update took {0:.3f} sek".format(timeit.default_timer() - start_it))
        print("Finished.")

## example for application:



#df_train = Spark RDD
#partitions = 20 #Number of Partitions -> should be roughly cores times 3 (parallelization)
#df_train.repartition(partitions)

# Create SOM object
#som = SOM(9, 9, 137, sigma=0.5, lr=0.3) #(nodes x, nodes y, dimensions, neighborhood size, learning rate)

#epochs = 500 #Define number of epochs

# Initialize SOM (define initial weights)
#som.initialize(df_train, fromDs = False) #fromDs = True -> initialize random from input data (takes a long time)

# Train the SOM
#som.train(df_train, epochs, partitions, calcQ = False) #calcQ -> calculate Quantization error after each epoch (based on Euclidean distance; takes time!)

# Print Codebook
#print(pd.DataFrame(som.codebook))