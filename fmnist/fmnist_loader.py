from tensorflow import keras

class Singleton(object):
        def __new__(cls, *args, **kwds):
            it = cls.__dict__.get("__it__")
            if it is not None:
                return it
            cls.__it__ = it = object.__new__(cls)
            it.init(*args, **kwds)
            return it

        def init(self, *args, **kwds):
            pass

class FMnistLoader(Singleton):
    def init(self):
        # load the MNIST dataset
        mnist = keras.datasets.fashion_mnist
        (_, _), (self.x_test, self.y_test) = mnist.load_data()

        print("++++++++++++++++++++ FMnistLoader created")

    def get_x_test(self):
        return self.x_test

    def get_y_test(self):
        return self.y_test
    
fmnist_loader = FMnistLoader()
