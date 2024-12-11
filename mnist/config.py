import os
import uuid

# MNIST
POPSIZE          = int(os.getenv('DH_POPSIZE', '80'))
NGEN             = int(os.getenv('DH_NGEN', '500'))

RUNTIME          = int(os.getenv('DH_RUNTIME', '10'))
INTERVAL         = int(os.getenv('DH_INTERVAL', '900'))

# Mutation Hyperparameters
# range of the mutation
MUTLOWERBOUND    = float(os.getenv('DH_MUTLOWERBOUND', '0.01'))
MUTUPPERBOUND    = float(os.getenv('DH_MUTUPPERBOUND', '0.6'))

SELECTIONOP    = str(os.getenv('DH_SELECTIONOP', 'random')) # random or ranked or dynamic_ranked
SELECTIONPROB    = float(os.getenv('DH_SELECTIONPROB', '0.0'))
RANK_BIAS    = float(os.getenv('DH_RANK_BIAS', '1.5')) # value between 1 and 2
RANK_BASE    = str(os.getenv('DH_RANK_BASE', 'contribution_score')) # perf or density or contribution_score

# Dataset
EXPECTED_LABEL   = int(os.getenv('DH_EXPECTED_LABEL', '7'))

#------- NOT TUNING ----------

# mutation operator probability
MUTOPPROB        = float(os.getenv('DH_MUTOPPROB', '0.5'))
MUTOFPROB        = float(os.getenv('DH_MUTOFPROB', '0.5'))


IMG_SIZE         = int(os.getenv('DH_IMG_SIZE', '28'))
num_classes      = int(os.getenv('DH_NUM_CLASSES', '10'))

INITIALPOP       = os.getenv('DH_INITIALPOP', 'seeded')

# MODEL            = os.getenv('DH_MODEL', 'problem/mnist/models/model_mnist.h5')
MNIST_MODEL            = os.getenv('DH_MODEL', 'mnist/models/model_mnist.h5')

ORIGINAL_SEEDS   = os.getenv('DH_ORIGINAL_SEEDS', 'bootstraps_five')

BITMAP_THRESHOLD = float(os.getenv('DH_BITMAP_THRESHOLD', '0.5'))

DISTANCE_SEED         = float(os.getenv('DH_DISTANCE_SEED', '5.0'))
DISTANCE         = float(os.getenv('DH_DISTANCE', '5.0'))

FEATURES             = os.getenv('FEATURES', ["Bitmaps", "Moves"])
NUM_CELLS           = int(os.getenv("NUM_CELLS", '25'))
# FEATURES             = os.getenv('FEATURES', ["Orientation","Bitmaps"])
# FEATURES             = os.getenv('FEATURES', ["Orientation","Moves"])

TSHD_TYPE             = os.getenv('TSHD_TYPE', '1') # 1: threshold on vectorized-rasterized seed, use DISTANCE = 2

RUN = 1

FEATURES = ["Orientation", "Moves"]

NAME = None

HE_HASH = str(uuid.uuid4().hex)

FMNIST_MODEL_PATH = 'fmnist/models/Model1_fmnist.h5'
