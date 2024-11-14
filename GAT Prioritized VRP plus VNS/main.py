import tensorflow as tf
from time import gmtime, strftime
import copy

from GAT_RL_Model import *
from train import train_model

from utils import create_data_on_disk, get_cur_time

if __name__ == "__main__":

    # Parameters of model
    SAMPLES = 512 # 128*10000
    BATCH = 8
    START_EPOCH = 0
    END_EPOCH = 50
    update_target = 2
    FROM_CHECKPOINT = False
    embedding_dim = 32
    LEARNING_RATE = 0.0001
    ROLLOUT_SAMPLES = 10 #10000
    NUMBER_OF_WP_EPOCHS = 1
    GRAD_NORM_CLIPPING = 1.0
    BATCH_VERBOSE = 5 #1000
    VAL_BATCH_SIZE = 8 #1000
    VALIDATE_SET_SIZE = 64 #10000
    SEED = 1234
    GRAPH_SIZE = 20
    N_Deopts = 3
    N_Agents = N_Deopts
    FILENAME = 'VRP_{}_{}'.format(GRAPH_SIZE, strftime("%Y-%m-%d", gmtime()))

    ## Time Window Penalty Coefficients
    ALPHA = 0.1
    BETA = 0.2

    # Initialize model
    model_tf = GAT_RL(embedding_dim, n_agents=N_Agents, n_depots=N_Deopts, n_encode_layers=3, num_heads=8, clip=10., alpha=ALPHA, beta=BETA)
    print(get_cur_time(), 'model initialized')
    
    #baseline = copy.deepcopy(model_tf)
    baseline = GAT_RL(embedding_dim, n_agents=N_Agents, n_depots=N_Deopts, n_encode_layers=3, num_heads=8, clip=10., alpha=ALPHA, beta=BETA)
    baseline.set_weights(model_tf.get_weights())
    print(get_cur_time(), 'baseline initialized')


    # Create and save validation dataset
    validation_dataset = create_data_on_disk(GRAPH_SIZE,
                                            VALIDATE_SET_SIZE,
                                            is_save=True,
                                            filename=FILENAME,
                                            is_return=True,
                                            num_depots= N_Deopts,
                                            seed = SEED)
    print(get_cur_time(), 'validation dataset created and saved on the disk')

    # Initialize optimizer
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

    train_model(optimizer,
            model_tf,
            baseline,
            validation_dataset,
            samples = SAMPLES,
            batch = BATCH,
            val_batch_size = VAL_BATCH_SIZE,
            start_epoch = START_EPOCH,
            end_epoch = END_EPOCH,
            from_checkpoint = FROM_CHECKPOINT,
            grad_norm_clipping = GRAD_NORM_CLIPPING,
            batch_verbose = BATCH_VERBOSE,
            graph_size = GRAPH_SIZE,
            n_depots= N_Deopts,
            filename = FILENAME,
            update_step = update_target
            )
    
