from tqdm import tqdm
import pandas as pd
import tensorflow as tf
import copy
from GAT_RL_Model import *

from utils import generate_data_onfly, get_results, get_cur_time, create_vectors
from time import gmtime, strftime


def rollout(model, dataset, batch_size = 1000, disable_tqdm = False):
    # Evaluate model in greedy mode
    costs_list = []

    for batch in tqdm(dataset.batch(batch_size), disable=disable_tqdm, desc="Rollout execution"):
        depots, customers, demands, time_windows, service_times = batch
        c, d = create_vectors(customers, depots, demands, time_windows, service_times)
        cost, _ = model(c, demands, time_windows, service_times, d, customers, depots)
        costs_list.append(cost)

    return tf.concat(costs_list, axis=0)


def validate(dataset, model, batch_size=1000):
    """Validates model on given dataset in greedy mode
    """
    val_costs = rollout(model, dataset, batch_size=batch_size)
    mean_cost = tf.reduce_mean(val_costs)
    print(f"Validation score: {np.round(mean_cost, 4)}")
    return mean_cost


def train_model(optimizer,
                model_tf,
                baseline,
                validation_dataset,
                samples = 1280000,
                batch = 128,
                val_batch_size = 1000,
                start_epoch = 0,
                end_epoch = 5,
                from_checkpoint = False,
                grad_norm_clipping = 1.0,
                batch_verbose = 1000,
                graph_size = 20,
                n_depots = 2,
                filename = None,
                update_step = 10
                ):
    
    print("Training Phase")

    if filename is None:
        filename = 'VRP_{}_{}'.format(graph_size, strftime("%Y-%m-%d", gmtime()))
    
    def rein_loss(model, inputs, baseline, num_batch):
        # Evaluate model, get costs and log probabilities
        depots, customers, demands, time_windows, service_times = inputs
        custs_vectors, depots_vectors = create_vectors(customers, depots, demands, time_windows, service_times)

        cost, log_likelihood = model(custs_vectors, demands, time_windows, service_times, depots_vectors, customers, depots)

        bl_val, _ = baseline(custs_vectors, demands, time_windows, service_times, depots_vectors, customers, depots)

        cost = tf.reduce_mean(cost)
        bl_val = tf.reduce_mean(bl_val)

        # Calculate loss
        baseline_loss = tf.reduce_mean((cost - bl_val) * log_likelihood)
        # print(f"\n\n>> cost: {tf.reduce_mean(cost)}, log_likelihood:{log_likelihood}, reinforce_loss:{reinforce_loss}\n")

        return baseline_loss, tf.reduce_mean(cost)
    

    def grad(model, inputs, baseline, num_batch):
        """Calculate gradients
        """
        with tf.GradientTape() as tape:
            loss, cost = rein_loss(model, inputs, baseline, num_batch)
        return loss, cost, tape.gradient(loss, model.trainable_variables)
    

    # For plotting
    train_loss_results = []
    train_cost_results = []
    val_cost_avg = []

    # Training loop
    for epoch in range(start_epoch, end_epoch):
        # Create dataset on current epoch
        data = generate_data_onfly(num_samples=samples, graph_size=graph_size, num_depots=n_depots)

        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_cost_avg = tf.keras.metrics.Mean()

        print("Current decode type: {}".format(model_tf.decode_type))

        for num_batch, x_batch in tqdm(enumerate(data.batch(batch)), 
                                       desc="batch calculation at epoch {}".format(epoch)):
            
            # Optimize the model
            loss_value, cost_val, grads = grad(model_tf, x_batch, baseline, num_batch)

            # Clip gradients by grad_norm_clipping
            init_global_norm = tf.linalg.global_norm(grads)
            grads, _ = tf.clip_by_global_norm(grads, grad_norm_clipping)
            global_norm = tf.linalg.global_norm(grads)

            if num_batch%batch_verbose == 0:
                print("grad_global_norm = {}, clipped_norm = {}".format(init_global_norm.numpy(), global_norm.numpy()))

            optimizer.apply_gradients(zip(grads, model_tf.trainable_variables))

            # Track progress
            epoch_loss_avg.update_state(loss_value)
            epoch_cost_avg.update_state(cost_val)

            if num_batch%batch_verbose == 0:
                print("Epoch {} (batch = {}): Loss: {}: Cost: {}".format(epoch, num_batch, epoch_loss_avg.result(), epoch_cost_avg.result()))
        
        # Update baseline if the candidate model is good enough. In this case also create new baseline dataset
        if epoch % update_step == 0:
            baseline.set_weights(model_tf.get_weights()) 

        # Save model weights
        # model_tf.save_weights('model_checkpoint_epoch_{}_{}.h5'.format(epoch, filename), save_format='h5')
        model_tf.save_weights('model_checkpoint_epoch_{}_{}.weights.h5'.format(epoch, filename))

        # Validate current model
        val_cost = validate(validation_dataset, model_tf, val_batch_size)
        val_cost_avg.append(val_cost)

        train_loss_results.append(epoch_loss_avg.result())
        train_cost_results.append(epoch_cost_avg.result())

        pd.DataFrame(data={'epochs': list(range(start_epoch, epoch+1)),
                           'train_loss': [x.numpy() for x in train_loss_results],
                           'train_cost': [x.numpy() for x in train_cost_results],
                           'val_cost': [x.numpy() for x in val_cost_avg]
                           }).to_csv('backup_results_' + filename + '.csv', index=False)

        print(get_cur_time(), "Epoch {}: Loss: {}: Cost: {}".format(epoch, epoch_loss_avg.result(), epoch_cost_avg.result()))

    # Make plots and save results
    filename_for_results = filename + '_start={}, end={}'.format(start_epoch, end_epoch)
    get_results([x.numpy() for x in train_loss_results],
                [x.numpy() for x in train_cost_results],
                [x.numpy() for x in val_cost_avg],
                save_results=True,
                filename=filename_for_results,
                plots=True)
   
        

