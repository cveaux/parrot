import os
from algorithms import Adasecant
from blocks import initialization
from blocks.algorithms import (
    Adam, CompositeRule, GradientDescent, StepClipping)
from blocks.extensions import (Printing, Timing)
from blocks.extensions.monitoring import (
    DataStreamMonitoring, TrainingDataMonitoring)
from blocks.extensions.predicates import OnLogRecord
from blocks.extensions.saveload import Checkpoint, Load
from blocks.extensions.training import TrackTheBest
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from blocks.model import Model
import cPickle
from extensions import LearningRateSchedule, Plot, TimedFinish
from datasets import parrot_stream
from model import Parrot
from utils import train_parse

args = train_parse()

if args.algorithm == "adasecant":
    args.lr_schedule = False

exp_name = args.experiment_name
save_dir = args.save_dir


if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if not os.path.exists(os.path.join(save_dir, "config")):
    os.makedirs(os.path.join(save_dir, "config"))

if not os.path.exists(os.path.join(save_dir, "pkl")):
    os.makedirs(os.path.join(save_dir, "pkl"))

if not os.path.exists(os.path.join(save_dir, "progress")):
    os.makedirs(os.path.join(save_dir, "progress"))




print "Saving config ..."
with open(os.path.join(save_dir, 'config', exp_name + '.pkl'), 'w') as f:
    cPickle.dump(args, f)
print "Finished saving."

w_init = initialization.IsotropicGaussian(0.01)
b_init = initialization.Constant(0.)

train_stream = parrot_stream(
    args.dataset, args.use_speaker, ('train',), args.batch_size,
    noise_level=args.feedback_noise_level, labels_type=args.labels_type,
    seq_size=args.seq_size)

if args.feedback_noise_level is None:
    val_noise_level = None
else:
    val_noise_level = 0.

valid_stream = parrot_stream(
    args.dataset, args.use_speaker, ('valid',), args.batch_size,
    noise_level=val_noise_level, labels_type=args.labels_type,
    seq_size=10000)

example_batch = next(train_stream.get_epoch_iterator())

for idx, source in enumerate(train_stream.sources):
    if source not in ['start_flag', 'feedback_noise_level']:
        print source, "shape: ", example_batch[idx].shape, \
            source, "dtype: ", example_batch[idx].dtype
    else:
        print source, ": ", example_batch[idx]

parrot_args = {
    'input_dim': args.input_dim,
    'output_dim': args.output_dim,
    'rnn_h_dim': args.rnn_h_dim,
    'readouts_dim': args.readouts_dim,
    'labels_type': args.labels_type,
    'weak_feedback': args.weak_feedback,
    'full_feedback': args.full_feedback,
    'feedback_noise_level': args.feedback_noise_level,
    'layer_norm': args.layer_norm,
    'use_speaker': args.use_speaker,
    'num_speakers': args.num_speakers,
    'speaker_dim': args.speaker_dim,
    'which_cost': args.which_cost,
    'num_characters': args.num_characters,
    'attention_type': args.attention_type,
    'attention_alignment': args.attention_alignment,
    'encoder_type': args.encoder_type,
    'weights_init': w_init,
    'biases_init': b_init,
    'name': 'parrot'}

parrot = Parrot(**parrot_args)
parrot.initialize()

features, features_mask, labels, labels_mask, speaker, latent_var, start_flag = \
    parrot.symbolic_input_variables()

cost, extra_updates, attention_vars, kl_cost = parrot.compute_cost(
    features, features_mask, labels, labels_mask,
    speaker, start_flag, args.batch_size)

cost_name = 'nll'
cost.name = cost_name

if parrot.use_latent:
    kl_cost.name = "KL"

cg = ComputationGraph(cost)
model = Model(cost)

gradients = None
if args.adaptive_noise:
    from graph import apply_adaptive_noise

    cost, cg, gradients, noise_brick = \
        apply_adaptive_noise(
            computation_graph=cg,
            cost=cost,
            variables=cg.parameters,
            num_examples=900,
            parameters=model.get_parameter_dict().values())

    model = Model(cost)
    cost_name = 'nll'
    cost.name = cost_name

parameters = cg.parameters

if args.algorithm == "adam":
    step_rule = CompositeRule(
        [StepClipping(10. * args.grad_clip), Adam(args.learning_rate)])
elif args.algorithm == "adasecant":
    step_rule = Adasecant(grad_clip=args.grad_clip)

algorithm = GradientDescent(
    cost=cost,
    parameters=parameters,
    step_rule=step_rule,
    gradients=gradients,
    on_unused_sources='warn')
algorithm.add_updates(extra_updates)

monitoring_vars = [cost]
plot_names = [['train_nll', 'valid_nll']]

if parrot.use_latent:
    monitoring_vars.append(kl_cost)
    plot_names.append(['train_KL', 'valid_KL'])


if args.lr_schedule:
    lr = algorithm.step_rule.components[1].learning_rate
    monitoring_vars.append(lr)
    plot_names += [['valid_learning_rate']]

train_monitor = TrainingDataMonitoring(
    variables=monitoring_vars,
    every_n_batches=args.save_every,
    prefix="train")

valid_monitor = DataStreamMonitoring(
    monitoring_vars,
    valid_stream,
    every_n_batches=args.save_every,
    after_epoch=False,
    prefix="valid")


# Multi GPU
worker = None
if args.platoon_port:
    from blocks_extras.extensions.synchronization import (
        Synchronize, SynchronizeWorker)
    from platoon.param_sync import ASGD

    sync_rule = ASGD()
    worker = SynchronizeWorker(
        sync_rule, control_port=args.platoon_port, socket_timeout=2000)

extensions = []

if args.load_experiment and (not worker or worker.is_main_worker):
    extensions += [Load(os.path.join(
        save_dir, "pkl", "best_" + args.load_experiment + ".tar"))]

extensions += [
    Timing(every_n_batches=args.save_every),
    train_monitor]

if not worker or worker.is_main_worker:
    extensions += [
        valid_monitor,
        TrackTheBest(
            'valid_nll',
            every_n_batches=args.save_every,
            before_first_epoch=True),
        Plot(
            os.path.join(save_dir, "progress", exp_name + ".png"),
            plot_names,
            every_n_batches=args.save_every,
            email=False),
        Checkpoint(
            os.path.join(save_dir, "pkl", "best_" + exp_name + ".tar"),
            after_training=False,
            save_separately=['log'],
            use_cpickle=True,
            save_main_loop=False,
            before_first_epoch=True)
        .add_condition(
            ["after_batch", "before_training"],
            predicate=OnLogRecord('valid_nll_best_so_far')),
        Checkpoint(
            os.path.join(save_dir, "pkl", "last_" + exp_name + ".tar"),
            after_training=True,
            save_separately=['log'],
            use_cpickle=True,
            every_n_batches=args.save_every,
            save_main_loop=False)]

    if args.lr_schedule:
        extensions += [
            LearningRateSchedule(
                lr, 'valid_nll',
                os.path.join(save_dir, "pkl", "best_" + exp_name + ".tar"),
                patience=10,
                num_cuts=5,
                every_n_batches=args.save_every)]

if worker:
    extensions += [
        Synchronize(
            worker,
            after_batch=True,
            before_epoch=True)]

extensions += [
    Printing(
        after_epoch=False,
        every_n_batches=args.save_every)]

if args.time_limit:
    extensions += [TimedFinish(args.time_limit)]

main_loop = MainLoop(
    model=model,
    data_stream=train_stream,
    algorithm=algorithm,
    extensions=extensions)

print "Training starting:"
main_loop.run()
