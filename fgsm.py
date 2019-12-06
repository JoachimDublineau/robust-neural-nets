import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.losses as klosses
import tensorflow.keras.models as kmodels
import tensorflow.keras.utils as kutils
import tqdm

import src

parser = argparse.ArgumentParser(
    description='Attack a cifar10 model using FGSM method with cifar10 test set')
parser.add_argument('model', help='The model to attack as an .h5 file')
parser.add_argument('--epsilons', '-e', nargs='+', type=float, required=True,
                    help='Space separated list of epsilons values with which attack the model')
parser.add_argument('--batch-size', default=128, type=int,
                    help='The batch sized used to compute gradients. Default is 128')
parser.add_argument('--verbose', '-v', action='store_true',
                    help='If set, output details on the execution')
parser.add_argument('--output-csv', '-csv', default='{}fgsm_evaluation_log.csv'.format(src.results_dir),
                    help='the path of the csv file in which the script writes its outputs. Default is "{}fgsm_evaluation_log.csv"'.format(src.results_dir))
parser.add_argument('--save-img', type=int,
                    help='The number of random images to save for each epsilon. Images will be saved under "[output-csv]_[epsilon]_image.png"')
parser.add_argument('--tf-log-level', default='3',
                    choices=['0', '1', '2', '3'], help='Tensorflow minimum cpp log level. Default is 3')

args = parser.parse_args()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = args.tf_log_level

model = kmodels.load_model(args.model)
_, _, x_test, y_test = src.cifar10.load_data()  # Retrieve cifar10 test set
del _  # delete useless train set

x_test = x_test.astype('float32') / 255
y_test = kutils.to_categorical(y_test)

signed_gradients = src.attacks.compute_signed_gradients(
    x_test, y_test, model, klosses.categorical_crossentropy, batch_size=args.batch_size, verbose=args.verbose)
signed_gradients = np.clip(signed_gradients, 0.0, 1.0)

with open(args.output_csv, 'w') as output:
    output.write('epsilon;{}\n'.format(';'.join(model.metrics_names)))

    if args.verbose:
        args.epsilons = tqdm.tqdm(
            args.epsilons, desc='Attacking model for each epsilon', unit='epsilon')

    if args.save_img is not None:
        random_indexes = np.random.choice(x_test.shape[0], args.save_img)

    for epsilon in args.epsilons:
        x = np.clip(x_test + epsilon * signed_gradients, 0.0, 1.0)
        evaluation = model.evaluate(
            x, y_test, batch_size=args.batch_size, verbose=0)
        output.write('{};{}\n'.format(epsilon, ';'.join(
            map(lambda number: '{:.5f}'.format(number), evaluation))))

        if args.save_img is not None:
            x = x_test[random_indexes]
            y = y_test[random_indexes]
            y_predicted = model.predict(x, batch_size=args.batch_size)
            gradients = signed_gradients[random_indexes]
            x_adversarials = np.clip(x + epsilon * gradients, 0.0, 1.0)
            y_adversarials_predicted = model.predict(
                x_adversarials, batch_size=args.batch_size)

            fig, axes = plt.subplots(
                args.save_img, 3, squeeze=False, figsize=(9, 3 * args.save_img))
            font = {'fontsize': 16}

            iterator = range(args.save_img)
            if args.verbose:
                iterator = tqdm.tqdm(
                    iterator, desc='Computing images to save', leave=False, unit='image')

            for image_index in iterator:
                axes[image_index][0].imshow(x[image_index])
                axes[image_index][0].set_title('{}: {:.4f}'.format(src.cifar10.labels[np.argmax(
                    y_predicted[image_index])], np.max(y_predicted[image_index])), fontdict=font)
                axes[image_index][1].imshow(gradients[image_index])
                axes[image_index][1].set_title(
                    'Signed gradient', fontdict=font)
                axes[image_index][2].imshow(x_adversarials[image_index])
                axes[image_index][2].set_title('{}: {:.4f}'.format(src.cifar10.labels[np.argmax(
                    y_adversarials_predicted[image_index])], np.max(y_adversarials_predicted[image_index])), fontdict=font)

                for column_index in range(1, 3):
                    axes[image_index][column_index].set_axis_off()
                    axes[image_index][column_index].set_frame_on(False)

                axes[image_index][0].set_frame_on(False)
                axes[image_index][0].set_ylabel(
                    src.cifar10.labels[np.argmax(y[image_index])], fontdict=font)
                axes[image_index][0].set_xticks([])
                axes[image_index][0].set_yticks([])

            fig.savefig('{}_{}_image.png'.format(
                args.output_csv, epsilon), dpi=400, transparent=True)
