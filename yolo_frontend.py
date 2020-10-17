from keras.models import Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
import tensorflow as tf
import numpy as np
import os
import cv2
from yolo_utils import decode_netout
from keras.applications.mobilenet import MobileNet
from keras.layers.merge import concatenate
from keras.optimizers import SGD, Adam, RMSprop
from yolo_preprocessing import YoloBatchGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
#from yolo_backend import TinyYoloFeature
import keras
import sys
import matplotlib.pyplot as plt
#import threading

class SpecialYOLO(object):
    def __init__(self, input_width,
                       input_height,
                       labels ):

        self.input_width = input_width
        self.input_height = input_height

        self.labels   = list(labels)
        self.nb_class = len(self.labels)
        self.nb_kpp   = 1  #predefined number of keypoint pairs per grid cell, here always 1, can be removed
        self.class_wt = np.ones(self.nb_class, dtype='float32')
        self.max_kpp_per_image = 1  #kpp = key point pairs
        self.seen = 0

        ##########################
        # Make the model
        ##########################

        # make the feature extractor layers
        input_image    = Input(shape=(self.input_height, self.input_width, 1))# Input image with height,width and image with only one channel

        num_layer = 0# intial layers assigned as 0

        # stack 1
        x = Conv2D(16, (3,3), strides=(1,1), padding='same', name='conv_' + str( num_layer ), use_bias=False)(input_image)  #16
        x = BatchNormalization(name='norm_' + str( num_layer ))(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        num_layer += 1

        # stack 2
        for i in range(0,2):  #(0,2)
            #x = Conv2D(32*(2**i), (3,3), strides=(1,1), padding='same', name='conv_' + str(num_layer), use_bias=False)(x)  #32
            #x = BatchNormalization(name='norm_' + str(num_layer))(x)
            #x = LeakyReLU(alpha=0.1)(x)
            #num_layer += 1

            x = Conv2D(32*(2**i), (3,3), strides=(1,1), padding='same', name='conv_' + str(num_layer), use_bias=False)(x)  #32
            x = BatchNormalization(name='norm_' + str(num_layer))(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            num_layer += 1

        # stack 3
        for i in range(0,20):
            x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_' + str(num_layer), use_bias=False)(x)  #32
            x = BatchNormalization(name='norm_' + str(num_layer))(x)
            x = LeakyReLU(alpha=0.1)(x)
            num_layer += 1

        x = Conv2D(3+1+ self.nb_class, (3,3), strides=(1,1), padding='same', name='conv_'+str( num_layer ), use_bias=False)(x)
        x = BatchNormalization(name='norm_' + str( num_layer ))(x)
        x = LeakyReLU(alpha=0.1)(x)
        num_layer += 1

        # make the object detection layer
        output = Conv2D(self.nb_kpp * (3 + 1 + self.nb_class),#(x,y,alpha+conf+no.of classes)
                        (1,1), strides=(1,1),
                        padding='same',
                        name='DetectionLayer',
                        kernel_initializer='lecun_normal')(x)


        print( "x.shape=", x.shape.as_list() )
        self.grid_h = x.shape.as_list()[1]
        self.grid_w = x.shape.as_list()[2]

        print( "self.grid_h, self.grid_w=", self.grid_h, self.grid_w )

        output = Reshape((self.grid_h, self.grid_w, self.nb_kpp, 3 + 1 + self.nb_class))(x)


        print( "model_1 input shape=", input_image.shape )
        print( "model_2 output shape=", output.shape )

        self.model = Model(inputs=input_image, outputs=output)

        #--------------------------------------------------------------------------------
        # self.model.load_weights( "transparent.h5" )
        #--------------------------------------------------------------------------------


        # print a summary of the whole model
        self.model.summary(positions=[.25, .60, .80, 1.])
        # tf.logging.set_verbosity(tf.logging.INFO)  ## testein

    def custom_loss(self, y_true, y_pred):
        # shape of y_pred and y_true: <batch_size> <gridsize_x> <gridsize_y> <1> <5> where last dimension <5> is <x0 y0 alpha conf (classes as one-hot vector)>
        # before-last dimension <1> is a remains from YOLOV2 which is not needed here. This dimension could be deleted here.
        # y_true, y_pred are data of a complete batch
        # y_true are grid-coordinate units (here e.g. 0...(gridsize_x-1))
        # y_pred are cell-coordinate units (0...1 within the Cell)

        # netout must be in image_width and image_height units, i.e. in the range [0...1] over the whole image...really??
        print_op = tf.print( "y_true[...,2]=", y_true[...,2], output_stream=sys.stderr, summarize=-1)
        with tf.control_dependencies([print_op]):
            y_true = tf.add( y_true, 0 )
        mask_shape = tf.shape(y_true)[:4] # mask_shape is then (batch_size, nb_grid_x, nb_grid_y, 1)

        #print_op = tf.print( "mask_shape=", mask_shape, output_stream=sys.stderr)
        #with tf.control_dependencies([print_op]):
        #    mask_shape = tf.add( mask_shape, 0 )


        # cell_x and cell_y then contain their own x and y coordinates, one for each cell, this is a lookup table for fast coordinate access in training loop
        cell_x = tf.cast( tf.reshape(tf.tile(tf.range(self.grid_w), [self.grid_h]), (1, self.grid_h, self.grid_w, 1, 1)), dtype=tf.float32 )  #same dimension number as y_pred, contains grid cell x coordinates
        cell_y = tf.cast( tf.reshape( tf.transpose( tf.reshape( tf.tile( tf.range(self.grid_h), [self.grid_w] ),(self.grid_w, self.grid_h))),(1, self.grid_h, self.grid_w, 1, 1)), dtype=tf.float32 )#same dimension number as y_pred, contains grid cell y coordinates


        # cell_grid contains x and y coordinates for each batch and for each grid cell, this is a lookup table for fas coordinate access in training loop
        cell_grid = tf.tile(tf.concat([cell_x,cell_y], -1), [self.batch_size, 1, 1, self.nb_kpp, 1]) # has now x and y coordinates ascending in one dimension [[x, x, x, x, y, y, y, y],[x,x,x,x,y,y,y,y],...]...

        # Intialize all masks with 0
        coord_mask = tf.zeros(mask_shape, dtype='float32')#used to locate the coordinates of the keypoints
        conf_mask  = tf.zeros(mask_shape, dtype='float32')#Used to tell the confidence of the located keypoints
        class_mask = tf.zeros(mask_shape, dtype='float32')#Used to identify the class of the object

        total_recall = tf.Variable(0., dtype='float32')  #used in status report only

        """
        Extract  p r e d i c t i o n  xy, alpha, confidence and class from y_pred
        """
        ### extract and transform predicted keypoint pair coordinates to grid coordinates
        # take x and y from last dimension only and transform from cell coordinmates to grid coordinates and limit to [0...1], predicted keypoint0 grid coordinates are the first two elements in last dimension
        # keypoint0 is the position of object
        # shape is (batch_size, gridsize_x, gridsize_y, <1 == nb_anchors>, <2 == x, y>))
        pred_kp0_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid

        pred_alpha = (y_pred[..., 2:3])  # predicted alphas in in element 2 in last dimension shape is (batch_size, gridsize_x, gridsize_y, nb_anchors, <1 == alpha))

        ### extract (=limit to [0...1]) predicted confidence
        pred_kpp_conf = tf.sigmoid(y_pred[..., 3]) #predicted keypoint pair confidences shape is (batch_size, gridsize_x, gridsize_y, nb_anchors, <3==confidence>))

        ### extract predicted class probabilities
        pred_kpp_class = y_pred[..., 4:]  # one or more classes (one-hot) starting with element 4 in last dimension,shape is (batch_size, gridsize_x, gridsize_y, <1==nb_anchors>, <3==confidence>)), why tf.argmax is missing?


        """
        Extract  g r o u n d   t r u t h   xy, alpha, conf and class from y_true
        """
        ### keypoint0 x y alpha
        true_kp0_xy = y_true[..., 0:2] # x-y-coordinates of y_true, in grid cells units, LUT
        true_alpha = y_true[..., 2:3] # alpha

        ### keypoint-pair confidence
        true_kpp_conf = tf.sigmoid(y_true[..., 3]) # confidence
        ### The argmax function determines the index tensor of the maximum arguments over the last dimension. But here the last axis is specified
        ### i.e. the shape of the result vector determines the argmax of the lowest dimension. The lowest dimension is removed from the shape.
        ### Here Shape of the result vector: (nb_batches, grid_x, grid_y, nb_kpp ), where the last dimension contains the respective argmax (here always 1)
        true_kpp_class = y_true[..., 4:]  # one hot class vectors
        true_kpp_class_argmax = tf.argmax(y_true[..., 4:], -1) # index of highest class


        """
        Compose the masks for loss calculation and punishing
        """
        ### coordinate mask: simply the position of the ground truth boxes (the predictors)
        # this is the confidence for each keypoint pair, multiplied by the coord_scale. A dimension is appended to this.
        coord_mask = tf.expand_dims(y_true[..., 3], axis=-1) * self.coord_scale  # extract 3rd axis only, this is <nb_anchors>, and append a dimension with size 1 at the end -> (nb_batches, nb_grid_x, nb_grid_y, 1 )

        #test print
        #print_op = tf.print( "coord_mask=", coord_mask, output_stream=sys.stderr, summarize=-1 )
        #with tf.control_dependencies([print_op]):
        #    coord_mask = tf.add( coord_mask, 0 )


        # at the end conf_mask has all elements set either to no_object_scale or to object_scale
        # penalize the confidence difference of all keypoints which are farer away from true keypoints
        conf_mask = conf_mask + 1.0  # set all to 1.0
        #conf_mask.shape==(nb_batches, nb_grid_x, nb_grid_y, nb_anchors)

        # penalize the confidence difference of all keypoints which are reponsible for corresponding ground truth keypoint0
        conf_mask = conf_mask + y_true[..., 3] * self.object_scale  # set the cells containing keypoints to object_scale, remove last dimension from y_true and keep conf values only

        ### class mask
        # class_mask = y_true[..., 3] * tf.gather(self.class_wt, true_kpp_class_argmax) * self.class_scale
        class_mask = y_true[..., 3] * tf.gather(self.class_wt, true_kpp_class_argmax) * self.class_scale


        """
        Warm-up training
        """
        no_kpp_mask = tf.cast(coord_mask < self.coord_scale/2., dtype=tf.float32)
        self.seen += 1
        #print_op = tf.print( "seen=", self.seen, output_stream=sys.stderr)
        #with tf.control_dependencies([print_op]):
        # self.seen = tf.add( self.seen, 0 )


        true_kp0_xy,  coord_mask = tf.cond(tf.less(self.seen, self.warmup_batches+1),
                              lambda: [true_kp0_xy + (0.5 + cell_grid) * no_kpp_mask,
                                       tf.ones_like(coord_mask)],
                              lambda: [true_kp0_xy,
                                       coord_mask])


        """
        Finalize the loss
        """

        # numbers are needed for a kind of normalization
        nb_coord_kpp = tf.reduce_sum(tf.cast(coord_mask > 0.0, dtype=tf.float32))
        nb_conf_kpp  = tf.reduce_sum(tf.cast(conf_mask  > 0.0, dtype=tf.float32))
        nb_class_kpp = tf.reduce_sum(tf.cast(class_mask > 0.0, dtype=tf.float32))

        loss_kp0_xy    = tf.reduce_sum(tf.square(true_kp0_xy-pred_kp0_xy) * coord_mask) / (nb_coord_kpp + 1e-6) / 2.
        loss_alpha    = tf.reduce_sum(tf.square(true_alpha-pred_alpha) * coord_mask * self.direction_scale) / (nb_coord_kpp + 1e-6) / 2.
        loss_conf  = tf.reduce_sum(tf.square(true_kpp_conf-pred_kpp_conf) * conf_mask)  / (nb_conf_kpp  + 1e-6) / 2.
        #loss_conf = tf.sigmoid( loss_conf )

        # test
        #class_mask_expanded = tf.expand_dims( class_mask, -1)
        loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_kpp_class_argmax, logits=pred_kpp_class)

        #loss_class  = tf.reduce_sum(tf.square(true_kpp_class-pred_kpp_class)*class_mask_expanded) / (nb_class_kpp + 1e-6)/2. # * tf.expand_dims( class_mask, axis=-1 ))  / (nb_class_kpp  + 1e-6) / 2.

        # testaus
        loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_kpp + 1e-6)
        #loss_class = tf.reduce_sum( tf.square(true_kpp_class - pred_kpp_class)*class_mask ) / (nb_class_kpp + 1e-6)
        #loss_class = tf.sigmoid( loss_class )

        #self.seen = tf.add( self.seen, 1 )

        loss = tf.cond(tf.less(self.seen, self.warmup_batches+1),
                      lambda: loss_kp0_xy + loss_alpha + loss_conf + loss_class, #100, # Adding bias as 10
                      lambda: loss_kp0_xy + loss_alpha + loss_conf + loss_class)*100



        #print_op = tf.print( "loss_class=", loss_class, output_stream=sys.stderr)
        #with tf.control_dependencies([print_op]):
        #    loss_class = tf.add( loss_class, 0 )

        if self.debug:
            nb_true_kpp = tf.reduce_sum(y_true[..., 3])
            nb_pred_kpp = tf.reduce_sum(tf.cast(true_kpp_conf > 0.5, dtype=tf.float32) * tf.cast(pred_kpp_conf > 0.3, dtype=tf.float32))

            current_recall = nb_pred_kpp/(nb_true_kpp + 1e-6)
            total_recall = total_recall + current_recall


        return loss

    def load_weights(self, weight_path):
        self.model.load_weights(weight_path)
        self.model.save(weight_path+"full")

        print( "input layer name=" )
        print( [node.op.name for node in self.model.inputs] )
        print( "output layer name=" )
        print( [node.op.name for node in self.model.outputs] )


        return self.model.output.shape[1:3]

    def normalize(self, image):
        return image / 255.

    def train(self, train_imgs,     # the list of images to train the model
                    valid_imgs,     # the list of images used to validate the model
                    nb_epochs,      # number of epoches
                    learning_rate,  # the learning rate
                    batch_size,     # the size of the batch
                    warmup_epochs,  # number of initial batches to let the model familiarize with the new dataset
                    object_scale,
                    no_object_scale,
                    coord_scale,
                    class_scale,
                    direction_scale,
                    saved_weights_name='transparent.h5',
                    debug=False,
                    train_times=1,
                    valid_times=1):

        self.batch_size = batch_size
        self.object_scale    = object_scale
        self.no_object_scale = no_object_scale
        self.coord_scale     = coord_scale
        self.class_scale     = class_scale
        self.direction_scale = direction_scale
        self.debug = debug

        ############################################
        # Make train and validation generators
        ############################################

        generator_config = {
            'IMAGE_H'         : self.input_height,
            'IMAGE_W'         : self.input_width,
            'GRID_H'          : self.grid_h,
            'GRID_W'          : self.grid_w,
            'KPP'             : self.nb_kpp,
            'LABELS'          : self.labels,
            'CLASS'           : len(self.labels),
            'BATCH_SIZE'      : self.batch_size,
        }

        train_generator = YoloBatchGenerator(train_imgs,
                                     generator_config,
                                     norm=self.normalize)
        valid_generator = YoloBatchGenerator(valid_imgs,
                                     generator_config,
                                     norm=self.normalize,
                                     jitter=False)

        self.warmup_batches  = warmup_epochs *(train_times*len(train_generator) + valid_times*len(valid_generator))

        ############################################
        # Compile the model
        ############################################

        optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(loss=self.custom_loss, optimizer=optimizer)

        ############################################
        # Make a few callbacks
        ############################################

        early_stop = EarlyStopping(monitor='val_loss',
                           min_delta=0.001,
                           patience=500000,#2or3
                           mode='min',
                           verbose=1)
        checkpoint = ModelCheckpoint(saved_weights_name,
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only = False,
                                     mode='min',
                                     period=1)
        tensorboard = TensorBoard(log_dir=os.path.expanduser('~/logs/'),
                                  histogram_freq=0,
                                  #write_batch_performance=True,
                                  write_graph=True,
                                  write_images=False)

        ############################################
        # Start the training process
        ############################################


        print( "call fit_generator" )
        self.model.fit_generator(generator        = train_generator,
                                 steps_per_epoch  = len(train_generator)*train_times,
                                 epochs           = warmup_epochs + nb_epochs,
                                 verbose          = 2 if debug else 1,
                                 validation_data  = valid_generator,
                                 validation_steps = len(valid_generator) * valid_times,
                                 callbacks        = [early_stop, checkpoint, tensorboard],
                                 workers          = 3,  # vormals 3
                                 max_queue_size   = 8)#8
                                 #use_multiprocessing = False)

        ############################################
        # Compute mAP on the validation set
        ############################################

        ##### test prediction ###########################
        print( "test prediction start\n" )
        image = cv2.imread("D:\\CNN SRIMAN\\report\\sriman20\\Likith_1000\\train\\Img_000000.bmp")
        image = image[:,:,0] # red channel only
        image = np.expand_dims( image, -1 )
        image = self.normalize( image )
        self.predict(image)
        print( "test prediction end\n" )
        ##### test prediction ende ######################


    def predict(self, image):
        print("image.shape=",image.shape)
        image_h, image_w, _  = image.shape
        #image = cv2.resize(image, (self.input_width, self.input_height))
        image = self.normalize(image)

        input_image = image[:,:,::-1] #flip rgb to bgr or vice versa

        input_image = np.expand_dims(input_image, 0)
        #dummy_array = np.zeros((1,1,1,1,self.max_kpp_per_image,4))

        netout = self.model.predict([input_image])[0]# add dummy_array

        print( "netout=", [netout] )  # print the netout


        netout_decoded = decode_netout(netout, image_w, image_h, self.nb_class)


        for kpp in netout_decoded:
            print( kpp.x0, kpp.y0, kpp.alpha_norm, kpp.c , kpp.classes )

        return netout_decoded
