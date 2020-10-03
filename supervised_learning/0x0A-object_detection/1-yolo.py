#!/usr/bin/env python3
"""
 1-yolo module
 contains the Yolo class
"""
import tensorflow.keras as K
import numpy as np


class Yolo:
    """
    class Yolo uses the Yolo v3 algorithm to perform object detection
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        class constructor

        Args:
          model_path: `str`, the path to where a Darknet Keras model is stored
          classes_path: `str`, the path to where the list of class names used
            for the Darknet model, listed in order of index, can be found
          class_t: `float` representing the box score threshold for the initial
            filtering step
          nms_t: `float` representing the IOU threshold for non-max suppression
          anchors: `numpy.ndarray` of shape (outputs, anchor_boxes, 2)
            containing all of the anchor boxes:
              outputs: `int`, is the number of outputs (predictions) made
                by the Darknet model
              anchor_boxes: `int`, is the number of anchor boxes used for each
                prediction
              2 => [anchor_box_width, anchor_box_height]

        Public instance attributes:
          model: the Darknet Keras model
          class_names: a list of the class names for the model
          class_t: the box score threshold for the initial filtering step
          nms_t: the IOU threshold for non-max suppression
          anchors: the anchor boxes
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
		"""
		Process outputs

		Parameters
        ----------
        outputs : list of numpy.ndarrays 
            Contains the predictions from the Darknet model for a single image
			Each output will have the shape:
			(grid_height, grid_width, anchor_boxes, 4 + 1 + classes)
				grid_height & grid_width => the height and width of the grid
					used for the output
				anchor_boxes => the number of anchor boxes used
				4 => (t_x, t_y, t_w, t_h)
				1 => box_confidence
				classes => class probabilities for all classes
		image_size : numpy.ndarray
			Contains the image’s original size [image_height, image_width]

        Returns
        -------
        (boxes, box_confidences, box_class_probs) : tuple
			boxes: a list of numpy.ndarrays of shape (grid_height, grid_width,
				anchor_boxes, 4) containing the processed boundary boxes for
				each output, respectively:
				+ 4 => (x1, y1, x2, y2)
				+ (x1, y1, x2, y2) should represent the boundary box relative
				to original image
			box_confidences: a list of numpy.ndarrays of shape (grid_height,
				grid_width, anchor_boxes, 1) containing the box confidences
				for each output, respectively
			box_class_probs: a list of numpy.ndarrays of shape (grid_height,
				grid_width, anchor_boxes, classes) containing the box’s class
				probabilities for each output, respectively
		"""
        boxes = []
        box_confidence = []
        box_class_probs = []
        image_height, image_width = image_size

		
        for i in range(len(outputs)):
            grid_height, grid_width, anchor_boxes, _ = outputs[i].shape
            t_x = outputs[i][:, :, :, 0]
            t_y = outputs[i][:, :, :, 1]
            t_w = outputs[i][:, :, :, 2]
            t_h = outputs[i][:, :, :, 3]

            p_w = self.anchors[i, :, 0]
            p_h = self.anchors[i, :, 1]

            cx = np.array([np.arange(grid_width) for i in range(grid_height)])
            cx = cx.reshape(grid_width, grid_width, 1)
            cy = np.array([np.arange(grid_width) for i in range(grid_height)])
            cy = cy.reshape(grid_height,
                            grid_height).T.reshape(grid_height, grid_height, 1)

            bx = ((1 / (1 + np.exp(-t_x))) + cx) / grid_width
            by = ((1 / (1 + np.exp(-t_y))) + cy) / grid_height

            bw = p_w * np.exp(t_w)
            bh = p_h * np.exp(t_h)

            bw /= self.model.input.shape[1].value
            bh /= self.model.input.shape[2].value

            boxes[i][:, :, :, 0] = (bx - (bw / 2)) * image_width
            boxes[i][:, :, :, 1] = (by - (bh / 2)) * image_height
            boxes[i][:, :, :, 2] = (bx + (bw / 2)) * image_width
            boxes[i][:, :, :, 3] = (by + (bh / 2)) * image_height

            box_conf = (1 / (1 + np.exp(-outputs[i][:, :, :, 4:5])))
            box_conf.reshape(grid_height, grid_width, anchor_boxes, 1)
            box_confidence.append(box_conf)

            box_class = (1 / (1 + np.exp(-outputs[i][:, :, :, 5:])))
            box_class_probs.append(box_class)

        return boxes, box_confidence, box_class_probs
