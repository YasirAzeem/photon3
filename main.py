
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import tensorflow as tf
import flask
from flask import request, jsonify
import os
import requests


app = flask.Flask(__name__)
dir_path = os.path.dirname(os.path.realpath(__file__))

@app.errorhandler(404)
def page_not_found(e):
    return "<h1>404</h1><p>The resource could not be found!!.</p>", 404


@app.errorhandler(500)
def page_not_found(e):
    return "<h1>500</h1><p>The resource could not be found!!.</p>", 500

@app.route('/', methods=['GET'])
def home():
    return '''<h1>Incision Classifier is running</h1>
<p>A prototype API for distant reading of science fiction novels.</p>'''


@app.route('/api', methods=['GET'])
def api_id():
    url = request.args['img']
    response = requests.get(url, stream=True)
    print(url)
    with open('1.jpg', 'wb') as handle:
        if not response.ok:
            print(response)
        for block in response.iter_content(1024):
          if not block:
            break
          handle.write(block)

    try:
        output =classifier('1.jpg')
        os.remove('1.jpg')
        return output
    except Exception as e:
        print(e)
        return '''<h1>Error! 404</h1>'''


def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
				input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label


def classifier(given):
  model_file = "retrained_graph.pb"
  label_file = "retrained_labels.txt"
  input_height = 299
  input_width = 299
  input_mean = 0
  input_std = 255
  input_layer = "Mul"
  output_layer = "final_result"

  file_name = dir_path +'/'+ given

  graph = load_graph(model_file)
  t = read_tensor_from_image_file(file_name,
                                  input_height=input_height,
                                  input_width=input_width,
                                  input_mean=input_mean,
                                  input_std=input_std)

  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name);
  output_operation = graph.get_operation_by_name(output_name);

  with tf.Session(graph=graph) as sess:
    start = time.time()
    results = sess.run(output_operation.outputs[0],
                      {input_operation.outputs[0]: t})
    end=time.time()
  results = np.squeeze(results)

  top_k = results.argsort()[-5:][::-1]
  labels = load_labels(label_file)
  return jsonify({labels[0]: str(results[0]),labels[1]: str(results[1])})

if __name__ == '__main__':
    print('Running Server!')
    app.run()
