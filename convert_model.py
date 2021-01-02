# ModularGAN extend build_graph fn
import tensorflow as tf
from compare_gan.gans.modular_gan import ModularGAN


def build_graph(self, model=None, is_training=False):
    batch_size = None
    y_gen, y_disc = None, None
    inputs_gen, inputs_disc = {}, {}
    outputs_gen, outputs_disc = {}, {}
    inputs_gen["z"] = None if model=='disc' else tf.placeholder(
        shape=(batch_size, self._z_dim),
        dtype=tf.float32,
        name="z_for_eval"
    )
    inputs_disc["images"] = None if model=='gen' else tf.placeholder(
        shape=[batch_size] + list(self._dataset.image_shape),
        dtype=tf.float32,
        name="images_for_eval"
    )
    if self.conditional:
        inputs_gen["labels"] = None if model=='disc' else tf.placeholder(
            shape=(batch_size,),
            dtype=tf.int32,
            name="labels_for_gen_eval"
        )
        y_gen = None if model=='disc' else self._get_one_hot_labels(inputs_gen["labels"])
        inputs_disc["labels"] = None if model=='gen' else tf.placeholder(
            shape=(batch_size,),
            dtype=tf.int32,
            name="labels_for_disc_eval"
        )
        y_disc = None if model=='gen' else self._get_one_hot_labels(inputs_disc["labels"])
    else:
      y_gen, y_disc = None, None

    if model != 'gen':
      outputs_disc["prediction"], _, _ = self.discriminator(
          inputs_disc["images"], y=y_disc, is_training=is_training
      )

    z = inputs_gen["z"]
    generated = None if model=='disc' else self.generator(z=z, y=y_gen, is_training=is_training)
    generated_ema = generated
    if self._g_use_ema and model != 'disc':
        g_vars = [var for var in tf.trainable_variables()
              if "generator" in var.name]
        ema = tf.train.ExponentialMovingAverage(decay=self._ema_decay)
        # Create the variables that will be loaded from the checkpoint.
        ema.apply(g_vars)
        def ema_getter(getter, name, *args, **kwargs):
            var = getter(name, *args, **kwargs)
            ema_var = ema.average(var)
            if ema_var is None:
                var_names_without_ema = {"u_var", "u_var_update", "accu_mean", "accu_variance",
                              "accu_counter", "update_accus"}
                if name.split("/")[-1] not in var_names_without_ema:
                    print("Could not find EMA variable for %s." % name)
                return var
            return ema_var
        with tf.variable_scope("", values=[z, y_gen], reuse=True, custom_getter=ema_getter), gin.config_scope("ema"):
            generated_ema = self.generator(z, y=y_gen, is_training=is_training)
    if model != 'disc':
      outputs_gen["generated"] = generated
      outputs_gen["generated_ema"] = generated_ema
    return {
        "gen": {
            "inputs": inputs_gen, 
            "outputs": outputs_gen
        },
        "disc": {
            "inputs": inputs_disc,
            "outputs": outputs_disc
        }
    }
ModularGAN.build_graph = build_graph



class FakeDataset:
    num_classes = 1000
    random_labels = True
    image_shape = (256, 256, 3)


from natsort import natsorted
import os
import compare_gan.main
import compare_gan.runner_lib
import compare_gan.datasets
import gin
from scipy.stats import truncnorm
import numpy as np


def get_latest_gin_config(model_dir):
  return list(natsorted(tf.io.gfile.glob(os.path.join(model_dir, '*.gin'))))[-1]


def parse_gin_config(gin_config_path):
  with tf.io.gfile.GFile(gin_config_path) as f:
    cfg_txt = f.read()
  with open('config.gin', 'w') as f:
    f.write(cfg_txt)
  gin.parse_config_file('config.gin')


class CompareGANLoader:
  def __init__(self, model_dir, gin_config=None, use_ema=None):
    if gin_config is None:
      gin_config = get_latest_gin_config(model_dir)
    parse_gin_config(gin_config)
    compare_gan.runner_lib.FLAGS.__dict__['__flags_parsed'] = True # hack to load options
    self.options = compare_gan.runner_lib.get_options_dict()
    self.dataset = compare_gan.datasets.get_dataset(options=self.options)
    if use_ema is None:
      use_ema = gin.query_parameter('ModularGAN.g_use_ema')
    self.use_ema = use_ema
    self.model_dir = model_dir
    self.model = ModularGAN(self.dataset, self.options, self.model_dir, g_use_ema=self.use_ema)

  @property
  def initializers(self):
    return [v.initializer for v in tf.local_variables() if v.name.rsplit('/', 1)[-1].rsplit(':',1)[0].startswith('u_var')]

  def u_var_set_updates_allowed(self, enabled=1):
    return [tf.assign(v, enabled) for v in tf.local_variables() if v.name.rsplit('/', 1)[-1].rsplit(':',1)[0].startswith('u_var_update')]

  def load(self, ckpt=None, session=None):
    if session is None:
      session = tf.get_default_session()
    session.run(self.initializers)
    saver = tf.train.Saver();
    if ckpt == None:
      ckpt = tf.train.latest_checkpoint(self.model_dir)
    saver.restore(session, ckpt)
    return saver

  def build(self, is_training, which='gen'):
    if which != 'gen' and which != 'disc' and which is not None:
      raise ValueError("Expected which to be 'gen', 'disc', or None")
    with tf.variable_scope("", reuse=tf.AUTO_REUSE):
      return self.model.build_graph(model=which, is_training=is_training)

  def random_state(self, seed=None):
    return np.random.RandomState(seed)

  def truncated_z_sample(self, batch_size=1, z_dim=None, truncation=1.0, seed=None):
    if z_dim is None:
      z_dim = self.model._z_dim
    rand = self.random_state(seed=seed)
    values = truncnorm.rvs(-2, 2, size=(batch_size, z_dim), random_state=rand)
    return truncation * values

  @property
  def num_classes(self):
    return self.options['num_classes']

  def one_hot(self, i, num_classes=None):
    if num_classes is None:
      num_classes = self.num_classes
    if isinstance(i, int):
      i = [i]
      return self.one_hot(i, num_classes=num_classes)[0]
    a = np.array(i, dtype=np.int32)
    #num_classes = a.max()+1
    b = np.zeros((a.size, num_classes))
    b[np.arange(a.size),a] = 1
    return b

  def run(self, built, z=None, y=None, seed=None, ema_only=False, session=None):
    if session is None:
      session = tf.get_default_session()
    if y is None:
      rand = self.random_state(seed=seed)
      y = rand.randint(self.num_classes)
    if isinstance(y, int):
      y = self.one_hot(y)
    if z is None:
      z = self.truncated_z_sample(seed=seed)
    g_outputs = built['gen']['outputs']
    if ema_only:
      g_outputs = g_outputs['generated_ema']
    return session.run(g_outputs, {
          built['gen']['inputs']['labels']: y,
          built['gen']['inputs']['z']: z,
        })



def f32rgb_to_u8(x):
  return tf.cast(tf.math.floordiv(tf.clip_by_value(x*256.0, 0.0, 255.0), 1.0), tf.uint8)


def fimg2png(img):
  return tf.image.encode_png(f32rgb_to_u8(img))
  #return tf.image.encode_png(f32rgb_to_u8((img + 1) / 2))


def datdisk(data, filename=None):
  if filename is None:
    filename = 'bgd512_0.png' if deep else 'bg{res}_0.png'.format(res=res)
  with open(filename, 'wb') as f:
    f.write(data)
  return filename




#----------------------------------------------------------------------------
# Image utils.

import PIL.Image

def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data

def create_image_grid(images, grid_size=None):
    if images.shape[-1] == 3:
      images = images.transpose(0,3,1,2)
    assert images.ndim == 3 or images.ndim == 4
    num, img_w, img_h = images.shape[0], images.shape[-1], images.shape[-2]

    if grid_size is not None:
        grid_w, grid_h = tuple(grid_size)
    else:
        grid_w = max(int(np.ceil(np.sqrt(num))), 1)
        grid_h = max((num - 1) // grid_w + 1, 1)

    grid = np.zeros(list(images.shape[1:-2]) + [grid_h * img_h, grid_w * img_w], dtype=images.dtype)
    for idx in range(num):
        x = (idx % grid_w) * img_w
        y = (idx // grid_w) * img_h
        grid[..., y : y + img_h, x : x + img_w] = images[idx]
    return grid

def convert_to_pil_image(image, drange=[0,1]):
    assert image.ndim == 2 or image.ndim == 3
    if image.ndim == 3:
        if image.shape[0] == 1:
            image = image[0] # grayscale CHW => HW
        else:
            image = image.transpose(1, 2, 0) # CHW -> HWC

    image = adjust_dynamic_range(image, drange, [0,255])
    image = np.rint(image).clip(0, 255).astype(np.uint8)
    fmt = 'RGB' if image.ndim == 3 else 'L'
    return PIL.Image.fromarray(image, fmt)

def save_image_grid(images, filename, drange=[0,1], grid_size=None):
    convert_to_pil_image(create_image_grid(images, grid_size), drange).save(filename)


def with_dep(dep, op):
  with tf.control_dependencies(dep() if callable(dep) else dep):
    return op() if callable(op) else tf.identity(op)


# import convert_model; reload(convert_model); loader = convert_model.CompareGANLoader( 'gs://arfa-euw4/runs/bigrun/nov19_01' ); inout = loader.build(is_training=True)
# loader.load() # only need to do this once
# convert_model.datdisk(r(convert_model.fimg2png(loader.run(inout, z=loader.truncated_z_sample(seed=0, truncation=1.0), y=[1])['generated_ema'][0])), 'foo.png')
# or
# samples = loader.run(inout, z=loader.truncated_z_sample(seed=0, truncation=1.0), y=[1])['generated'][0]
# PIL.Image.fromarray((sample*255).astype(np.uint8)).save('foo.png')


# seed=None; rng = np.random.RandomState(seed=seed); batch_size = 16; sample = r(inout['gen']['outputs']['generated_ema'], {inout['gen']['inputs']['z']: loader.truncated_z_sample(seed=seed, truncation=1.0, batch_size=batch_size), inout['gen']['inputs']['labels']: [rng.randint(loader.num_classes) for _ in range(batch_size)]}); convert_model.save_image_grid(sample, 'generated.png')


# means, variances, counters = [v for v in tf.global_variables() if "accu/accu_mean" in v.name], [v for v in tf.global_variables() if "accu/accu_variance" in v.name], [v for v in tf.global_variables() if "accu/accu_counter" in v.name]; commit_accu = [(m.assign(m / c, read_value=False), v.assign(v / c, read_value=False), c.assign(1, read_value=False)) for m, v, c in zip(means, variances, counters)];


# r([v.initializer for v in tf.all_variables() if v.name.rsplit('/', 1)[-1].rsplit(':',1)[0] == 'u_var']); r([v.initializer for v in tf.all_variables() if 'accu/' in v.name]); r(enable_updates); r(disable_update_accus); 

#import random; p = lambda *x: print(*x) or x[0]; seed = p(random.randint(0, 1000), 'seed'); rng = np.random.RandomState(seed=seed); batch_size = 4; sample = r(inout['gen']['outputs']['generated_ema'], {inout['gen']['inputs']['z']: loader.truncated_z_sample(seed=seed, truncation=0.1, batch_size=batch_size), inout['gen']['inputs']['labels']: [rng.randint(loader.num_classes) for _ in range(batch_size)]}); convert_model.save_image_grid(sample, 'generated.png');



# means, variances, counters = [v for v in tf.global_variables() if "accu/accu_mean" in v.name], [v for v in tf.global_variables() if "accu/accu_variance" in v.name], [v for v in tf.global_variables() if "accu/accu_counter" in v.name]; commit_accu_dep = [tf.group([(m.assign(m / tf.math.maximum(1.0, c), read_value=False), v.assign(v / tf.math.maximum(1.0,c), read_value=False))]) for m, v, c in zip(means, variances, counters)]; commit_accu = with_dep(commit_accu_dep, lambda: [c.assign(1) for m, v, c in zip(means, variances, counters)])
