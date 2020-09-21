import gpt_2_simple as gpt2

model_dir="/home/gabriel/opt/models/"
file_name = "/home/gabriel/opt/93732702_2020-08_6418.txt"
checkpoint_dir = "/home/gabriel/opt/checkpoint"

sess = gpt2.start_tf_sess()

gpt2.finetune(sess,
              dataset=file_name,
              model_dir=model_dir,
              model_name='model',
              checkpoint_dir=checkpoint_dir,
              steps=-1,
              learning_rate=0.00001,
              use_memory_saving_gradients=True,
              restore_from='fresh',
              print_every=20,
              sample_every=10000,
              #overwrite=True,
              save_every=200
              )