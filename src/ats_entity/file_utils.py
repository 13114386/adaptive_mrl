from __future__ import unicode_literals, print_function, division
import os

def get_save_load_dir(args, model_name):
    return os.path.join(args.modeldata_root, args.model_dir, model_name)
