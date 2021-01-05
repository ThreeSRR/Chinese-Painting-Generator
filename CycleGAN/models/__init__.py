import importlib
from models.base_model import BaseModel


def find_model_using_name(model_name):

    model_filename = "models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model


def create_model(gpu_ids='0', isTrain=True, checkpoints_dir='./checkpoints', name='experiment_name', continue_train=False, model='cycle_gan'):

    model = find_model_using_name(model)
    instance = model(gpu_ids=gpu_ids, isTrain=isTrain, checkpoints_dir=checkpoints_dir, name=name, continue_train=continue_train)
    print("model [%s] was created" % type(instance).__name__)
    return instance
