import os


def get_model_file(name, root=os.path.join(os.path.expanduser('~'), '.torch', 'models')):
    file_path = os.path.join(root, name + '.pth')
    if os.path.exists(file_path):
        return file_path
    else:
        raise ValueError('Model file is not found. Please convert it from gluon first.')
