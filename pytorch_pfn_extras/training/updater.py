class Updater(object):

    def __call__(self):
        raise NotImplementedError(
            'Updater implementation must override __call__')