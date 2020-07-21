def Experimental(f):
    def decorated(*args, **kwargs):
        return f(*args, **kwargs)
    return decorated
