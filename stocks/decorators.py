import json
from functools import wraps, update_wrapper
from datetime import datetime

from flask import request, Response, make_response

def accept(mimetype):
    def decorator(func):
        """
        Decorator which returns a 406 Not Acceptable if the client won't accept 
        a certain mimetype
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            if "application/json" in request.accept_mimetypes:
                return func(*args, **kwargs)
            message = "Request must accept {} data".format(mimetype)
            data = json.dumps({"message": message})
            return Response(data, 406, mimetype="application/json")
        return wrapper
    return decorator

def require(mimetype):
    def decorator(func):
        """
        Decorator which returns a 415 Unsupported Media Type if the client sends
        something other than a certain mimetype
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            if (request.mimetype ==  mimetype):
                return func(*args, **kwargs)
            message = "Request must contain {} data".format(mimetype)
            data = json.dumps({"message": message})
            return Response(data, 415, mimetype="application/json")
        return wrapper
    return decorator

def nocache(view):
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers['Last-Modified'] = datetime.now()
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response
        
    return update_wrapper(no_cache, view)
