import json
import requests
import numpy as np
from scipy.optimize import minimize
import random

######### DO NOT CHANGE ANYTHING IN THIS FILE ##################
API_ENDPOINT = 'http://10.4.21.147'
PORT = 3000
MAX_DEG = 11

# Use very wisely
SECRET_KEY = "6Fwiv8RlF6adiEr2CoqQZFRxGhg8XnQszfULpkQCGEXTzcjuNB"

# functions that you can call


def get_errors(id, vector):
    """
    returns python array of length 2 
    (train error and validation error)
    """
    for i in vector:
        assert -10 <= abs(i) <= 10
    assert len(vector) == MAX_DEG

    return json.loads(send_request(id, vector, 'geterrors'))


def submit(id, vector):
    """
    used to make official submission of your weight vector
    returns string "successfully submitted" if properly submitted.
    """
    for i in vector:
        assert -10 <= abs(i) <= 10
    assert len(vector) == MAX_DEG
    return send_request(id, vector, 'submit')

# utility functions


def urljoin(root, port, path=''):
    root = root + ':' + str(port)
    if path:
        root = '/'.join([root.rstrip('/'), path.rstrip('/')])
    return root


def send_request(id, vector, path):
    api = urljoin(API_ENDPOINT, PORT, path)
    vector = json.dumps(vector)
    response = requests.post(api, data={'id': id, 'vector': vector}).text
    if "reported" in response:
        print(response)
        exit()

    return response


overfit_array = [0.0, 0.1240317450077846, -6.211941063144333, 0.04933903144709126, 0.03810848157715883, 8.132366097133624e-05, -
                 6.018769160916912e-05, -1.251585565299179e-07, 3.484096383229681e-08, 4.1614924993407104e-11, -6.732420176902565e-12]


def func(weights):
    for w in weights:
        if abs(w) > 10:
            return np.inf
    errors = get_errors(SECRET_KEY, list(weights))
    return (0.8 * errors[0]) + (0.1 * errors[1])


init_array = [-9.24592152e-03,  2.02994594e-02, -1.42211175e-02,  1.94909772e-02,
              6.78329622e-03,  8.54781073e-05, -1.33290569e-05, -7.47067330e-08,
              8.18861099e-09,  1.99195515e-11, -1.60844989e-12]


if __name__ == "__main__":
    """
    Run simulated annealing
    """

    array = np.array(init_array)

    bounds = np.full((11, 2), -10)
    bounds[:, 1] = np.full((11,), 10)

    max_calls = 1000

    result = minimize(func, array, method='Nelder-Mead',
                      options={'maxfev': max_calls})

    print(result)

    # error for overfit array:
    # [79569.63536912124, 3625792.834235452]
