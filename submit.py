import json
import requests
import numpy as np

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


array = [2.80207813e-01,  1.94891526e+00, -7.88240619e-02,  3.87876981e-02,
         7.33138501e-03,  4.58229907e-05, -1.35724700e-05, -5.08008878e-08,
         7.96103182e-09,  1.53394012e-11, -1.50340434e-12]


if __name__ == "__main__":
    """
    Replace "test" with your secret ID and just run this file 
    to verify that the server is working for your ID.
    """

    err = get_errors(SECRET_KEY, array)
    print(err)
    assert len(err) == 2

    submit_status = submit(SECRET_KEY, array)
    print(submit_status)
    assert "submitted" in submit_status
