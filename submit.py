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

array = [0.0, 0.1240317450077846, -6.211941063144333, 0.04933903144709126, 0.03810848157715883, 8.132366097133624e-05, -
         6.018769160916912e-05, -1.251585565299179e-07, 3.484096383229681e-08, 4.1614924993407104e-11, -6.732420176902565e-12]

array = [0.00000000e+00,  8.55768256e-02, - 4.12295555e+00,  7.60908269e-02,
         6.73041279e-02,  5.62206308e-05, - 4.64482902e-05, - 7.83558218e-08,
         5.90583072e-08,  3.37840409e-11, - 3.66897271e-12]

array = [-0.00000000e+00, - 1.49833740e-03, - 4.66246801e+00,  1.93319835e-02,
         6.54281957e-04,  5.54479196e-05, - 5.62424063e-06,  3.23837066e-08,
         6.36016183e-09, - 2.65942196e-11, - 1.79999624e-12]

array = [-0.0, 6.844301926688699e-05, 0.9630117350983917, 0.013852937474123788, 0.0023741738877534322, -5.1073305429743666e-08,
         4.904822173952492e-08, -1.2302982569315922e-10, -2.1469308424938268e-12, 4.079572212107323e-15, -3.1749353474217e-13]

array = [-0.0, 5.74933492923572e-06, -0.6994155667063526, -0.000921635309754811, -1.9105315348542117e-05, 1.3127840531868187e-05, -
         2.201551273088197e-08, 8.722847213949144e-11, -1.038091680104222e-11, 2.7710469539170183e-16, 8.211288745780781e-15]

array = [-0.0, -4.361722283546157e-10, -0.0005055241554558914, -2.6487960011396005e-06, -9.650032699704849e-05, 2.7156795149820535e-05,
         9.820802314379616e-09, 6.904133164193354e-11, 4.4053253977527154e-12, 2.0492712039198205e-17, -3.0152644974129365e-18]

array = [0.0, -3.918260396950646e-12, -0.0009953617625881616, 6.612490247771486e-10, -2.545502471174068e-09, 2.5095292089993095e-05,
         2.1953845136845325e-11, -2.4881000307027943e-14, -5.010660833900114e-15, 3.446300307639425e-20, 2.7132631323334086e-16]

array = [-0.0, -1.8631364185591987e-13, 1.0057486357736274e-05, 9.14625116608227e-12, -2.207074847192279e-11, 2.5369947492675722e-05, -
         1.2738658982959776e-12, -5.3060032179407856e-14, 7.937596455173086e-18, -1.0240693055465825e-21, -1.4679208704900088e-16]


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
