import json
import requests
import numpy as np

######### DO NOT CHANGE ANYTHING IN THIS FILE ##################
API_ENDPOINT = 'http://10.4.21.147'
PORT = 3000
MAX_DEG = 11

# Use very wisely
SECRET_KEY = "6Fwiv8RlF6adiEr2CoqQZFRxGhg8XnQszfULpkQCGEXTzcjuNB"

#### functions that you can call
def get_errors(id, vector):
    """
    returns python array of length 2 
    (train error and validation error)
    """
    for i in vector: assert -10<=abs(i)<=10
    assert len(vector) == MAX_DEG

    return json.loads(send_request(id, vector, 'geterrors'))

def submit(id, vector):
    """
    used to make official submission of your weight vector
    returns string "successfully submitted" if properly submitted.
    """
    for i in vector: assert -10<=abs(i)<=10
    assert len(vector) == MAX_DEG
    return send_request(id, vector, 'submit')

#### utility functions
def urljoin(root, port, path=''):
    root = root + ':' + str(port)
    if path: root = '/'.join([root.rstrip('/'), path.rstrip('/')])
    return root

def send_request(id, vector, path):
    api = urljoin(API_ENDPOINT, PORT, path)
    vector = json.dumps(vector)
    response = requests.post(api, data={'id':id, 'vector':vector}).text
    if "reported" in response:
        print(response)
        exit()

    return response

overfit_array = [0.0, 0.1240317450077846, -6.211941063144333, 0.04933903144709126, 0.03810848157715883, 8.132366097133624e-05, -6.018769160916912e-05, -1.251585565299179e-07, 3.484096383229681e-08, 4.1614924993407104e-11, -6.732420176902565e-12]

# Parameters for gradient descent
alpha = 10**(-12)

# For computing derivative
delta = 10**(-12)

def compute_derivative(weights_array,error, dimension):
    peturbed = np.copy(weights_array)
    peturbed[dimension] += delta

    errors_peturbed = get_errors(SECRET_KEY,list(peturbed))
    print(errors_peturbed)
    return (errors_peturbed[0] - error)/delta

    



if __name__ == "__main__":
    """
    Run gradient descent
    """

    # err = get_errors(SECRET_KEY, overfit_array)
    # print(err)
    # assert len(err) == 2

    # submit_status = submit(SECRET_KEY, overfit_array)
    # print(submit_status)
    # assert "submitted" in submit_status

    num_iterations = 1
    array = np.array(overfit_array)

    # error for overfit array:
    # [79569.63536912124, 3625792.834235452]

    current_error = 79569.63536912124

    for iter in range(num_iterations):
        gradients = np.zeros((11,))
        for dim in range(11):
            gradients[dim] = compute_derivative(array, current_error, dim)

        array = np.subtract(array , (np.multiply(alpha , gradients)))
        print("Iteration " + str(iter))
        print("Array:")
        print(array)

        print("Gradients:")
        print(gradients)
        
        error_returned = get_errors(SECRET_KEY, list(array))
        print("errors")
        print(error_returned)
        print("")
        
        current_error = error_returned[0]
        
            
            
    
