#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void matmul(const float* x, const float* w, float* y, const size_t m, const size_t n, const size_t k){
    /*
    x: m, n
    w: n, k
    y: m, k
    */
    for(size_t m_idx = 0; m_idx < m; m_idx++){
        for(size_t k_idx = 0; k_idx < k; k_idx++){
            float accum = 0.0f; 
            for(size_t n_idx = 0; n_idx < n; n_idx++){
                float x_val = 0.0f; 
                x_val = x[m_idx * n + n_idx]; 
                float w_val = w[n_idx * k + k_idx]; 
                accum += x_val * w_val; 
            }
            y[m_idx * k + k_idx] = accum; 
        }
    }
}

void compute_crossentropy_grad(const float* out, const unsigned char * label, float* grad, size_t batch, size_t classes){
    for(size_t batch_idx = 0; batch_idx < batch; batch_idx++){
        float accum = 0.0f; 
        for(size_t class_idx = 0; class_idx < classes; class_idx++){
            float out_val = out[batch_idx * classes + class_idx]; 
            accum += std::exp(out_val); 
        }
        float grad_val = 0.0f; 
        unsigned char label_val = label[batch_idx]; 
        for(size_t class_idx = 0; class_idx < classes; class_idx++){
            float out_val = out[batch_idx * classes + class_idx]; 
            grad_val = std::exp(out_val) / accum; 
            if(class_idx == label_val){
                grad_val -= 1; 
            }
            // Reduce Mean
            grad_val /= batch; 
            grad[batch_idx * classes + class_idx] = grad_val; 
        }
    }
}

void transpose(const float* in, float* out, size_t m, size_t n){
    for(size_t m_idx = 0; m_idx < m; m_idx++){
        for(size_t n_idx = 0; n_idx < n; n_idx++){
            out[n_idx * m + m_idx] = in[m_idx * n + n_idx]; 
        }
    }
}


void update(const float* grad, float* w, float lr, size_t elem_cnt){
    for(size_t i = 0; i < elem_cnt; i++){
        w[i] -= lr * grad[i]; 
    }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    size_t sample_num = m; 
    size_t iter_num = sample_num / batch; 
    float* matmul_result; 
    matmul_result = (float*)malloc(batch*k*sizeof(float)); 
    float* cross_entropy_grad; 
    cross_entropy_grad = (float*)malloc(batch*k*sizeof(float)); 
    float* theta_grad; 
    theta_grad = (float*)malloc(n*k*sizeof(float)); 
    float* transposed_x; 
    transposed_x = (float*)malloc(n*batch*sizeof(float)); 

    for(size_t iter = 0; iter < iter_num; iter++){
        const float* iter_x = X + iter * batch * n; 
        const unsigned char * iter_y = y + iter * batch; 
        matmul(iter_x, theta, matmul_result, batch, n, k); 
        compute_crossentropy_grad(matmul_result, iter_y, cross_entropy_grad, batch, k); 
        transpose(iter_x, transposed_x, batch, n); 
        matmul(transposed_x, cross_entropy_grad, theta_grad, n, batch, k); 
        update(theta_grad, theta, lr, n * k); 
    }

    free(matmul_result); 
    free(cross_entropy_grad); 
    free(theta_grad); 
    free(transposed_x); 
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
