namespace internal {
    void forward(float *X_ptr, int x_dims, int y_dims);
    void backward(const float * const X_ptr, const float * const dL_dout_ptr, float *dX_dV_ptr, float *dX_dC_ptr, int x_dims, int y_dims);
}
