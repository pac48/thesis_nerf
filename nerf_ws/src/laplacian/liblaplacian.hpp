namespace internal {
    void forward(float *X_ptr, int x_dims, int y_dims, int z_dims, const int *const indexes_ptr, int indexes_len);
    void backward(const float * const X_ptr, int x_dims, int y_dims, int z_dims, const int *const indexes_ptr, int indexes_len, const float * const dL_dout_ptr, float *dX_dV_ptr, float *dX_dC_ptr);
}
