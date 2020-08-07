void mat_interp_extrap(double* x, double* xp, double* yp, double* out, int N, int M){
   for(int i = 0; i < N; ++i){
      int j =0;
      int offset = M*(i);
      while(x[i] > xp[j + offset] && j < M) ++j;
      if(j == 0){
         double left_slope = (yp[1 + offset] - yp[0 + offset])/(xp[1+offset] - xp[0 + offset]);
         out[i] = yp[0 + offset] + left_slope*(x[i] - xp[0 + offset]);
      } else if(j == M){
         double right_slope = (yp[M-1 + offset] - yp[M-2 + offset])/(xp[M-1 + offset] - xp[M-2 + offset]);
         out[i] = yp[M-1 + offset] + right_slope*(x[i] - xp[M-1 + offset]);
      } else{
         out[i] = (yp[j + offset] - yp[j-1 + offset])*(x[i] - xp[j-1 + offset])/(xp[j + offset] - xp[j-1 + offset]) +  yp[j-1 + offset];
      }

   }
}

static PyObject* interp_extrap_2d(PyObject* self, PyObject* args){
                PyObject* arr;
   PyObject* xp;
   PyObject* yp;
   PyObject* out;
   if(!PyArg_ParseTuple(args,"OOOO", &arr, &xp, &yp, &out)) return NULL;
   
   if( PyArray_Check(arr) == 0 || PyArray_Check(xp) == 0 ||
      PyArray_Check(yp) == 0 ||  PyArray_Check(out) == 0 ) return NULL;

   if(PyArray_NDIM(arr) > 1 || PyArray_NDIM(out) > 1 ) return NULL;
   if(PyArray_NDIM(xp) != 2 || PyArray_NDIM(yp) != 2) return NULL;
                if(PyArray_TYPE(arr) != NPY_DOUBLE || PyArray_TYPE(xp) != NPY_DOUBLE ||
         PyArray_TYPE(yp) != NPY_DOUBLE || PyArray_TYPE(out) != NPY_DOUBLE ) return NULL;
                int N = static_cast<int>(PyArray_DIM(arr,0));
   if(N != static_cast<int>(PyArray_DIM(xp,0)) || N != static_cast<int>(PyArray_DIM(yp,0)) ) return NULL;  
   int M = static_cast<int>(PyArray_DIM(xp,1));
   if(M != static_cast<int>(PyArray_DIM(yp,1))) return NULL;
   if( !(PyArray_FLAGS(xp) & NPY_ARRAY_C_CONTIGUOUS > 0) ||  !(PyArray_FLAGS(yp) & NPY_ARRAY_C_CONTIGUOUS > 0) ) return NULL;
   double* a = static_cast<double*>(PyArray_DATA(arr));
   double* x = static_cast<double*>(PyArray_DATA(xp));
   double* y = static_cast<double*>(PyArray_DATA(yp));
   double* o = static_cast<double*>(PyArray_DATA(out));

   mat_interp_extrap(a,x, y,o, N, M);
   Py_INCREF(Py_None);
                return Py_None;
}
