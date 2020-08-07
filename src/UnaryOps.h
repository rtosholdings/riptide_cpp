

PyObject *
BasicMathOneInput(PyObject *self, PyObject *args);

PyObject *
BasicMathOneInputFromNumber(PyArrayObject* inObject1, INT64 funcNumber, BOOL inplace);

PyObject *
BasicMathUnaryOp(PyObject *self, PyObject *args, PyObject *kwargs);

//--------------------------------------------------------------------
// multithreaded struct used for calling unary op codes
struct UNARY_CALLBACK {
   union {
      UNARY_FUNC pUnaryCallback;
      UNARY_FUNC_STRIDED pUnaryCallbackStrided;
   };

   char* pDataIn;
   char* pDataOut;

   INT64 itemSizeIn;
   INT64 itemSizeOut;
};
