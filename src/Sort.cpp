#include "RipTide.h"
#include "ndarray.h"
#include "Sort.h"
#include "MultiKey.h"
#include "MathWorker.h"
#include "Recycler.h"

#define LOGGING(...)
//#define LOGGING printf

#define PLOGGING(...)
//#define PLOGGING printf


//#define TSWAP(A, i, j) { T tmp = A[i]; A[i] = A[j]; A[j] = tmp; }
//
//template <typename T>
//void sort(T* A, INT64 left, INT64 right) {
//   if (right > left) {
//      // Choose outermost elements as pivots
//      if (A[left] > A[right]) TSWAP(A, left, right);
//
//      T p = A[left], q = A[right];
//
//      // Partition A according to invariant below
//      INT64 l = left + 1, g = right - 1, k = l;
//      while (k <= g) {
//         if (A[k] < p) {
//            TSWAP(A, k, l);
//            ++l;
//         }
//         else if (A[k] >= q) {
//            while (A[g] > q && k < g) --g;
//            TSWAP(A, k, g);
//            --g;
//            if (A[k] < p) {
//               TSWAP(A, k, l);
//               ++l;
//            }
//         }
//         ++k;
//      }
//      --l; ++g;
//
//      // Swap pivots to final place
//      TSWAP(A, left, l); TSWAP(A, right, g);
//
//      // Recursively sort partitions
//      sort(A, left, l - 1);
//      sort(A, l + 1, g - 1);
//      sort(A, g + 1, right);
//   }
//}



#define NPY_ENOMEM 1
#define NPY_ECOMP 2

static __inline int npy_get_msb(UINT64 unum)
{
   int depth_limit = 0;
   while (unum >>= 1) {
      depth_limit++;
   }
   return depth_limit;
}

#define SMALL_MERGESORT 16

#define PYA_QS_STACK (NPY_BITSOF_INTP * 2)  // 128
#define SMALL_QUICKSORT 15

#define T_LT(_X_,_Y_) (_X_ < _Y_)

// Anything compared to a nan will return 0
#define FLOAT_LT(_X_,_Y_) (_X_ < _Y_ || (_Y_ != _Y_ && _X_ == _X_))  


__inline bool COMPARE_LT(float X, float Y)   { return (X < Y || (Y != Y && X == X)); }
__inline bool COMPARE_LT(double X, double Y) { return (X < Y || (Y != Y && X == X)); }
__inline bool COMPARE_LT(long double X, long double Y) { return (X < Y || (Y != Y && X == X)); }
__inline bool COMPARE_LT(INT32 X, INT32 Y)   { return (X < Y); }
__inline bool COMPARE_LT(INT64 X, INT64 Y)   { return (X < Y); }
__inline bool COMPARE_LT(UINT32 X, UINT32 Y) { return (X < Y); }
__inline bool COMPARE_LT(UINT64 X, UINT64 Y) { return (X < Y); }
__inline bool COMPARE_LT(INT8 X, INT8 Y) { return (X < Y); }
__inline bool COMPARE_LT(INT16 X, INT16 Y) { return (X < Y); }
__inline bool COMPARE_LT(UINT8 X, UINT8 Y) { return (X < Y); }
__inline bool COMPARE_LT(UINT16 X, UINT16 Y) { return (X < Y); }

NPY_INLINE static int
STRING_LT(const char *s1, const char *s2, size_t len)
{
   const unsigned char *c1 = (unsigned char *)s1;
   const unsigned char *c2 = (unsigned char *)s2;
   size_t i;

   for (i = 0; i < len; ++i) {
      if (c1[i] != c2[i]) {
         return c1[i] < c2[i];
      }
   }
   return 0;
}

//---------------------------------
// Assumes Py_UCS4
// Assumes int is 32bits
NPY_INLINE static int
UNICODE_LT(const char *s1, const char *s2, size_t len)
{
   const unsigned int *c1 = (unsigned int *)s1;
   const unsigned int *c2 = (unsigned int *)s2;
   size_t i;

   size_t lenunicode = len / 4;

   for (i = 0; i < lenunicode; ++i) {
      if (c1[i] != c2[i]) {
         return c1[i] < c2[i];
      }
   }
   return 0;
}


NPY_INLINE static int
VOID_LT(const char *s1, const char *s2, size_t len)
{
   const unsigned char *c1 = (unsigned char *)s1;
   const unsigned char *c2 = (unsigned char *)s2;

   switch (len) {
   case 1:
      if (*c1 != *c2) {
         return *c1 < *c2;
      }
      return 0;
   case 2:
      if (*(UINT16*)c1 != *(UINT16*)c2) {
         return *(UINT16*)c1 < *(UINT16*)c2;
      }
      return 0;
   case 3:
      if (*(UINT16*)c1 != *(UINT16*)c2) {
         return *(UINT16*)c1 < *(UINT16*)c2;
      }
      c1 += 2;
      c2 += 2;
      if (*c1 != *c2) {
         return *c1 < *c2;
      }
      return 0;
   case 4:
      if (*(UINT32*)c1 != *(UINT32*)c2) {
         return *(UINT32*)c1 < *(UINT32*)c2;
      }
      return 0;
   case 5:
      if (*(UINT32*)c1 != *(UINT32*)c2) {
         return *(UINT32*)c1 < *(UINT32*)c2;
      }
      c1 += 4;
      c2 += 4;
      if (*c1 != *c2) {
         return *c1 < *c2;
      }
      return 0;
   case 6:
      if (*(UINT32*)c1 != *(UINT32*)c2) {
         return *(UINT32*)c1 < *(UINT32*)c2;
      }
      c1 += 4;
      c2 += 4;
      if (*(UINT16*)c1 != *(UINT16*)c2) {
         return *(UINT16*)c1 < *(UINT16*)c2;
      }
      return 0;
   case 7:
      if (*(UINT32*)c1 != *(UINT32*)c2) {
         return *(UINT32*)c1 < *(UINT32*)c2;
      }
      c1 += 4;
      c2 += 4;
      if (*(UINT16*)c1 != *(UINT16*)c2) {
         return *(UINT16*)c1 < *(UINT16*)c2;
      }
      c1 += 2;
      c2 += 2;
      if (*c1 != *c2) {
         return *c1 < *c2;
      }
      return 0;
   case 8:
      if (*(UINT64*)c1 != *(UINT64*)c2) {
         return *(UINT64*)c1 < *(UINT64*)c2;
      }
      return 0;
   default:
   {
      // compare 8 bytes at a time
      while (len > 8) {
         if (*(UINT64*)c1 != *(UINT64*)c2) {
            return *(UINT64*)c1 < *(UINT64*)c2;
         }
         c1 += 8;
         c2 += 8;
         len -= 8;
      }
      switch (len) {
      case 1:
         if (*c1 != *c2) {
            return *c1 < *c2;
         }
         return 0;
      case 2:
         if (*(UINT16*)c1 != *(UINT16*)c2) {
            return *(UINT16*)c1 < *(UINT16*)c2;
         }
         return 0;
      case 3:
         if (*(UINT16*)c1 != *(UINT16*)c2) {
            return *(UINT16*)c1 < *(UINT16*)c2;
         }
         c1 += 2;
         c2 += 2;
         if (*c1 != *c2) {
            return *c1 < *c2;
         }
         return 0;
      case 4:
         if (*(UINT32*)c1 != *(UINT32*)c2) {
            return *(UINT32*)c1 < *(UINT32*)c2;
         }
         return 0;
      case 5:
         if (*(UINT32*)c1 != *(UINT32*)c2) {
            return *(UINT32*)c1 < *(UINT32*)c2;
         }
         c1 += 4;
         c2 += 4;
         if (*c1 != *c2) {
            return *c1 < *c2;
         }
         return 0;
      case 6:
         if (*(UINT32*)c1 != *(UINT32*)c2) {
            return *(UINT32*)c1 < *(UINT32*)c2;
         }
         c1 += 4;
         c2 += 4;
         if (*(UINT16*)c1 != *(UINT16*)c2) {
            return *(UINT16*)c1 < *(UINT16*)c2;
         }
         return 0;
      case 7:
         if (*(UINT32*)c1 != *(UINT32*)c2) {
            return *(UINT32*)c1 < *(UINT32*)c2;
         }
         c1 += 4;
         c2 += 4;
         if (*(UINT16*)c1 != *(UINT16*)c2) {
            return *(UINT16*)c1 < *(UINT16*)c2;
         }
         c1 += 2;
         c2 += 2;
         if (*c1 != *c2) {
            return *c1 < *c2;
         }
         return 0;
      case 8:
         if (*(UINT64*)c1 != *(UINT64*)c2) {
            return *(UINT64*)c1 < *(UINT64*)c2;
         }
         return 0;
      default:
         return 0;
      }
   }
   }
   return 0;
}



NPY_INLINE static int
BINARY_LT(const char *s1, const char *s2, size_t len)
{
   const unsigned char *c1 = (unsigned char *)s1;
   const unsigned char *c2 = (unsigned char *)s2;

   switch (len) {
   case 1:
      if (*c1 != *c2) {
         return 1;
      }
      return 0;
   case 2:
      if (*(UINT16*)c1 != *(UINT16*)c2) {
         return 1;
      }
      return 0;
   case 3:
      if (*(UINT16*)c1 != *(UINT16*)c2) {
         return 1;
      }
      c1 += 2;
      c2 += 2;
      if (*c1 != *c2) {
         return 1;
      }
      return 0;
   case 4:
      if (*(UINT32*)c1 != *(UINT32*)c2) {
         return 1;
      }
      return 0;
   case 5:
      if (*(UINT32*)c1 != *(UINT32*)c2) {
         return 1;
      }
      c1 += 4;
      c2 += 4;
      if (*c1 != *c2) {
         return 1;
      }
      return 0;
   case 6:
      if (*(UINT32*)c1 != *(UINT32*)c2) {
         return 1;
      }
      c1 += 4;
      c2 += 4;
      if (*(UINT16*)c1 != *(UINT16*)c2) {
         return 1;
      }
      return 0;
   case 7:
      if (*(UINT32*)c1 != *(UINT32*)c2) {
         return 1;
      }
      c1 += 4;
      c2 += 4;
      if (*(UINT16*)c1 != *(UINT16*)c2) {
         return 1;
      }
      c1 += 2;
      c2 += 2;
      if (*c1 != *c2) {
         return *c1 < *c2;
      }
      return 0;
   case 8:
      if (*(UINT64*)c1 != *(UINT64*)c2) {
         return 1;
      }
      return 0;
   default:
   {
      while (len > 8) {
         if (*(UINT64*)c1 != *(UINT64*)c2) {
            return 1;
         }
         c1 += 8;
         c2 += 8;
         len -= 8;
      }
      switch (len) {
      case 1:
         if (*c1 != *c2) {
            return 1;
         }
         return 0;
      case 2:
         if (*(UINT16*)c1 != *(UINT16*)c2) {
            return 1;
         }
         return 0;
      case 3:
         if (*(UINT16*)c1 != *(UINT16*)c2) {
            return 1;
         }
         c1 += 2;
         c2 += 2;
         if (*c1 != *c2) {
            return 1;
         }
         return 0;
      case 4:
         if (*(UINT32*)c1 != *(UINT32*)c2) {
            return 1;
         }
         return 0;
      case 5:
         if (*(UINT32*)c1 != *(UINT32*)c2) {
            return 1;
         }
         c1 += 4;
         c2 += 4;
         if (*c1 != *c2) {
            return 1;
         }
         return 0;
      case 6:
         if (*(UINT32*)c1 != *(UINT32*)c2) {
            return 1;
         }
         c1 += 4;
         c2 += 4;
         if (*(UINT16*)c1 != *(UINT16*)c2) {
            return 1;
         }
         return 0;
      case 7:
         if (*(UINT32*)c1 != *(UINT32*)c2) {
            return 1;
         }
         c1 += 4;
         c2 += 4;
         if (*(UINT16*)c1 != *(UINT16*)c2) {
            return 1;
         }
         c1 += 2;
         c2 += 2;
         if (*c1 != *c2) {
            return 1;
         }
         return 0;
      case 8:
         if (*(UINT64*)c1 != *(UINT64*)c2) {
            return 1;
         }
         return 0;
      default:
         return 0;
      }
   }
   }
   return 0;
}


// see: https://github.com/certik/python-3.3/blob/master/Objects/unicodeobject.c
// we will assume
//static int
//unicode_compare(PyObject *str1, PyObject *str2)
//{
//   int kind1, kind2;
//   void *data1, *data2;
//   Py_ssize_t len1, len2, i;
//
//   kind1 = PyUnicode_KIND(str1);
//   kind2 = PyUnicode_KIND(str2);
//   data1 = PyUnicode_DATA(str1);
//   data2 = PyUnicode_DATA(str2);
//   len1 = PyUnicode_GET_LENGTH(str1);
//   len2 = PyUnicode_GET_LENGTH(str2);
//
//   for (i = 0; i < len1 && i < len2; ++i) {
//      Py_UCS4 c1, c2;
//      c1 = PyUnicode_READ(kind1, data1, i);
//      c2 = PyUnicode_READ(kind2, data2, i);
//
//      if (c1 != c2)
//         return (c1 < c2) ? -1 : 1;
//   }
//
//   return (len1 < len2) ? -1 : (len1 != len2);
//}



//NPY_INLINE static int
//FLOAT_LT(npy_float a, npy_float b)
//{
//   return a < b || (b != b && a == a);
//}
//
//
//NPY_INLINE static int
//DOUBLE_LT(npy_double a, npy_double b)
//{
//   return a < b || (b != b && a == a);
//}


#define INT32_LT(_X_,_Y_) (_X_ < _Y_)
#define INT32_SWAP(_X_,_Y_) { INT32 temp; temp=_X_; _X_=_Y_; _Y_=temp;}
#define INTP_SWAP(_X_,_Y_) { auto temp=_X_; _X_=_Y_; _Y_=temp;}

//#define T_SWAP(_X_, _Y_) { auto temp;  temp = _X_; _X_ = _Y_; _Y_ = temp; }
#define T_SWAP(_X_, _Y_)  std::swap(_X_,_Y_); 


////--------------------------------------------------------------------------------------
//static int
//orig_quicksort_(INT32 *start, INT64 num)
//{
//   INT32 vp;
//   INT32 *pl = start;
//   INT32 *pr = pl + num - 1;
//   INT32 *stack[PYA_QS_STACK];
//   INT32 **sptr = stack;
//   INT32 *pm, *pi, *pj, *pk;
//   int depth[PYA_QS_STACK];
//   int * psdepth = depth;
//   int cdepth = npy_get_msb(num) * 2;
//
//   for (;;) {
//      if (NPY_UNLIKELY(cdepth < 0)) {
//         orig_heapsort_(pl, pr - pl + 1);
//         goto stack_pop;
//      }
//      while ((pr - pl) > SMALL_QUICKSORT) {
//         /* quicksort partition */
//         pm = pl + ((pr - pl) >> 1);
//         if (INT32_LT(*pm, *pl)) INT32_SWAP(*pm, *pl);
//         if (INT32_LT(*pr, *pm)) INT32_SWAP(*pr, *pm);
//         if (INT32_LT(*pm, *pl)) INT32_SWAP(*pm, *pl);
//         vp = *pm;
//         pi = pl;
//         pj = pr - 1;
//         INT32_SWAP(*pm, *pj);
//         for (;;) {
//            do ++pi; while (INT32_LT(*pi, vp));
//            do --pj; while (INT32_LT(vp, *pj));
//            if (pi >= pj) {
//               break;
//            }
//            INT32_SWAP(*pi, *pj);
//         }
//         pk = pr - 1;
//         INT32_SWAP(*pi, *pk);
//         /* push largest partition on stack */
//         if (pi - pl < pr - pi) {
//            *sptr++ = pi + 1;
//            *sptr++ = pr;
//            pr = pi - 1;
//         }
//         else {
//            *sptr++ = pl;
//            *sptr++ = pi - 1;
//            pl = pi + 1;
//         }
//         *psdepth++ = --cdepth;
//      }
//
//      /* insertion sort */
//      for (pi = pl + 1; pi <= pr; ++pi) {
//         vp = *pi;
//         pj = pi;
//         pk = pi - 1;
//         while (pj > pl && INT32_LT(vp, *pk)) {
//            *pj-- = *pk--;
//         }
//         *pj = vp;
//      }
//   stack_pop:
//      if (sptr == stack) {
//         break;
//      }
//      pr = *(--sptr);
//      pl = *(--sptr);
//      cdepth = *(--psdepth);
//   }
//
//   return 0;
//}
//
//
//static int
//orig_aquicksort_(INT32 *vv, INT64* tosort, INT64 num)
//{
//   INT32 *v = vv;
//   INT32 vp;
//   INT64 *pl = tosort;
//   INT64 *pr = tosort + num - 1;
//   INT64 *stack[PYA_QS_STACK];
//   INT64 **sptr = stack;
//   INT64 *pm, *pi, *pj, *pk, vi;
//   int depth[PYA_QS_STACK];
//   int * psdepth = depth;
//   int cdepth = npy_get_msb(num) * 2;
//
//   for (;;) {
//      if (NPY_UNLIKELY(cdepth < 0)) {
//         orig_aheapsort_(vv, pl, pr - pl + 1);
//         goto stack_pop;
//      }
//
//      while ((pr - pl) > SMALL_QUICKSORT) {
//         /* quicksort partition */
//         pm = pl + ((pr - pl) >> 1);
//         if (INT32_LT(v[*pm], v[*pl])) INTP_SWAP(*pm, *pl);
//         if (INT32_LT(v[*pr], v[*pm])) INTP_SWAP(*pr, *pm);
//         if (INT32_LT(v[*pm], v[*pl])) INTP_SWAP(*pm, *pl);
//         vp = v[*pm];
//         pi = pl;
//         pj = pr - 1;
//         INTP_SWAP(*pm, *pj);
//         for (;;) {
//            do ++pi; while (INT32_LT(v[*pi], vp));
//            do --pj; while (INT32_LT(vp, v[*pj]));
//            if (pi >= pj) {
//               break;
//            }
//            INTP_SWAP(*pi, *pj);
//         }
//         pk = pr - 1;
//         INTP_SWAP(*pi, *pk);
//         /* push largest partition on stack */
//         if (pi - pl < pr - pi) {
//            *sptr++ = pi + 1;
//            *sptr++ = pr;
//            pr = pi - 1;
//         }
//         else {
//            *sptr++ = pl;
//            *sptr++ = pi - 1;
//            pl = pi + 1;
//         }
//         *psdepth++ = --cdepth;
//      }
//
//      /* insertion sort */
//      for (pi = pl + 1; pi <= pr; ++pi) {
//         vi = *pi;
//         vp = v[vi];
//         pj = pi;
//         pk = pi - 1;
//         while (pj > pl && INT32_LT(vp, v[*pk])) {
//            *pj-- = *pk--;
//         }
//         *pj = vi;
//      }
//   stack_pop:
//      if (sptr == stack) {
//         break;
//      }
//      pr = *(--sptr);
//      pl = *(--sptr);
//      cdepth = *(--psdepth);
//   }
//
//   return 0;
//}
//
//
//
//
//
//
//

//-----------------------------------------------------------------------------------------------
template <typename T>
/*static*/ int
heapsort_(T *start, INT64 n)
{
   T     tmp, *a;
   INT64 i, j, l;

   /* The array needs to be offset by one for heapsort indexing */
   a = (T *)start - 1;

   for (l = n >> 1; l > 0; --l) {
      tmp = a[l];
      for (i = l, j = l << 1; j <= n;) {
         if (j < n && T_LT(a[j], a[j + 1])) {
            j += 1;
         }
         if (T_LT(tmp, a[j])) {
            a[i] = a[j];
            i = j;
            j += j;
         }
         else {
            break;
         }
      }
      a[i] = tmp;
   }

   for (; n > 1;) {
      tmp = a[n];
      a[n] = a[1];
      n -= 1;
      for (i = 1, j = 2; j <= n;) {
         if (j < n && T_LT(a[j], a[j + 1])) {
            j++;
         }
         if (T_LT(tmp, a[j])) {
            a[i] = a[j];
            i = j;
            j += j;
         }
         else {
            break;
         }
      }
      a[i] = tmp;
   }

   return 0;
}


//-----------------------------------------------------------------------------------------------
template <typename T, typename UINDEX>
static int
aheapsort_(T *vv, UINDEX *tosort, UINDEX n)
{
   T *v = vv;
   UINDEX *a, i, j, l, tmp;
   /* The arrays need to be offset by one for heapsort indexing */
   a = tosort - 1;

   for (l = n >> 1; l > 0; --l) {
      tmp = a[l];
      for (i = l, j = l << 1; j <= n;) {
         if (j < n && T_LT(v[a[j]], v[a[j + 1]])) {
            j += 1;
         }
         if (T_LT(v[tmp], v[a[j]])) {
            a[i] = a[j];
            i = j;
            j += j;
         }
         else {
            break;
         }
      }
      a[i] = tmp;
   }

   for (; n > 1;) {
      tmp = a[n];
      a[n] = a[1];
      n -= 1;
      for (i = 1, j = 2; j <= n;) {
         if (j < n && T_LT(v[a[j]], v[a[j + 1]])) {
            j++;
         }
         if (T_LT(v[tmp], v[a[j]])) {
            a[i] = a[j];
            i = j;
            j += j;
         }
         else {
            break;
         }
      }
      a[i] = tmp;
   }

   return 0;
}


//-----------------------------------------------------------------------------------------------
template <typename T, typename UINDEX>
static int
aheapsort_float(T *vv, UINDEX *tosort, UINDEX n)
{
   T *v = vv;
   UINDEX *a, i, j, l, tmp;
   /* The arrays need to be offset by one for heapsort indexing */
   a = tosort - 1;

   for (l = n >> 1; l > 0; --l) {
      tmp = a[l];
      for (i = l, j = l << 1; j <= n;) {
         if (j < n && FLOAT_LT(v[a[j]], v[a[j + 1]])) {
            j += 1;
         }
         if (FLOAT_LT(v[tmp], v[a[j]])) {
            a[i] = a[j];
            i = j;
            j += j;
         }
         else {
            break;
         }
      }
      a[i] = tmp;
   }

   for (; n > 1;) {
      tmp = a[n];
      a[n] = a[1];
      n -= 1;
      for (i = 1, j = 2; j <= n;) {
         if (j < n && FLOAT_LT(v[a[j]], v[a[j + 1]])) {
            j++;
         }
         if (FLOAT_LT(v[tmp], v[a[j]])) {
            a[i] = a[j];
            i = j;
            j += j;
         }
         else {
            break;
         }
      }
      a[i] = tmp;
   }

   return 0;
}



   //--------------------------------------------------------------------------------------
   // N.B. This function isn't static like the others because it's used in a couple of places in GroupBy.cpp.
   template <typename T>
   /*static*/ int
   quicksort_(T *start, INT64 num)
   {
      T vp;
      T *pl = start;
      T *pr = pl + num - 1;
      T *stack[PYA_QS_STACK];
      T **sptr = stack;
      T *pm, *pi, *pj, *pk;

      int depth[PYA_QS_STACK];
      int * psdepth = depth;
      int cdepth = npy_get_msb(num) * 2;

      for (;;) {
         if (NPY_UNLIKELY(cdepth < 0)) {
            heapsort_<T>(pl, pr - pl + 1);
            goto stack_pop;
         }
         while ((pr - pl) > SMALL_QUICKSORT) {
            /* quicksort partition */
            pm = pl + ((pr - pl) >> 1);
            if (COMPARE_LT(*pm, *pl)) T_SWAP(*pm, *pl);
            if (COMPARE_LT(*pr, *pm)) T_SWAP(*pr, *pm);
            if (COMPARE_LT(*pm, *pl)) T_SWAP(*pm, *pl);
            vp = *pm;
            pi = pl;
            pj = pr - 1;
            T_SWAP(*pm, *pj);
            for (;;) {
               do ++pi; while (COMPARE_LT(*pi, vp));
               do --pj; while (COMPARE_LT(vp, *pj));
               if (pi >= pj) {
                  break;
               }
               T_SWAP(*pi, *pj);
            }
            pk = pr - 1;
            T_SWAP(*pi, *pk);
            /* push largest partition on stack */
            if (pi - pl < pr - pi) {
               *sptr++ = pi + 1;
               *sptr++ = pr;
               pr = pi - 1;
            }
            else {
               *sptr++ = pl;
               *sptr++ = pi - 1;
               pl = pi + 1;
            }
            *psdepth++ = --cdepth;
         }

         /* insertion sort */
         for (pi = pl + 1; pi <= pr; ++pi) {
            vp = *pi;
            pj = pi;
            pk = pi - 1;
            while (pj > pl && COMPARE_LT(vp, *pk)) {
               *pj-- = *pk--;
            }
            *pj = vp;
         }
      stack_pop:
         if (sptr == stack) {
            break;
         }
         pr = *(--sptr);
         pl = *(--sptr);
         cdepth = *(--psdepth);
      }

      return 0;
   }


   //-----------------------------------------------------------------------------------------------
   template <typename T, typename UINDEX>
   static int
   aquicksort_(T *vv, UINDEX* tosort, UINDEX num)
   {
      T *v = vv;
      T vp;
      UINDEX *pl = tosort;
      UINDEX *pr = tosort + num - 1;
      UINDEX *stack[PYA_QS_STACK];
      UINDEX **sptr = stack;
      UINDEX *pm, *pi, *pj, *pk, vi;
      int depth[PYA_QS_STACK];
      int * psdepth = depth;
      int cdepth = npy_get_msb(num) * 2;

      for (;;) {
         if (NPY_UNLIKELY(cdepth < 0)) {
            aheapsort_<T,UINDEX>(vv, pl, (UINDEX)(pr - pl + 1));
            goto stack_pop;
         }

         while ((pr - pl) > SMALL_QUICKSORT) {
            /* quicksort partition */
            pm = pl + ((pr - pl) >> 1);
            if (COMPARE_LT(v[*pm], v[*pl])) INTP_SWAP(*pm, *pl);
            if (COMPARE_LT(v[*pr], v[*pm])) INTP_SWAP(*pr, *pm);
            if (COMPARE_LT(v[*pm], v[*pl])) INTP_SWAP(*pm, *pl);
            vp = v[*pm];
            pi = pl;
            pj = pr - 1;
            INTP_SWAP(*pm, *pj);
            for (;;) {
               do ++pi; while (COMPARE_LT(v[*pi], vp));
               do --pj; while (COMPARE_LT(vp, v[*pj]));
               if (pi >= pj) {
                  break;
               }
               INTP_SWAP(*pi, *pj);
            }
            pk = pr - 1;
            INTP_SWAP(*pi, *pk);
            /* push largest partition on stack */
            if (pi - pl < pr - pi) {
               *sptr++ = pi + 1;
               *sptr++ = pr;
               pr = pi - 1;
            }
            else {
               *sptr++ = pl;
               *sptr++ = pi - 1;
               pl = pi + 1;
            }
            *psdepth++ = --cdepth;
         }

         /* insertion sort */
         for (pi = pl + 1; pi <= pr; ++pi) {
            vi = *pi;
            vp = v[vi];
            pj = pi;
            pk = pi - 1;
            while (pj > pl && COMPARE_LT(vp, v[*pk])) {
               *pj-- = *pk--;
            }
            *pj = vi;
         }
      stack_pop:
         if (sptr == stack) {
            break;
         }
         pr = *(--sptr);
         pl = *(--sptr);
         cdepth = *(--psdepth);
      }

      return 0;
   }


   //-----------------------------------------------------------------------------------------------
   template <typename T, typename UINDEX>
   static int
   aquicksort_float(T *vv, UINDEX* tosort, UINDEX num)
   {
      T *v = vv;
      T vp;
      UINDEX *pl = tosort;
      UINDEX *pr = tosort + num - 1;
      UINDEX *stack[PYA_QS_STACK];
      UINDEX **sptr = stack;
      UINDEX *pm, *pi, *pj, *pk, vi;
      int depth[PYA_QS_STACK];
      int * psdepth = depth;
      int cdepth = npy_get_msb(num) * 2;

      for (;;) {
         if (NPY_UNLIKELY(cdepth < 0)) {
            aheapsort_float<T,UINDEX>(vv, pl, (UINDEX)(pr - pl + 1));
            goto stack_pop;
         }

         while ((pr - pl) > SMALL_QUICKSORT) {
            /* quicksort partition */
            pm = pl + ((pr - pl) >> 1);
            if (FLOAT_LT(v[*pm], v[*pl])) INTP_SWAP(*pm, *pl);
            if (FLOAT_LT(v[*pr], v[*pm])) INTP_SWAP(*pr, *pm);
            if (FLOAT_LT(v[*pm], v[*pl])) INTP_SWAP(*pm, *pl);
            vp = v[*pm];
            pi = pl;
            pj = pr - 1;
            INTP_SWAP(*pm, *pj);
            for (;;) {
               do ++pi; while (FLOAT_LT(v[*pi], vp));
               do --pj; while (FLOAT_LT(vp, v[*pj]));
               if (pi >= pj) {
                  break;
               }
               INTP_SWAP(*pi, *pj);
            }
            pk = pr - 1;
            INTP_SWAP(*pi, *pk);
            /* push largest partition on stack */
            if (pi - pl < pr - pi) {
               *sptr++ = pi + 1;
               *sptr++ = pr;
               pr = pi - 1;
            }
            else {
               *sptr++ = pl;
               *sptr++ = pi - 1;
               pl = pi + 1;
            }
            *psdepth++ = --cdepth;
         }

         /* insertion sort */
         for (pi = pl + 1; pi <= pr; ++pi) {
            vi = *pi;
            vp = v[vi];
            pj = pi;
            pk = pi - 1;
            while (pj > pl && FLOAT_LT(vp, v[*pk])) {
               *pj-- = *pk--;
            }
            *pj = vi;
         }
      stack_pop:
         if (sptr == stack) {
            break;
         }
         pr = *(--sptr);
         pl = *(--sptr);
         cdepth = *(--psdepth);
      }

      return 0;
   }


   //--------------------------------------------------------------------------------------
   template <typename T>
   /*static*/ void 
   mergesort0_(T *pl, T *pr, T *pw)
   {
      T vp, *pi, *pj, *pk, *pm;

      if (pr - pl > SMALL_MERGESORT) {
         /* merge sort */
         pm = pl + ((pr - pl) >> 1);
         mergesort0_(pl, pm, pw);
         mergesort0_(pm, pr, pw);

#ifndef USE_MEMCPY
         memcpy(pw, pl, (pm - pl) * sizeof(T));
#else
         pi = pw;
         pj = pl;
         while (pj < pm) {
            *pi++ = *pj++;
         }
#endif

         pi = pw + (pm - pl);
         pj = pw;
         pk = pl;
         while (pj < pi && pm < pr) {
            if (T_LT(*pm, *pj)) {
               *pk++ = *pm++;
            }
            else {
               *pk++ = *pj++;
            }
         }
#ifdef USE_MEMCPY
         diff = pi - pj;
         if (diff > 0) {
            memcpy(pk, pj, sizeof(T)*diff);
            pk += diff;
            pj += diff;
         }
#else
         while (pj < pi) {
            *pk++ = *pj++;
         }
#endif
      }
      else {
         /* insertion sort */
         for (pi = pl + 1; pi < pr; ++pi) {
            vp = *pi;
            pj = pi;
            pk = pi - 1;
            while (pj > pl && T_LT(vp, *pk)) {
               *pj-- = *pk--;
            }
            *pj = vp;
         }
      }
   }


   //-----------------------------------------------------------------------------------------------
   template <typename T>
   /*static*/ int 
   mergesort_(T *start, INT64 num)
   {
      T *pl, *pr, *pw;

      pl = start;
      pr = pl + num;
      pw = (T*)WORKSPACE_ALLOC((num / 2) * sizeof(T));
      if (pw == NULL) {
         return -1;
      }
      mergesort0_(pl, pr, pw);

      WORKSPACE_FREE(pw);
      return 0;
   }



   //-----------------------------------------------------------------------------------------------
   // binary mergesort with no recursion (num must be power of 2)
   // TJD NOTE: Does not appear much faster
   template <typename T>
   static int 
   mergesort_binary_norecursion(T *start, INT64 num)
   {
      T *pl, *pr, *pw, *pm, *pk, *pi, *pj;

      pl = start;
      pr = pl + num;
      pw = (T*)WORKSPACE_ALLOC((num / 2) * sizeof(T));
      if (pw == NULL) {
         return -1;
      }

      T* pEnd;
      
      pEnd = pr - SMALL_MERGESORT;

      while (pl < pEnd) {

         T* pMiddle;
         T  vp;

         pMiddle = pl + SMALL_MERGESORT;

         /* insertion sort */
         for (pi = pl + 1; pi < pMiddle; ++pi) {
            vp = *pi;
            pj = pi;
            pk = pi - 1;
            while (pj > pl && T_LT(vp, *pk)) {
               *pj-- = *pk--;
            }
            *pj = vp;
         }

         pl += SMALL_MERGESORT;

      }

      //-- reamining for insertion sort
      pEnd = pr;
      while (pl < pEnd) {

         T  *pMiddle;
         T  vp;

         pMiddle = pl + SMALL_MERGESORT;

         /* insertion sort */
         for (pi = pl + 1; pi < pMiddle; ++pi) {
            vp = *pi;
            pj = pi;
            pk = pi - 1;
            while (pj > pl && T_LT(vp, *pk)) {
               *pj-- = *pk--;
            }
            *pj = vp;
         }

         pl += SMALL_MERGESORT;

      }

      // START MERGING ----------------------------------
      //----------------------------------------------------------------
      INT64 startSize = SMALL_MERGESORT;

      pEnd = start + num;
      pl = start;

      while ((pl + startSize) < pEnd) {

         while ((pl + startSize) < pEnd) {

            pr = pl + (2 * startSize);
            pm = pl + ((pr - pl) >> 1);

            // memcpy first half into workspace
            memcpy(pw, pl, (pm - pj) * sizeof(T));

            // merge
            pi = pw + (pm - pl);
            pj = pw;
            pk = pl;

            while (pj < pi && pm < pr) {
               if (T_LT(*pm, *pj)) {
                  *pk++ = *pm++;
               }
               else {
                  *pk++ = *pj++;
               }
            }

            // memcpy 
            while (pj < pi) {
               *pk++ = *pj++;
            }

            // move to next segment to merger
            pl = pr;
         }

         // Now merge again
         pl = start;
         startSize <<= 1;
      }

      //printf("%llu  vs %llu\n", startSize, num);
      //mergesort0_(pl, pr, pw);

      WORKSPACE_FREE(pw);
      return 0;
   }



   //--------------------------------------------------------------------------------------
   template <typename T>
   static void 
   mergesort0_float(T *pl, T *pr, T *pw, T* head)
   {
      T vp, *pi, *pj, *pk, *pm;

      if (pr - pl > SMALL_MERGESORT) {
         /* merge sort */
         pm = pl + ((pr - pl) >> 1);
         mergesort0_float(pl, pm, pw, head);
         mergesort0_float(pm, pr, pw, head);


         pi = pw;
         pj = pl;

// TD NOTE: Jan 2018 -- the memcpy improves float sorting slightly, it does not improve INT copying for some reason...
#ifndef USE_MEMCPY
         INT64 diff = pm - pj;
         memcpy(pi, pj, diff * sizeof(T));
         pi += diff;
         pj += diff;
#else
         while (pj < pm) {
            *pi++ = *pj++;
         }
#endif
         pi = pw + (pm - pl);
         pj = pw;
         pk = pl;
         while (pj < pi && pm < pr) {
            if (FLOAT_LT(*pm, *pj)) {
               *pk++ = *pm++;
            }
            else {
               *pk++ = *pj++;
            }
         }

#ifdef USE_MEMCPY
         diff = pi - pj;
         if (diff > 0) {
            memcpy(pk, pj, sizeof(T)*diff);
            pk += diff;
            pj += diff;
         }
#else
         while (pj < pi) {
            *pk++ = *pj++;
         }
#endif
      }
      else {
         /* insertion sort */
         for (pi = pl + 1; pi < pr; ++pi) {
            vp = *pi;
            pj = pi;
            pk = pi - 1;
            while (pj > pl && FLOAT_LT(vp, *pk)) {
               *pj-- = *pk--;
            }
            *pj = vp;
         }
      }
   }


   //-----------------------------------------------------------------------------------------------
   template <typename T>
   static int 
   mergesort_float(T *start, INT64 num)
   {
      T *pl, *pr, *pw;

      pl = start;
      pr = pl + num;
      pw = (T*)WORKSPACE_ALLOC((num / 2) * sizeof(T));
      if (pw == NULL) {
         return -1;
      }
      mergesort0_float(pl, pr, pw, start);

      WORKSPACE_FREE(pw);
      return 0;
   }



   //-----------------------------------------------------------------------------------------------
  template <typename UINDEX>
  static void
  amergesort0_string(UINDEX *pl, UINDEX *pr, const char *strItem, UINDEX *pw, INT64 strlen)
   {
      const char* vp;
      UINDEX vi, *pi, *pj, *pk, *pm;

      if (pr - pl > SMALL_MERGESORT) {
         /* merge sort */
         pm = pl + ((pr - pl) >> 1);
         amergesort0_string(pl, pm, strItem, pw, strlen);
         amergesort0_string(pm, pr, strItem, pw, strlen);
         for (pi = pw, pj = pl; pj < pm;) {
            *pi++ = *pj++;
         }
         pi = pw + (pm - pl);
         pj = pw;
         pk = pl;
         while (pj < pi && pm < pr) {
            if (STRING_LT(strItem + (*pm)*strlen, strItem + (*pj)*strlen, strlen)) {
               *pk++ = *pm++;
            }
            else {
               *pk++ = *pj++;
            }
         }
         while (pj < pi) {
            *pk++ = *pj++;
         }
      }
      else {
         /* insertion sort */
         for (pi = pl + 1; pi < pr; ++pi) {
            vi = *pi;
            vp = strItem + (vi*strlen);
            pj = pi;
            pk = pi - 1;
            while (pj > pl && STRING_LT(vp, strItem + (*pk)*strlen, strlen)) {
               *pj-- = *pk--;
            }
            *pj = vi;
         }
      }
   }


  //-----------------------------------------------------------------------------------------------
  template <typename UINDEX>
  static void
  amergesort0_unicode(UINDEX *pl, UINDEX *pr, const char *strItem, UINDEX *pw, INT64 strlen)
  {
     const char* vp;
     UINDEX vi, *pi, *pj, *pk, *pm;

     if (pr - pl > SMALL_MERGESORT) {
        /* merge sort */
        pm = pl + ((pr - pl) >> 1);
        amergesort0_unicode(pl, pm, strItem, pw, strlen);
        amergesort0_unicode(pm, pr, strItem, pw, strlen);
        for (pi = pw, pj = pl; pj < pm;) {
           *pi++ = *pj++;
        }
        pi = pw + (pm - pl);
        pj = pw;
        pk = pl;
        while (pj < pi && pm < pr) {
           if (UNICODE_LT(strItem + (*pm)*strlen, strItem + (*pj)*strlen, strlen)) {
              *pk++ = *pm++;
           }
           else {
              *pk++ = *pj++;
           }
        }
        while (pj < pi) {
           *pk++ = *pj++;
        }
     }
     else {
        /* insertion sort */
        for (pi = pl + 1; pi < pr; ++pi) {
           vi = *pi;
           vp = strItem + (vi*strlen);
           pj = pi;
           pk = pi - 1;
           while (pj > pl && UNICODE_LT(vp, strItem + (*pk)*strlen, strlen)) {
              *pj-- = *pk--;
           }
           *pj = vi;
        }
     }
  }



  //-----------------------------------------------------------------------------------------------
  template <typename UINDEX>
  static void
  amergesort0_void(UINDEX *pl, UINDEX *pr, const char *strItem, UINDEX *pw, INT64 strlen)
  {
     const char* vp;
     UINDEX vi, *pi, *pj, *pk, *pm;

     if (pr - pl > SMALL_MERGESORT) {
        /* merge sort */
        pm = pl + ((pr - pl) >> 1);
        amergesort0_void(pl, pm, strItem, pw, strlen);
        amergesort0_void(pm, pr, strItem, pw, strlen);
        for (pi = pw, pj = pl; pj < pm;) {
           *pi++ = *pj++;
        }
        pi = pw + (pm - pl);
        pj = pw;
        pk = pl;
        while (pj < pi && pm < pr) {
           if (VOID_LT(strItem + (*pm)*strlen, strItem + (*pj)*strlen, strlen)) {
              *pk++ = *pm++;
           }
           else {
              *pk++ = *pj++;
           }
        }
        while (pj < pi) {
           *pk++ = *pj++;
        }
     }
     else {
        /* insertion sort */
        for (pi = pl + 1; pi < pr; ++pi) {
           vi = *pi;
           vp = strItem + (vi*strlen);
           pj = pi;
           pk = pi - 1;
           while (pj > pl && VOID_LT(vp, strItem + (*pk)*strlen, strlen)) {
              *pj-- = *pk--;
           }
           *pj = vi;
        }
     }
  }


   //-----------------------------------------------------------------------------------------------
   template <typename T, typename UINDEX> 
   static void
   // T= data type == float32/float64
   // UINDEX = INT32 or INT64
   amergesort0_float(UINDEX *pl, UINDEX *pr, T *v, UINDEX *pw, INT64 strlen=0)
   {
      T vp;
      UINDEX vi, *pi, *pj, *pk, *pm;

      //PLOGGING("merging %llu bytes of data at %p\n", pr - pl, v);

      if (pr - pl > SMALL_MERGESORT) {
         /* merge sort */
         pm = pl + ((pr - pl) >> 1);
         amergesort0_float(pl, pm, v, pw);
         amergesort0_float(pm, pr, v, pw);

         //if (COMPARE_LT(v[*pm], v[*(pm - 1)])) {
         {
            // MERGES DATA, memcpy
#ifdef USE_MEMCPY
            //INT64 bytescopy = (pm - pl) * sizeof(UINDEX);
            //aligned_block_copy_backwards((INT64*)pw, (INT64*)pl, bytescopy);
            //   
            memcpy(pw, pl, (pm-pl) * sizeof(UINDEX));

            //pi = pw +(pm-pl) -1;
            //pj = pl + (pm-pl) -1;
            //while (pj >= pl) {
            //   *pi-- = *pj--;
            //}

#else
         // Copy left side into workspace
            pi = pw;
            pj = pl;
            while (pj < pm) {
               *pi++ = *pj++;
            }
#endif
            // merge
            pi = pw + (pm - pl);
            pj = pw;
            pk = pl;

            while (pj < pi && pm < pr) {
               if (COMPARE_LT(v[*pm], v[*pj])) {
                  *pk++ = *pm++;
               }
               else {
                  *pk++ = *pj++;
               }
            }
            while (pj < pi) {
               *pk++ = *pj++;
            }
         }
      }
      else {
         /* insertion sort */
         for (pi = pl + 1; pi < pr; ++pi) {
            vi = *pi;
            vp = v[vi];
            pj = pi;
            pk = pi - 1;
            while (pj > pl && COMPARE_LT(vp, v[*pk])) {
               *pj-- = *pk--;
            }
            *pj = vi;
         }
      }
   }


   //-----------------------------------------------------------------------------------------------
   // T= data type == int16,int32,uint32,int64.uint64
   // UINDEX = INT32 or INT64
   template <typename T, typename UINDEX>
   static void 
   amergesort0_(UINDEX *pl, UINDEX *pr, T *v, UINDEX *pw)
   {
      T vp;
      UINDEX vi, *pi, *pj, *pk, *pm;

      if (pr - pl > SMALL_MERGESORT) {
         /* merge sort */
         pm = pl + ((pr - pl) >> 1);
         amergesort0_(pl, pm, v, pw);
         amergesort0_(pm, pr, v, pw);

         // check if already sorted
         // if the first element on the right is less than the last element on the left
         //printf("comparing %d to %d ", (int)pm[0], (int)pm[-1]);
         if (T_LT(v[*pm], v[*(pm-1)])) {

            for (pi = pw, pj = pl; pj < pm;) {
               *pi++ = *pj++;
            }
            pi = pw + (pm - pl);
            pj = pw;
            pk = pl;
            while (pj < pi && pm < pr) {
               if (T_LT(v[*pm], v[*pj])) {
                  *pk++ = *pm++;
               }
               else {
                  *pk++ = *pj++;
               }
            }
            while (pj < pi) {
               *pk++ = *pj++;
            }
         }

      }
      else {
         /* insertion sort */
         for (pi = pl + 1; pi < pr; ++pi) {
            vi = *pi;
            vp = v[vi];
            pj = pi;
            pk = pi - 1;
            while (pj > pl && T_LT(vp, v[*pk])) {
               *pj-- = *pk--;
            }
            *pj = vi;
         }
      }
   }



   //-----------------------------------------------------------------------------------------------
   // allocates workspace
   template <typename T, typename UINDEX>
   static int 
   amergesort_(T *v, UINDEX *tosort, UINDEX num)
   {
      UINDEX *pl, *pr, *pworkspace;

      pl = tosort;
      pr = pl + num;

      pworkspace = (UINDEX*)WORKSPACE_ALLOC((num / 2) * sizeof(UINDEX));
      if (pworkspace == NULL) {
         return -1;
      }
      amergesort0_(pl, pr, v, pworkspace);
      WORKSPACE_FREE(pworkspace);

      return 0;
   }


   template <typename T, typename UINDEX>
   static int
   amergesort_float(T *v, UINDEX *tosort, UINDEX num)
   {
      UINDEX *pl, *pr, *pworkspace;

      pl = tosort;
      pr = pl + num;

      pworkspace = (UINDEX*)WORKSPACE_ALLOC((num / 2) * sizeof(UINDEX));
      if (pworkspace == NULL) {
         return -1;
      }
      amergesort0_float(pl, pr, v, pworkspace);
      WORKSPACE_FREE(pworkspace);

      return 0;
   }


   //---------------------------------------------------------------------------
   // Called to combine the result of the left and right merge
   template <typename UINDEX>
   static void
   ParMergeString(void* pValue1, void* pToSort1, INT64 totalLen, INT64 strlen, void* pWorkSpace1) {
      UINDEX *pl, *pr;

      UINDEX* pWorkSpace = (UINDEX*)pWorkSpace1;
      UINDEX* pToSort = (UINDEX*)pToSort1;
      const char* pValue = (char*)pValue1;

      pl = pToSort;
      pr = pl + totalLen;

      //PLOGGING("string calling with %llu   %p  %p  %p  %p\n", totalLen, pl, pr, pValue, pWorkSpace);
      amergesort0_string(pl, pr, pValue, pWorkSpace, strlen);
   }


   //---------------------------------------------------------------------------
   // Called to combine the result of the left and right merge
   template <typename UINDEX>
   static void
   ParMergeUnicode(void* pValue1, void* pToSort1, INT64 totalLen, INT64 strlen, void* pWorkSpace1) {
      UINDEX *pl, *pr;

      UINDEX* pWorkSpace = (UINDEX*)pWorkSpace1;
      UINDEX* pToSort = (UINDEX*)pToSort1;
      const char* pValue = (char*)pValue1;

      pl = pToSort;
      pr = pl + totalLen;

      //PLOGGING("unicode calling with %llu   %p  %p  %p  %p\n", totalLen, pl, pr, pValue, pWorkSpace);
      amergesort0_unicode(pl, pr, pValue, pWorkSpace, strlen);
   }

   //---------------------------------------------------------------------------
   // Called to combine the result of the left and right merge
   template <typename UINDEX>
   static void
   ParMergeVoid(void* pValue1, void* pToSort1, INT64 totalLen, INT64 strlen, void* pWorkSpace1) {
      UINDEX *pl, *pr;

      UINDEX* pWorkSpace = (UINDEX*)pWorkSpace1;
      UINDEX* pToSort = (UINDEX*)pToSort1;
      const char* pValue = (char*)pValue1;

      pl = pToSort;
      pr = pl + totalLen;

      //PLOGGING("void calling with %llu   %p  %p  %p  %p\n", totalLen, pl, pr, pValue, pWorkSpace);
      amergesort0_void(pl, pr, pValue, pWorkSpace, strlen);
   }

   //---------------------------------------------------------------------------
   // Called to combine the result of the left and right merge
   template <typename T, typename UINDEX>
   static void
   ParMergeNormal(void* pValue1, void* pToSort1, INT64 totalLen, INT64 strlen, void* pWorkSpace1) {
      UINDEX *pl, *pr;

      UINDEX* pWorkSpace = (UINDEX*)pWorkSpace1;
      UINDEX* pToSort = (UINDEX*)pToSort1;
      T* pValue = (T*)pValue1;

      pl = pToSort;
      pr = pl + totalLen;

      PLOGGING("normal calling with %llu   %p  %p  %p  %p\n", totalLen, pl, pr, pValue, pWorkSpace);
      amergesort0_(pl, pr, pValue, pWorkSpace);
   }

   //---------------------------------------------------------------------------
   // Called to combine the result of the left and right merge
   template <typename T, typename UINDEX> 
   static void
   ParMergeFloat(void* pValue1, void* pToSort1, INT64 totalLen, INT64 strlen, void* pWorkSpace1) {
      UINDEX *pl, *pr;

      UINDEX* pWorkSpace = (UINDEX*)pWorkSpace1;
      UINDEX* pToSort = (UINDEX*)pToSort1;
      T* pValue = (T*)pValue1;

      pl = pToSort;
      pr = pl + totalLen;

      PLOGGING("float calling with %llu   %p  %p  %p  %p\n", totalLen, pl, pr, pValue, pWorkSpace);
      amergesort0_float(pl, pr, pValue, pWorkSpace);
   }



   //---------------------------------------------------------------------------
   // Called to combine the result of the left and right merge
   template <typename UINDEX>
   static void
   ParMergeMergeString(void* pValue1, void* pToSort1, INT64 totalLen, INT64 strlen, void* pWorkSpace1) {
      UINDEX *pl, *pr;

      UINDEX* pWorkSpace = (UINDEX*)pWorkSpace1;
      UINDEX* pToSort = (UINDEX*)pToSort1;
      const char* pValue = (char*)pValue1;

      pl = pToSort;
      pr = pl + totalLen;

      UINDEX *pi, *pj, *pk, *pm;
      pm = pl + ((pr - pl) >> 1);


      //if (COMPARE_LT(v[*pm], v[*(pm - 1)])) {

      //   for (pi = pw, pj = pl; pj < pm;) {
      //      *pi++ = *pj++;
      //   }
      //   pi = pw + (pm - pl);
      //   pj = pw;
      //   pk = pl;
      //   while (pj < pi && pm < pr) {
      //      if (COMPARE_LT(v[*pm], v[*pj])) {
      //         *pk++ = *pm++;
      //      }
      //      else {
      //         *pk++ = *pj++;
      //      }
      //   }
      //   while (pj < pi) {
      //      *pk++ = *pj++;
      //   }
      //}

      // BUG BUG doing lexsort on two arrays: string, int.  Once sorted, resorting does not work.
      if (TRUE || STRING_LT(pValue + (*pm)*strlen, pValue + (*pm-1)*strlen, strlen)) {

         //printf("%lld %lld %lld %lld\n", (INT64)pValue[*pl], (INT64)pValue[*(pm - 1)], (INT64)pValue[*pm], (INT64)pValue[*(pr - 1)]);
         //printf("%lld %lld %lld %lld %lld\n", (INT64)*pl, (INT64)*(pm - 2), (INT64)*(pm - 1), (INT64)*pm, (INT64)*(pr - 1));

         memcpy(pWorkSpace, pl, (pm - pl) * sizeof(UINDEX));

         // pi is end of workspace
         pi = pWorkSpace + (pm - pl);
         pj = pWorkSpace;
         pk = pl;

         while (pj < pi && pm < pr) {
            if (STRING_LT(pValue +(*pm)*strlen, pValue + (*pj)*strlen, strlen)) {
               *pk++ = *pm++;
            }
            else {
               *pk++ = *pj++;
            }
         }
         while (pj < pi) {
            *pk++ = *pj++;
         }

         //printf("last items %lld %lld %lld\n", pk[-3], pk[-2], pk[-1]);
      }
      else {
         PLOGGING("**Already sorted string %lld\n", (INT64)(*pm));

      }

   }



   //---------------------------------------------------------------------------
   // Called to combine the result of the left and right merge
   template <typename UINDEX>
   static void
   ParMergeMergeUnicode(void* pValue1, void* pToSort1, INT64 totalLen, INT64 strlen, void* pWorkSpace1) {
      UINDEX *pl, *pr;

      UINDEX* pWorkSpace = (UINDEX*)pWorkSpace1;
      UINDEX* pToSort = (UINDEX*)pToSort1;
      const char* pValue = (char*)pValue1;

      pl = pToSort;
      pr = pl + totalLen;

      UINDEX *pi, *pj, *pk, *pm;
      pm = pl + ((pr - pl) >> 1);

      if (TRUE || UNICODE_LT(pValue + (*pm)*strlen, pValue + (*pm - 1)*strlen, strlen)) {

         //printf("%lld %lld %lld %lld\n", (INT64)pValue[*pl], (INT64)pValue[*(pm - 1)], (INT64)pValue[*pm], (INT64)pValue[*(pr - 1)]);
         //printf("%lld %lld %lld %lld %lld\n", (INT64)*pl, (INT64)*(pm - 2), (INT64)*(pm - 1), (INT64)*pm, (INT64)*(pr - 1));

         memcpy(pWorkSpace, pl, (pm - pl) * sizeof(UINDEX));

         // pi is end of workspace
         pi = pWorkSpace + (pm - pl);
         pj = pWorkSpace;
         pk = pl;

         while (pj < pi && pm < pr) {
            if (UNICODE_LT(pValue + (*pm)*strlen, pValue + (*pj)*strlen, strlen)) {
               *pk++ = *pm++;
            }
            else {
               *pk++ = *pj++;
            }
         }
         while (pj < pi) {
            *pk++ = *pj++;
         }

         //printf("last items %lld %lld %lld\n", pk[-3], pk[-2], pk[-1]);
      }
      else {
         PLOGGING("**Already sorted unicode %lld\n", (INT64)(*pm));


      }

   }

   //---------------------------------------------------------------------------
   // Called to combine the result of the left and right merge
   template <typename UINDEX>
   static void
   ParMergeMergeVoid(void* pValue1, void* pToSort1, INT64 totalLen, INT64 strlen, void* pWorkSpace1) {
      UINDEX *pl, *pr;

      UINDEX* pWorkSpace = (UINDEX*)pWorkSpace1;
      UINDEX* pToSort = (UINDEX*)pToSort1;
      const char* pValue = (char*)pValue1;

      pl = pToSort;
      pr = pl + totalLen;

      UINDEX *pi, *pj, *pk, *pm;
      pm = pl + ((pr - pl) >> 1);

      // BUG BUG doing lexsort on two arrays: string, int.  Once sorted, resorting does not work.
      if (TRUE || VOID_LT(pValue + (*pm)*strlen, pValue + (*pm - 1)*strlen, strlen)) {

         memcpy(pWorkSpace, pl, (pm - pl) * sizeof(UINDEX));

         // pi is end of workspace
         pi = pWorkSpace + (pm - pl);
         pj = pWorkSpace;
         pk = pl;

         while (pj < pi && pm < pr) {
            if (VOID_LT(pValue + (*pm)*strlen, pValue + (*pj)*strlen, strlen)) {
               *pk++ = *pm++;
            }
            else {
               *pk++ = *pj++;
            }
         }
         while (pj < pi) {
            *pk++ = *pj++;
         }

         //printf("last items %lld %lld %lld\n", pk[-3], pk[-2], pk[-1]);
      }
      else {
         PLOGGING("**Already sorted void %lld\n", (INT64)(*pm));

      }

   }


   //---------------------------------------------------------------------------
   // Called to combine the result of the left and right merge
   // T is type to sort -- INT32, Float64, etc.
   // UINDEX is the argsort index -- INT32 or INT64 often
   //
   template <typename T, typename UINDEX>
   static void
   ParMergeMerge(void* pValue1, void* pToSort1, INT64 totalLen, INT64 strlen, void* pWorkSpace1) {
      UINDEX *pl, *pr;

      UINDEX* pWorkSpace = (UINDEX*)pWorkSpace1;
      UINDEX* pToSort = (UINDEX*)pToSort1;
      T* pValue = (T*)pValue1;

      pl = pToSort;
      pr = pl + totalLen;

      UINDEX *pi, *pj, *pk, *pm;
      pm = pl + ((pr - pl) >> 1);


      //if (COMPARE_LT(v[*pm], v[*(pm - 1)])) {

      //   for (pi = pw, pj = pl; pj < pm;) {
      //      *pi++ = *pj++;
      //   }
      //   pi = pw + (pm - pl);
      //   pj = pw;
      //   pk = pl;
      //   while (pj < pi && pm < pr) {
      //      if (COMPARE_LT(v[*pm], v[*pj])) {
      //         *pk++ = *pm++;
      //      }
      //      else {
      //         *pk++ = *pj++;
      //      }
      //   }
      //   while (pj < pi) {
      //      *pk++ = *pj++;
      //   }
      //}

      // quickcheck to see if we have to copy
      // BUG BUG doing lexsort on two arrays: string, int.  Once sorted, resorting does not work.
      if (TRUE || COMPARE_LT(pValue[*pm], pValue[*(pm - 1)])) {

         //printf("%lld %lld %lld %lld\n", (INT64)pValue[*pl], (INT64)pValue[*(pm - 1)], (INT64)pValue[*pm], (INT64)pValue[*(pr - 1)]);
         //printf("%lld %lld %lld %lld %lld\n", (INT64)*pl, (INT64)*(pm - 2), (INT64)*(pm - 1), (INT64)*pm, (INT64)*(pr - 1));

         memcpy(pWorkSpace, pl, (pm - pl) * sizeof(UINDEX));

         // pi is end of workspace
         pi = pWorkSpace + (pm - pl);
         pj = pWorkSpace;
         pk = pl;

         while (pj < pi && pm < pr) {
            if (COMPARE_LT(pValue[*pm], pValue[*pj])) {
               *pk++ = *pm++;
            }
            else {
               *pk++ = *pj++;
            }
         }
         while (pj < pi) {
            *pk++ = *pj++;
         }

         //printf("last items %lld %lld %lld\n", pk[-3], pk[-2], pk[-1]);
      }
      else {
         PLOGGING("**Already sorted %lld\n", (INT64)(*pm));
      }

   }


   typedef void(*MERGE_STEP_ONE)(void* pValue, void* pToSort, INT64 num, INT64 strlen, void* pWorkSpace);
   typedef void(*MERGE_STEP_TWO)(void* pValue1, void* pToSort1, INT64 totalLen, INT64 strlen, void* pWorkSpace1);
   //--------------------------------------------------------------------
   struct MERGE_STEP_ONE_CALLBACK {
      MERGE_STEP_ONE MergeCallbackOne;
      MERGE_STEP_TWO MergeCallbackTwo;

      void* pValues;
      void* pToSort;
      INT64 ArrayLength;

      // set to 0 if not a string, otherwise the string length
      INT64 StrLen;

      void* pWorkSpace;
      INT64 MergeBlocks;
      INT64 TypeSizeInput;
      INT64 TypeSizeOutput;

      // used to synchronize parallel merges
      INT64 EndPositions[8];
      INT64 Level[3];

   } stParMergeCallback;



   //------------------------------------------------------------------------------
   //  Concurrent callback from multiple threads
   static BOOL ParMergeThreadCallback(struct stMATH_WORKER_ITEM* pstWorkerItem, int core, INT64 workIndex) {
      MERGE_STEP_ONE_CALLBACK* Callback = (MERGE_STEP_ONE_CALLBACK*)pstWorkerItem->WorkCallbackArg;
      BOOL didSomeWork = FALSE;

      INT64 index;
      INT64 workBlock;


      // As long as there is work to do
      while ((index = pstWorkerItem->GetNextWorkIndex(&workBlock)) > 0) {
         // First index is 1 so we subtract
         index--;

         //PLOGGING("[%d] DoWork start loop -- %lld  index: %lld\n", core, workIndex, index);

         // the very first index starts at 0
         INT64 pFirst = 0;
         
         if (index >= 1) {
            pFirst = Callback->EndPositions[index - 1];
         }

         INT64 pSecond = Callback->EndPositions[index];
         char* pToSort1 = (char*)(Callback->pToSort);

         INT64 MergeSize = (pSecond - pFirst);
         INT64 OffsetSize = pFirst;
         PLOGGING("%d : MergeOne index: %llu  %lld  %lld  %lld\n", core, index, pFirst, MergeSize, OffsetSize);

         // Workspace uses half the size
         //char* pWorkSpace1 = (char*)pWorkSpace + (offsetAdjToSort / 2);
         char* pWorkSpace1 = (char*)Callback->pWorkSpace + (OffsetSize * Callback->TypeSizeOutput);

         Callback->MergeCallbackOne(Callback->pValues, pToSort1 + (OffsetSize * Callback->TypeSizeOutput), MergeSize, Callback->StrLen, pWorkSpace1);

         INT64 bitshift = 1LL << index;
         
         // Now find the buddy bit (adjacent bit)
         INT64 buddy = 0;
         if (index & 1) {
            buddy = 1LL << (index-1);
         }
         else {
            buddy = 1LL << (index + 1);
         }

         // Get back which bits were set before the OR operation
         INT64 result = FMInterlockedOr(&Callback->Level[0], bitshift);

         // Check if our buddy was already set
         PLOGGING("index: %lld  %lld %lld -- %s\n", index, buddy, (result & buddy), buddy == (result & buddy) ? "GOOD" : "WAIT");

         if (buddy == (result & buddy)) {
            // Move to next level -- 4 things to sort
            index = index / 2;
            index = index * 2;
            index += 1;

            pFirst = 0;

            if (index >= 2) {
               pFirst = Callback->EndPositions[index - 2];
            }

            pSecond = Callback->EndPositions[index];
            pToSort1 = (char*)(Callback->pToSort);
            MergeSize = (pSecond - pFirst);
            OffsetSize = pFirst;

            PLOGGING("%d : MergeTwo index: %llu  %lld  %lld  %lld\n", core, index, pFirst, MergeSize, OffsetSize);
            pWorkSpace1 = (char*)Callback->pWorkSpace + (OffsetSize * Callback->TypeSizeOutput);
            Callback->MergeCallbackTwo(Callback->pValues, pToSort1 + (OffsetSize * Callback->TypeSizeOutput), MergeSize, Callback->StrLen, pWorkSpace1);

            index /= 2;
            bitshift = 1LL << index;

            // Now find the buddy bit (adjacent bit)
            buddy = 0;
            if (index & 1) {
               buddy = 1LL << (index - 1);
            }
            else {
               buddy = 1LL << (index + 1);
            }

            // Get back which bits were set before the OR operation
            result = FMInterlockedOr(&Callback->Level[1], bitshift);

            // Check if our buddy was already set
            PLOGGING("index -- LEVEL 2: %lld  %lld %lld -- %s\n", index, buddy, (result & buddy), buddy == (result & buddy) ? "GOOD" : "WAIT");

            if (buddy == (result & buddy)) {
               // Move to next level -- 2 things to sort
               index = index / 2;
               index = index * 4;
               index += 3;

               pFirst = 0;

               if (index >= 4) {
                  pFirst = Callback->EndPositions[index - 4];
               }

               pSecond = Callback->EndPositions[index];
               pToSort1 = (char*)(Callback->pToSort);
               MergeSize = (pSecond - pFirst);
               OffsetSize = pFirst;

               PLOGGING("%d : MergeThree index: %llu  %lld  %lld  %lld\n", core, index, pFirst, MergeSize, OffsetSize);
               pWorkSpace1 = (char*)Callback->pWorkSpace + (OffsetSize * Callback->TypeSizeOutput);
               Callback->MergeCallbackTwo(Callback->pValues, pToSort1 + (OffsetSize * Callback->TypeSizeOutput), MergeSize, Callback->StrLen, pWorkSpace1);


               index /= 4;
               bitshift = 1LL << index;

               // Now find the buddy bit (adjacent bit)
               buddy = 0;
               if (index & 1) {
                  buddy = 1LL << (index - 1);
               }
               else {
                  buddy = 1LL << (index + 1);
               }

               // Get back which bits were set before the OR operation
               result = FMInterlockedOr(&Callback->Level[2], bitshift);

               if (buddy == (result & buddy)) {
                  // Final merge
                  PLOGGING("%d : MergeFinal index: %llu  %lld  %lld  %lld\n", core, index, 0LL, Callback->ArrayLength, 0LL);
                  stParMergeCallback.MergeCallbackTwo(Callback->pValues, Callback->pToSort, Callback->ArrayLength, Callback->StrLen, Callback->pWorkSpace);
               }

            }

         }

         // Indicate we completed a block
         didSomeWork = TRUE;

         // tell others we completed this work block
         pstWorkerItem->CompleteWorkBlock();
      }

      return didSomeWork;
   }

   //========================================================================
   //
   enum PAR_SORT_TYPE {
      Normal =0,
      Float = 1,
      String = 2,
      Unicode = 3,
      Void = 4
   };

   typedef int(*SINGLE_MERGESORT)(
      void*          pValuesT,
      void*          pToSortU,
      INT64          arrayLength,
      INT64          strlen,
      PAR_SORT_TYPE  sortType);
   
   //------------------------------------------------------------------------
   // single threaded version
   // T is the dtype int32/float32/float64/etc.
   // UINDEX is either INT32 or INT64
   // Returns -1 on failure
   template <typename T, typename UINDEX>
   static int
   single_amergesort(
      void*          pValuesT,
      void*          pToSortU,
      INT64          arrayLength,
      INT64          strlen,
      PAR_SORT_TYPE  sortType)
   {
      T*             pValues = (T*)pValuesT;
      UINDEX*        pToSort = (UINDEX*)pToSortU;

      // single threaded sort
      UINDEX *pWorkSpace;

      pWorkSpace = (UINDEX*)WORKSPACE_ALLOC((arrayLength / 2) * sizeof(UINDEX));
      if (pWorkSpace == NULL) {
         return -1;
      }

      switch (sortType) {
      case PAR_SORT_TYPE::Float:
         amergesort0_float(pToSort, pToSort + arrayLength, pValues, pWorkSpace);
         break;
      case PAR_SORT_TYPE::String:
         amergesort0_string(pToSort, pToSort + arrayLength, (const char*)pValues, pWorkSpace, strlen);
         break;
      case PAR_SORT_TYPE::Unicode:
         amergesort0_unicode(pToSort, pToSort + arrayLength, (const char*)pValues, pWorkSpace, strlen);
         break;
      case PAR_SORT_TYPE::Void:
         amergesort0_void(pToSort, pToSort + arrayLength, (const char*)pValues, pWorkSpace, strlen);
         break;
      default:
         amergesort0_(pToSort, pToSort + arrayLength, pValues, pWorkSpace);
      }

      WORKSPACE_FREE(pWorkSpace);
      return 0;
   }


   //------------------------------------------------------------------------
   // parallel version
   // if strlen==0, then not string (int or float)
   // If pCutOffs is not null, will go parallel per partition
   // If pCutOffs is null, the entire array is sorted
   // If the array is large enough, a parallel merge sort is invoked
   // Returns -1 on failure
   template <typename T, typename UINDEX> 
   static int
   par_amergesort(
      INT64*         pCutOffs,      // May be NULL (if so no partitions)
      INT64          cutOffLength,

      T*             pValues,
      UINDEX*        pToSort, 

      INT64          arrayLength,
      INT64          strlen, 
      PAR_SORT_TYPE  sortType)
   {
      if (pCutOffs) {
         PLOGGING("partition version col: %lld  %p  %p  %p\n", cutOffLength, pToSort, pToSort + arrayLength, pValues);

         struct stPSORT {
            SINGLE_MERGESORT  funcSingleMerge;
            INT64*            pCutOffs;       // May be NULL (if so no partitions)
            INT64             cutOffLength;

            char*             pValues;
            char*             pToSort;
            INT64             strlen;
            PAR_SORT_TYPE     sortType;

            INT64             sizeofT;
            INT64             sizeofUINDEX;

         } psort;

         psort.funcSingleMerge = single_amergesort<T,UINDEX>;
         psort.pCutOffs = pCutOffs;
         psort.cutOffLength = cutOffLength;
         psort.pValues = (char*)pValues;
         psort.pToSort = (char*)pToSort;
         psort.strlen = strlen;
         psort.sortType = sortType;

         psort.sizeofUINDEX = sizeof(UINDEX);
         if (strlen > 0) {
            psort.sizeofT = strlen;
         }
         else {
            psort.sizeofT = sizeof(T);
         }

         // Use threads per partition
         auto lambdaPSCallback = [](void* callbackArgT, int core, INT64 workIndex) -> BOOL {
            stPSORT* callbackArg = (stPSORT*)callbackArgT;
            INT64 t = workIndex;
            INT64 partLength;
            INT64 partStart;

            if (t == 0) {
               partStart = 0;
            }
            else {
               partStart = callbackArg->pCutOffs[t - 1];
            }

            partLength = callbackArg->pCutOffs[t] - partStart;

            PLOGGING("[%lld] start: %lld  length: %lld\n", t, partStart, partLength);

            // shift the data pointers to match the partition
            // call a single threaded merge
            callbackArg->funcSingleMerge(
               callbackArg->pValues + (partStart * callbackArg->sizeofT),
               callbackArg->pToSort + (partStart * callbackArg->sizeofUINDEX),
               partLength,
               callbackArg->strlen,
               callbackArg->sortType);

            return TRUE;
         };

         g_cMathWorker->DoMultiThreadedWork((int)cutOffLength, lambdaPSCallback, &psort);

      } else

      // If size is large, go parallel
      if (arrayLength >= CMathWorker::WORK_ITEM_BIG) {

         PLOGGING("Parallel version  %p  %p  %p\n", pToSort,  pToSort + arrayLength, pValues);
         // Divide into 8 jobs
         // Allocate all memory up front?
         //UINDEX* pWorkSpace = (UINDEX*)WORKSPACE_FREE(((arrayLength / 2) + 256)* sizeof(UINDEX));
         void* pWorkSpace = NULL;
         UINT64 allocSize = arrayLength * sizeof(UINDEX);

         //(UINDEX*)WORKSPACE_ALLOC(arrayLength * sizeof(UINDEX));
         pWorkSpace=WorkSpaceAllocLarge(allocSize);

         if (pWorkSpace == NULL) {
            return -1;
         }

         MERGE_STEP_ONE mergeStepOne = NULL;
         
         switch (sortType) {
         case PAR_SORT_TYPE::Float:
            mergeStepOne = ParMergeFloat<T, UINDEX>;
            break;
         case PAR_SORT_TYPE::String:
            mergeStepOne = ParMergeString<UINDEX>;
            break;
         case PAR_SORT_TYPE::Unicode:
            mergeStepOne = ParMergeUnicode<UINDEX>;
            break;
         case PAR_SORT_TYPE::Void:
            mergeStepOne = ParMergeVoid<UINDEX>;
            break;
         default:
            mergeStepOne = ParMergeNormal<T, UINDEX>;
         }

         stMATH_WORKER_ITEM* pWorkItem = g_cMathWorker->GetWorkItem(arrayLength);

         if (pWorkItem == NULL) {

            // Threading not allowed for this work item, call it directly from main thread
            mergeStepOne(pValues, pToSort, arrayLength, strlen, pWorkSpace);
         }
         else {

            pWorkItem->DoWorkCallback = ParMergeThreadCallback;
            pWorkItem->WorkCallbackArg = &stParMergeCallback;

            stParMergeCallback.MergeCallbackOne = mergeStepOne;
            switch (sortType) {

            case PAR_SORT_TYPE::String:
               stParMergeCallback.MergeCallbackTwo = ParMergeMergeString< UINDEX>;
               break;
            case PAR_SORT_TYPE::Unicode:
               stParMergeCallback.MergeCallbackTwo = ParMergeMergeUnicode< UINDEX>;
               break;
            case PAR_SORT_TYPE::Void:
               stParMergeCallback.MergeCallbackTwo = ParMergeMergeVoid< UINDEX>;
               break;
            default:
               // Last Merge
               stParMergeCallback.MergeCallbackTwo = ParMergeMerge<T, UINDEX>;
            };


            stParMergeCallback.pValues = pValues;
            stParMergeCallback.pToSort = pToSort;
            stParMergeCallback.ArrayLength = arrayLength;
            stParMergeCallback.StrLen = strlen;
            stParMergeCallback.pWorkSpace = pWorkSpace;
            stParMergeCallback.TypeSizeInput = sizeof(T);
            if (strlen) {
               stParMergeCallback.TypeSizeInput = strlen;
            }
            stParMergeCallback.TypeSizeOutput = sizeof(UINDEX);

            //NOTE set this value to 2,4 or 8
            stParMergeCallback.MergeBlocks = 8;

            for (int i = 0; i < 3; i++) {
               stParMergeCallback.Level[i] = 0;
            }

            if (stParMergeCallback.MergeBlocks == 2) {

               stParMergeCallback.EndPositions[1] = arrayLength;
               stParMergeCallback.EndPositions[0] = arrayLength / 2;
            }
            else if (stParMergeCallback.MergeBlocks == 4) {

               stParMergeCallback.EndPositions[3] = arrayLength;
               stParMergeCallback.EndPositions[1] = arrayLength / 2;
               stParMergeCallback.EndPositions[2] = stParMergeCallback.EndPositions[1] + (stParMergeCallback.EndPositions[3] - stParMergeCallback.EndPositions[1]) / 2;
               stParMergeCallback.EndPositions[0] = 0 + (stParMergeCallback.EndPositions[1] - 0) / 2;
            }

            else {
               // We use an 8 way merge, we need the size breakdown
               stParMergeCallback.EndPositions[7] = arrayLength;
               stParMergeCallback.EndPositions[3] = arrayLength / 2;
               stParMergeCallback.EndPositions[5] = stParMergeCallback.EndPositions[3] + (stParMergeCallback.EndPositions[7] - stParMergeCallback.EndPositions[3]) / 2;
               stParMergeCallback.EndPositions[1] = 0 + (stParMergeCallback.EndPositions[3] - 0) / 2;
               stParMergeCallback.EndPositions[6] = stParMergeCallback.EndPositions[5] + (stParMergeCallback.EndPositions[7] - stParMergeCallback.EndPositions[5]) / 2;
               stParMergeCallback.EndPositions[4] = stParMergeCallback.EndPositions[3] + (stParMergeCallback.EndPositions[5] - stParMergeCallback.EndPositions[3]) / 2;
               stParMergeCallback.EndPositions[2] = stParMergeCallback.EndPositions[1] + (stParMergeCallback.EndPositions[3] - stParMergeCallback.EndPositions[1]) / 2;
               stParMergeCallback.EndPositions[0] = 0 + (stParMergeCallback.EndPositions[1] - 0) / 2;
            }

            // This will notify the worker threads of a new work item
            g_cMathWorker->WorkMain(pWorkItem, stParMergeCallback.MergeBlocks, 0, 1, FALSE);

         }

         // Free temp memory used
         WorkSpaceFreeAllocLarge(pWorkSpace, allocSize);

      }
      else {

         // single threaded sort
         return   
         single_amergesort<T, UINDEX>(
            pValues,
            pToSort,
            arrayLength,
            strlen,
            sortType);
      }

      return 0;
   }




   //-----------------------------------------------------------------------------------------------
   // Sorts in place
   // TODO: Make multithreaded like
   template <typename T>
   static int
   SortInPlace(void* pDataIn1, INT64 arraySize1, SORT_MODE mode) {

      int result = 0;

      switch (mode) {
      case SORT_MODE::SORT_MODE_QSORT:
         result = quicksort_((T*)pDataIn1, arraySize1);
         break;

      case SORT_MODE::SORT_MODE_MERGE:
         result = mergesort_((T*)pDataIn1, arraySize1);
         break;

      case SORT_MODE::SORT_MODE_HEAP:
         result = heapsort_<T>((T*)pDataIn1, arraySize1);
         break;

      }

      if (result != 0) {
         printf("**Error sorting.  size %llu   mode %d\n", arraySize1, mode);
      }

      return result;
   }



   //-----------------------------------------------------------------------------------------------
   // Sorts in place
   template <typename T>
   static int
   SortInPlaceFloat(void* pDataIn1, INT64 arraySize1, SORT_MODE mode) {

      int result = 0;

      switch (mode) {
      case SORT_MODE::SORT_MODE_QSORT:
         result = quicksort_((T*)pDataIn1, arraySize1);
         break;

      case SORT_MODE::SORT_MODE_MERGE:
         result = mergesort_float((T*)pDataIn1, arraySize1);
         break;

      case SORT_MODE::SORT_MODE_HEAP:
         result = heapsort_<T>((T*)pDataIn1, arraySize1);
         break;

      }

      if (result != 0) {
         printf("**Error sorting.  size %llu   mode %d\n", (INT64)arraySize1, mode);
      }

      return result;
   }

   //-----------------------------------------------------------------------------------------------
   template <typename T, typename UINDEX>
   static int
   SortIndex(
      INT64* pCutOffs, INT64 cutOffLength, void* pDataIn1, UINDEX* toSort, UINDEX arraySize1, SORT_MODE mode) {

      int result = 0;

      switch (mode) {
      case SORT_MODE::SORT_MODE_QSORT:
         result = aquicksort_<T,UINDEX>((T*)pDataIn1, (UINDEX*)toSort, arraySize1);
         break;

      //case SORT_MODE::SORT_MODE_MERGE:
      //   result = amergesort_<T, UINDEX>((T*)pDataIn1, (UINDEX*)toSort, arraySize1);
      //   break;
      case SORT_MODE::SORT_MODE_MERGE:
         result = par_amergesort<T, UINDEX>(pCutOffs, cutOffLength, (T*)pDataIn1, (UINDEX*)toSort, arraySize1, 0, PAR_SORT_TYPE::Normal);
         break;

      case SORT_MODE::SORT_MODE_HEAP:
         result = aheapsort_<T, UINDEX>((T*)pDataIn1, (UINDEX*)toSort, arraySize1);
         break;

      }

      if (result != 0) {
         printf("**Error sorting.  size %llu   mode %d\n", (INT64)arraySize1, mode);
      }

      return result;
   }



   //-----------------------------------------------------------------------------------------------
   template <typename T, typename UINDEX>
   static int
   SortIndexFloat(INT64* pCutOffs, INT64 cutOffLength, void* pDataIn1, UINDEX* toSort, UINDEX arraySize1, SORT_MODE mode) {

      int result = 0;

      switch (mode) {
      case SORT_MODE::SORT_MODE_QSORT:
         result = aquicksort_float<T, UINDEX>((T*)pDataIn1, (UINDEX*)toSort, arraySize1);
         break;

      case SORT_MODE::SORT_MODE_MERGE:
         result = par_amergesort<T, UINDEX>(pCutOffs, cutOffLength, (T*)pDataIn1, (UINDEX*)toSort, arraySize1, 0, PAR_SORT_TYPE::Float);
         break;

      case SORT_MODE::SORT_MODE_HEAP:
         result = aheapsort_float<T, UINDEX>((T*)pDataIn1, (UINDEX*)toSort, arraySize1);
         break;

      }

      if (result != 0) {
         printf("**Error sorting.  size %llu   mode %d\n", (INT64)arraySize1, mode);
      }

      return result;
   }


   //-----------------------------------------------------------------------------------------------
   template <typename UINDEX>
   static int
   SortIndexString(INT64* pCutOffs, INT64 cutOffLength, const char* pDataIn1, UINDEX* toSort, UINDEX arraySize1, SORT_MODE mode, UINDEX strlen) {

      int result = 0;
      switch (mode) {
      default:
      case SORT_MODE::SORT_MODE_MERGE:
         result = par_amergesort<char, UINDEX>(pCutOffs, cutOffLength, (char*)pDataIn1, (UINDEX*)toSort, arraySize1, strlen, PAR_SORT_TYPE::String);
         break;

      }

      return result;
   }


   //-----------------------------------------------------------------------------------------------
   template <typename UINDEX>
   static int
   SortIndexUnicode(INT64* pCutOffs, INT64 cutOffLength, const char* pDataIn1, UINDEX* toSort, UINDEX arraySize1, SORT_MODE mode, UINDEX strlen) {

      int result = 0;
      switch (mode) {
      default:
      case SORT_MODE::SORT_MODE_MERGE:
         result = par_amergesort<char, UINDEX>(pCutOffs, cutOffLength, (char*)pDataIn1, (UINDEX*)toSort, arraySize1, strlen, PAR_SORT_TYPE::Unicode);
         break;

      }

      return result;
   }

   //-----------------------------------------------------------------------------------------------
   template <typename UINDEX>
   static int
   SortIndexVoid(INT64* pCutOffs, INT64 cutOffLength, const char* pDataIn1, UINDEX* toSort, UINDEX arraySize1, SORT_MODE mode, UINDEX strlen) {

      int result = 0;
      switch (mode) {
      default:
      case SORT_MODE::SORT_MODE_MERGE:
         result = par_amergesort<char, UINDEX>(pCutOffs, cutOffLength, (char*)pDataIn1, (UINDEX*)toSort, arraySize1, strlen, PAR_SORT_TYPE::Void);
         break;
      }

      return result;
   }




//================================================================================================
//===============================================================================
   static void SortArray(void* pDataIn1, INT64 arraySize1, INT32 arrayType1, SORT_MODE mode) {
      switch (arrayType1) {
      case NPY_STRING:
         SortInPlace<char>(pDataIn1, arraySize1, mode);
         break;
      case NPY_BOOL:
         SortInPlace<bool>(pDataIn1, arraySize1, mode);
         break;
      case NPY_INT8:
         SortInPlace<INT8>(pDataIn1, arraySize1, mode);
         break;
      case NPY_INT16:
         SortInPlace<INT16>(pDataIn1, arraySize1, mode);
         break;
      CASE_NPY_INT32:
         SortInPlace<INT32>(pDataIn1, arraySize1, mode);
         break;
      CASE_NPY_INT64:
         SortInPlace<INT64>(pDataIn1, arraySize1, mode);
         break;
      case NPY_UINT8:
         SortInPlace<UINT8>(pDataIn1, arraySize1, mode);
         break;
      case NPY_UINT16:
         SortInPlace<UINT16>(pDataIn1, arraySize1, mode);
         break;
      CASE_NPY_UINT32:
         SortInPlace<UINT32>(pDataIn1, arraySize1, mode);
         break;
      CASE_NPY_UINT64:
         SortInPlace<UINT64>(pDataIn1, arraySize1, mode);
         break;
      case NPY_FLOAT:
         SortInPlaceFloat<float>(pDataIn1, arraySize1, mode);
         break;
      case NPY_DOUBLE:
         SortInPlaceFloat<double>(pDataIn1, arraySize1, mode);
         break;
      case NPY_LONGDOUBLE:
         SortInPlaceFloat<long double>(pDataIn1, arraySize1, mode);
         break;
      default:
         printf("SortArray does not understand type %d\n", arrayType1);
         break;
      }
   }


//===============================================================================
// Pass in a numpy array, it will be sorted in place
// the same array is returned
// TODO: Make multithreaded
PyObject* SortInPlace(PyObject *self, PyObject *args) {
   PyArrayObject *inArr1 = NULL;
   int sortMode;

   if (!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &inArr1, &sortMode)) return NULL;

   INT32 arrayType1 = PyArray_TYPE(inArr1);
   int ndim = PyArray_NDIM(inArr1);
   npy_intp* dims = PyArray_DIMS(inArr1);

   INT64 arraySize1 = CalcArrayLength(ndim, dims);
   void* pDataIn1 = PyArray_BYTES(inArr1);

   //printf("Arary size %llu  type %d\n", arraySize1, arrayType1);

   SORT_MODE mode = (SORT_MODE)sortMode;

   SortArray(pDataIn1, arraySize1, arrayType1, mode);

   Py_IncRef((PyObject*)inArr1);
   return SetFastArrayView(inArr1);
}


//===============================================================================
// Pass in a numpy array, it will be sorted in place
// the same array is returned
// Arg1: iKey (array of uniques)
// Arg2: iUniqueSort (resorting of iUnique)
//
// NOTE: every element of iKey must be an index into iUniqueSort (the max element = max(iUniqueSort)-1)
// Output:
//   Arg1 = iUniqueSort[Arg1[i]]
PyObject* SortInPlaceIndirect(PyObject *self, PyObject *args) {
   PyArrayObject *inArr1 = NULL;
   PyArrayObject *inSort = NULL;
   
   // THIS CODE IS NOT FINSIHED

   if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &inArr1, &PyArray_Type, &inSort)) return NULL;

   INT32 arrayType1 = PyArray_TYPE(inArr1);
   INT32 sortType = PyArray_TYPE(inSort);

   INT64 arraySize1 = ArrayLength(inArr1);
   INT64 sortSize = ArrayLength(inSort);


   if (arrayType1 == NPY_INT32 && sortType == NPY_INT32) {
      INT32* pDataIn = (INT32*)PyArray_BYTES(inArr1);
      INT32* pSort = (INT32*)PyArray_BYTES(inSort);

      INT32* inverseSort = (INT32*)WORKSPACE_ALLOC(sortSize * sizeof(INT32));
      for (int i = 0; i < sortSize; i++) {
         inverseSort[pSort[i]] = i;
      }

      for (int i = 0; i < arraySize1; i++) {
         pDataIn[i] = inverseSort[pDataIn[i]];
      }

      WORKSPACE_FREE(inverseSort);
   }
   else if (arrayType1 == NPY_INT32 && sortType == NPY_INT64) {
      INT32* pDataIn = (INT32*)PyArray_BYTES(inArr1);
      INT64* pSort = (INT64*)PyArray_BYTES(inSort);

      INT64* inverseSort = (INT64*)WORKSPACE_ALLOC(sortSize * sizeof(INT64));
      for (INT64 i = 0; i < sortSize; i++) {
         inverseSort[pSort[i]] = i;
      }

      for (int i = 0; i < arraySize1; i++) {
         pDataIn[i] = (INT32)inverseSort[pDataIn[i]];
      }
      WORKSPACE_FREE(inverseSort);

   } else if (arrayType1 == NPY_INT64 && sortType == NPY_INT64) {
      INT64* pDataIn = (INT64*)PyArray_BYTES(inArr1);
      INT64* pSort = (INT64*)PyArray_BYTES(inSort);

      INT64* inverseSort = (INT64*)WORKSPACE_ALLOC(sortSize * sizeof(INT64));
      for (INT64 i = 0; i < sortSize; i++) {
         inverseSort[pSort[i]] = i;
      }

      for (INT64 i = 0; i < arraySize1; i++) {
         pDataIn[i] = inverseSort[pDataIn[i]];
      }
      WORKSPACE_FREE(inverseSort);

   }
   else {
      printf("**SortInplaceIndirect failure!  arrays must be int32 or int64\n");
   }

   Py_IncRef((PyObject*)inArr1);
   return SetFastArrayView(inArr1);
}


//===============================================================================
// Pass in a numpy array, it will be copied and sorted in the return array
// the same array is returned
PyObject* Sort(PyObject *self, PyObject *args) {
   PyArrayObject *inArr1 = NULL;
   int sortMode;

   if (!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &inArr1, &sortMode)) return NULL;

   INT32 arrayType1 = PyArray_TYPE(inArr1);
   int ndim = PyArray_NDIM(inArr1);
   npy_intp* dims = PyArray_DIMS(inArr1);

   INT64 arraySize1 = CalcArrayLength(ndim, dims);

   // The output is a boolean where the nth item was found
   PyArrayObject* duplicateArray = AllocateNumpyArray(ndim, dims, arrayType1);

   if (duplicateArray == NULL) {
      PyErr_Format(PyExc_ValueError, "Sort out of memory");
      return NULL;
   }
   
   void* pDataIn1 = PyArray_BYTES(inArr1);
   void* pDataOut1 = PyArray_BYTES(duplicateArray);

   INT64 itemSize = NpyItemSize((PyObject*)inArr1);

   memcpy(pDataOut1, pDataIn1, arraySize1 * itemSize);

   SORT_MODE mode = (SORT_MODE)sortMode;

   SortArray(pDataOut1, arraySize1, arrayType1, mode);

   return SetFastArrayView(duplicateArray);
}

//------------------------------------------------------------------------------------------
// Internal and can be called from groupby
// caller must allocate the pDataOut1 as INT64 with size arraySize1
// UINDEX = INT32 or INT64
template <typename UINDEX>
static void SortIndex(
   INT64*      pCutOffs,
   INT64       cutOffLength,

   void*       pDataIn1,
   UINDEX      arraySize1, 
   UINDEX*     pDataOut1, 
   SORT_MODE   mode, 
   INT32       arrayType1, 
   UINDEX      strlen) {

   switch (arrayType1) {
   case NPY_UNICODE:
      SortIndexUnicode<UINDEX>(pCutOffs, cutOffLength, (const char*)pDataIn1, pDataOut1, arraySize1, mode, strlen);
      break;
   case NPY_VOID:
      SortIndexVoid<UINDEX>(pCutOffs, cutOffLength, (const char*)pDataIn1, pDataOut1, arraySize1, mode, strlen);
      break;
   case NPY_STRING:
      SortIndexString<UINDEX>(pCutOffs, cutOffLength, (const char*)pDataIn1, pDataOut1, arraySize1, mode, strlen);
      break;
   case NPY_BOOL:
   case NPY_INT8:
      SortIndex<INT8, UINDEX>(pCutOffs, cutOffLength, pDataIn1, pDataOut1, arraySize1, mode);
      break;
   case NPY_INT16:
      SortIndex<INT16, UINDEX>(pCutOffs, cutOffLength, pDataIn1, pDataOut1, arraySize1, mode);
      break;
   CASE_NPY_INT32:
      SortIndex<INT32, UINDEX>(pCutOffs, cutOffLength, pDataIn1, pDataOut1, arraySize1, mode);
      break;
   CASE_NPY_INT64:
      SortIndex<INT64, UINDEX>(pCutOffs, cutOffLength, pDataIn1, pDataOut1, arraySize1, mode);
      break;
   case NPY_UINT8:
      SortIndex<UINT8, UINDEX>(pCutOffs, cutOffLength, pDataIn1, pDataOut1, arraySize1, mode);
      break;
   case NPY_UINT16:
      SortIndex<UINT16, UINDEX>(pCutOffs, cutOffLength, pDataIn1, pDataOut1, arraySize1, mode);
      break;
   CASE_NPY_UINT32:
      SortIndex<UINT32, UINDEX>(pCutOffs, cutOffLength, pDataIn1, pDataOut1, arraySize1, mode);
      break;
   CASE_NPY_UINT64:
      SortIndex<UINT64, UINDEX>(pCutOffs, cutOffLength, pDataIn1, pDataOut1, arraySize1, mode);
      break;
   case NPY_FLOAT:
      SortIndexFloat<float, UINDEX>(pCutOffs, cutOffLength, pDataIn1, pDataOut1, arraySize1, mode);
      break;
   case NPY_DOUBLE:
      SortIndexFloat<double, UINDEX>(pCutOffs, cutOffLength, pDataIn1, pDataOut1, arraySize1, mode);
      break;
   case NPY_LONGDOUBLE:
      SortIndexFloat<long double, UINDEX>(pCutOffs, cutOffLength, pDataIn1, pDataOut1, arraySize1, mode);
      break;
   default:
      printf("SortIndex does not understand type %d\n", arrayType1);
   }

}

//================================================================================
typedef int(*IS_SORTED_FUNC)(const char* pDataIn1, INT64 arraySize1, INT64 strlennotused);
//-----------------------------------------------------------------------------------------------
template <typename T>
static int
IsSortedFloat(const char* pDataIn1, INT64 arraySize1, INT64 strlennotused) {

   int result = 0;
   T* pData = (T*)pDataIn1;

   INT64 i = arraySize1 - 1;

   while ((i > 0) && (pData[i] != pData[i])) {
      i--;
   }

   while ((i > 0) && pData[i] >= pData[i - 1]) {
      i--;
   }

   return i <= 0;
}


//-----------------------------------------------------------------------------------------------
template <typename T>
static int
IsSorted(const char* pDataIn1, INT64 arraySize1, INT64 strlennotused) {

   int result = 0;
   T* pData = (T*)pDataIn1;

   INT64 i = arraySize1 - 1;

   while ((i > 0) && pData[i] >= pData[i - 1]) {
      i--;
   }

   return i <= 0;
}


//-----------------------------------------------------------------------------------------------
static int
IsSortedString(const char* pData, INT64 arraySize1, INT64 strlen) {

   int result = 0;

   INT64 i = arraySize1 - 1;

   while ((i > 0) && !(STRING_LT(&pData[i*strlen], &pData[(i - 1)*strlen], strlen))) {
      i--;
   }

   return i <= 0;
}


//-----------------------------------------------------------------------------------------------
static int
IsSortedUnicode(const char* pData, INT64 arraySize1, INT64 strlen) {

   int result = 0;

   INT64 i = arraySize1 - 1;

   while ((i > 0) && !(UNICODE_LT(&pData[i*strlen], &pData[(i - 1)*strlen], strlen))) {
      i--;
   }

   return i <= 0;
}

//-----------------------------------------------------------------------------------------------
static int
IsSortedVoid(const char* pData, INT64 arraySize1, INT64 strlen) {

   int result = 0;

   INT64 i = arraySize1 - 1;

   while ((i > 0) && !(VOID_LT(&pData[i*strlen], &pData[(i - 1)*strlen], strlen))) {
      i--;
   }

   return i <= 0;
}



//===============================================================================
// returns True or False
// Nans at the end are fine and still considered sorted
PyObject* IsSorted(PyObject *self, PyObject *args) {
   PyArrayObject *inArr1 = NULL;

   if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &inArr1)) return NULL;

   INT32 arrayType1 = PyArray_TYPE(inArr1);
   int ndim = PyArray_NDIM(inArr1);
   npy_intp* dims = PyArray_DIMS(inArr1);

   INT64 itemSize = PyArray_ITEMSIZE(inArr1);

   if (ndim != 1 || itemSize != PyArray_STRIDE(inArr1, 0)) {
      PyErr_Format(PyExc_ValueError, "IsSorted arrays must be one dimensional and contiguous.  ndim is %d\n", ndim);
      return NULL;
   }

   INT64 arraySize1 = CalcArrayLength(ndim, dims);
   void* pDataIn1 = PyArray_BYTES(inArr1);

   LOGGING("issorted size %llu  type %d\n", arraySize1, arrayType1);

   INT64 result = 0;
   IS_SORTED_FUNC pSortedFunc = NULL;

   switch (arrayType1) {
   case NPY_BOOL:
   case NPY_INT8:
      pSortedFunc = IsSorted<INT8>;
      break;
   case NPY_INT16:
      pSortedFunc = IsSorted<INT16>;
      break;
   CASE_NPY_INT32:
      pSortedFunc = IsSorted<INT32>;
      break;
   CASE_NPY_INT64:
      pSortedFunc = IsSorted<INT64>;
      break;
   case NPY_UINT8:
      pSortedFunc = IsSorted<UINT8>;
      break;
   case NPY_UINT16:
      pSortedFunc = IsSorted<UINT16>;
      break;
   CASE_NPY_UINT32:
      pSortedFunc = IsSorted<UINT32>;
      break;
   CASE_NPY_UINT64:
      pSortedFunc = IsSorted<UINT64>;
      break;
   case NPY_FLOAT:
      pSortedFunc = IsSortedFloat<float>;
      break;
   case NPY_DOUBLE:
      pSortedFunc = IsSortedFloat<double>;
      break;
   case NPY_LONGDOUBLE:
      pSortedFunc = IsSortedFloat<long double>;
      break;
   case NPY_VOID:
      pSortedFunc = IsSortedVoid;
      break;
   case NPY_STRING:
      pSortedFunc = IsSortedString;
      break;
   case NPY_UNICODE:
      pSortedFunc = IsSortedUnicode;
      break;

   default:
      PyErr_Format(PyExc_ValueError, "IsSorted does not understand type %d\n", arrayType1);
      return NULL;
   }


   // MT callback
   struct IsSortedCallbackStruct {
      INT64             IsSorted;
      IS_SORTED_FUNC    pSortedFunc;
      const char*       pDataIn1;
      INT64             ArraySize;
      INT64             ItemSize;
   } stISCallback{ 1, pSortedFunc, (const char*)pDataIn1, arraySize1, itemSize };

   // This is the routine that will be called back from multiple threads
   auto lambdaISCallback = [](void* callbackArgT, int core, INT64 start, INT64 length) -> BOOL {
      IsSortedCallbackStruct* cb = (IsSortedCallbackStruct*)callbackArgT;

      // check if short circuited (any segment not sorted)
      if (cb->IsSorted) {
         // If not the first segment, then overlap by going back
         if (start != 0) {
            start--;
            length++;
         }
         int result = cb->pSortedFunc(cb->pDataIn1 + (start * cb->ItemSize), length, cb->ItemSize);

         // on success, return TRUE 
         if (result) return TRUE;

         // on failure, set the failure flag and return FALSE
         cb->IsSorted = 0;
      }

      return FALSE;
   };

   // A zero length array is considered sorted
   g_cMathWorker->DoMultiThreadedChunkWork(arraySize1, lambdaISCallback, &stISCallback);

   result = stISCallback.IsSorted;

   if (result) {
      Py_INCREF(Py_True);
      return Py_True;
   }
   else {
      Py_INCREF(Py_False);
      return Py_False;

   }

}


//===============================================================================
// checks for kwargs cutoff
// if exists, and is INT64, returns pointer and length of cutoffs
// cutoffLength of -1 indicates an error
INT64* GetCutOffs(PyObject *kwargs, INT64& cutoffLength) {
   // Check for cutoffs kwarg to see if going into parallel mode
   if (kwargs && PyDict_Check(kwargs)) {
      PyArrayObject* pCutOffs = NULL;
      // Borrowed reference
      // Returns NULL if key not present
      pCutOffs = (PyArrayObject*)PyDict_GetItemString(kwargs, "cutoffs");

      if (pCutOffs != NULL && PyArray_Check(pCutOffs)) {
         switch (PyArray_TYPE(pCutOffs)) {
         CASE_NPY_INT64:
            cutoffLength = ArrayLength(pCutOffs);
            return (INT64*)PyArray_BYTES(pCutOffs);
         default:
            printf("Bad cutoff dtype... make sure INT64\n");
            cutoffLength = -1;
            return NULL;
         }
      }
   }
   cutoffLength = 0;
   return NULL;
}

//===============================================================================

template<typename UINDEX>
static BOOL ARangeCallback(void* callbackArgT, int core, INT64 start, INT64 length) {

   UINDEX* pDataOut = (UINDEX*)callbackArgT;
   UINDEX istart = (UINDEX)start;
   UINDEX iend = istart + (UINDEX)length;

   for (UINDEX i = istart; i < iend; i++) {
      pDataOut[i] = i;
   }

   return TRUE;
}


// index must be INT32 or INT64
static PyArrayObject* GetKwargIndex(PyObject *kwargs, INT64& indexLength, int &dtype) {
   // Check for 'index' kwarg to see if prime lexsort
   if (kwargs && PyDict_Check(kwargs)) {
      PyArrayObject* pStartIndex = NULL;
      // Borrowed reference
      // Returns NULL if key not present
      pStartIndex = (PyArrayObject*)PyDict_GetItemString(kwargs, "index");

      if (pStartIndex != NULL && PyArray_Check(pStartIndex)) {
         indexLength = ArrayLength(pStartIndex);

         switch (PyArray_TYPE(pStartIndex)) {
         CASE_NPY_INT64:
            dtype = NPY_INT64;
            return pStartIndex;
         CASE_NPY_INT32:
            dtype = NPY_INT32;
            return pStartIndex;
         default:
            printf("Bad index dtype... make sure INT64 or INT32\n");
            indexLength = -1;
            return NULL;
         }
      }
   }
   indexLength = 0;
   return NULL;
}



//===============================================================================
// LexSort32 and LexSort64 funnel into here
// Kwargs:
//--------
// cutoffs=
// index=  specify start index instead of doing arange
// groups=True (also group)
// base_index=
// ascending??
//
// Returns
// -------
// Fancy index in desired sort order
// when group=True
// Adds iKey
// Adds iFirstKey
// Add  nCount
// NOTE: From nCount, and iFirstKey can build
//
template<typename UINDEX>
PyObject* LexSort(PyObject *self, PyObject *args, PyObject *kwargs) {
   CMultiListPrepare mlp(args);

   if (mlp.aInfo && mlp.tupleSize > 0) {
      INT64    arraySize1 = mlp.totalRows;

      INT64    cutOffLength = 0;
      INT64*   pCutOffs = GetCutOffs(kwargs, cutOffLength);

      if (pCutOffs && pCutOffs[cutOffLength - 1] != arraySize1) {
         PyErr_Format(PyExc_ValueError, "LexSort last cutoff length does not match array length %lld", arraySize1);
         return NULL;
      }
      if (cutOffLength == -1) {
         PyErr_Format(PyExc_ValueError, "LexSort 'cutoffs' must be an array of type INT64");
         return NULL;
      }

      int indexDType = 0;
      INT64 indexLength = 0;

      PyArrayObject* index = GetKwargIndex(kwargs, indexLength, indexDType);

      PyArrayObject* result = NULL;
      
      if (indexLength == -1) {
         PyErr_Format(PyExc_ValueError, "LexSort 'index' must be an array of type INT64 or INT32");
         return NULL;
      }

      if (indexLength > 0) {
         if (indexLength > arraySize1) {
            PyErr_Format(PyExc_ValueError, "LexSort 'index' is larger than value array");
            return NULL;
         }

         if (sizeof(UINDEX) == 8 && indexDType != NPY_INT64) {
            PyErr_Format(PyExc_ValueError, "LexSort 'index' is not INT64");
            return NULL;
         }

         if (sizeof(UINDEX) == 4 && indexDType != NPY_INT32) {
            PyErr_Format(PyExc_ValueError, "LexSort 'index' is not INT32");
            return NULL;
         }

         // reduce what we sort to the startindex
         arraySize1 = indexLength;
         result = index;
         Py_IncRef((PyObject*)result);
      }
      else {
         result = AllocateLikeNumpyArray(mlp.aInfo[0].pObject, sizeof(UINDEX) == 4 ? NPY_INT32 : NPY_INT64);
      }

      if (result) {
         // Return the sorted index
         UINDEX* pDataOut = (UINDEX*)PyArray_BYTES(result);

         // BUG? what if we have index= and cutoffs= ??
         if (pCutOffs) {
            LOGGING("Have cutoffs %lld\n", cutOffLength);

            // For cutoffs, prep the indexes with 0:n for each partition
            UINDEX* pCounter = pDataOut;

            INT64 startPos = 0;
            for (INT64 j = 0; j < cutOffLength; j++) {

               INT64 endPos = pCutOffs[j];
               INT64 partitionLength = endPos - startPos;
               for (UINDEX i = 0; i < partitionLength; i++) {
                  *pCounter++ = i;
               }
               startPos = endPos;
            }
         }
         else {

            // If the user did not provide a start index, we make one
            if (index == NULL) {
               g_cMathWorker->DoMultiThreadedChunkWork(arraySize1, ARangeCallback<UINDEX>, pDataOut);
            }
         }

         if (pCutOffs) {
            // Turn off caching of large memory allocs
            g_cMathWorker->NoCaching = TRUE;
         }

         // When multiple arrays are passed, we sort in order of how it is passed
         // Thus, the last array is the last sort, and therefore determines the primary sort order
         for (UINDEX i = 0; i < mlp.tupleSize; i++) {
            // For each array...
            SortIndex<UINDEX>(pCutOffs, cutOffLength, mlp.aInfo[i].pData, (UINDEX)arraySize1, pDataOut, SORT_MODE::SORT_MODE_MERGE, mlp.aInfo[i].NumpyDType, (UINDEX)mlp.aInfo[i].ItemSize);

         }

         if (pCutOffs) {
            g_cMathWorker->NoCaching = FALSE;
         }
         return (PyObject*)result;
      }
   }

   Py_INCREF(Py_None);
   return Py_None;
}


//===============================================================================
// Returns INT32
PyObject* LexSort32(PyObject *self, PyObject *args, PyObject *kwargs) {

   return LexSort<INT32>(self, args, kwargs);
}


//===============================================================================
// Returns INT64
PyObject* LexSort64(PyObject *self, PyObject *args, PyObject *kwargs) {

   return LexSort<INT64>(self, args, kwargs);
}


//===============================================================================
//===============================================================================
// checks for kwargs filter
// if exists, and is BOOL, returns pointer and length of bool
static bool* GetFilter(PyObject *kwargs, INT64& filterLength) {
   // Check for cutoffs kwarg to see if going into parallel mode
   if (kwargs && PyDict_Check(kwargs)) {
      PyArrayObject* pFilter = NULL;
      // Borrowed reference
      // Returns NULL if key not present
      pFilter = (PyArrayObject*)PyDict_GetItemString(kwargs, "filter");

      if (pFilter != NULL && PyArray_Check(pFilter)) {
         switch (PyArray_TYPE(pFilter)) {
         case NPY_BOOL:
            filterLength = ArrayLength(pFilter);
            return (bool*)PyArray_BYTES(pFilter);
         }
      }
   }
   filterLength = 0;
   return NULL;
}


static INT64 GetBaseIndex(PyObject *kwargs) {
   if (kwargs && PyDict_Check(kwargs)) {
      PyObject* pBaseIndex = NULL;
      // Borrowed reference
      // Returns NULL if key not present
      pBaseIndex = PyDict_GetItemString(kwargs, "base_index");
      if (pBaseIndex != NULL && PyLong_Check(pBaseIndex)) {

         long baseindex = PyLong_AsLong(pBaseIndex);

         // only zero or one allowed
         if (baseindex == 0) return 0;
      }
   }
   return 1;
}


//================================================================================
// Group via lexsort
// CountOut
//================================================================================
//typedef INT64 *(GROUP_INDEX_FUNC)()

template <typename T, typename UINDEX>
static INT64 GroupIndexStep2(
   void*       pDataIn1,
   UINDEX      arraySize1,
   UINDEX*     pDataIndexIn,
   UINDEX*     pGroupOut,
   UINDEX*     pFirstOut,
   UINDEX*     pCountOut,
   bool*       pFilter,
   INT64       base_index,
   UINDEX      strlen = 0) {

   T*          pDataIn = (T*)pDataIn1;
   UINDEX      curIndex = pDataIndexIn[0];
   UINDEX      curCount = 1;
   T           val1 = pDataIn[curIndex];
   UINDEX      baseIndex = (UINDEX)base_index;
   UINDEX      curGroup = 0;

   // NOTE: filtering does not work!!
   //if (pFilter) {
   //   // base index must be 1
   //   // Invalid bin init
   //   // currently nothing filtered out
   //   pCountOut[0] = 0;

   //   UINDEX invalid = *(UINDEX*)GetInvalid<UINDEX>();

   //   pFirstOut[0] = invalid;
   //   val1 = -1;

   //   // TJD NOTE...
   //   // Think have to see how many filtered out values up front
   //   curCount = 1;

   //   UINDEX zeroCount = 0;

   //   for (UINDEX i = 0; i < arraySize1; i++) {
   //      curIndex = pDataIndexIn[i];

   //      if (pFilter[i]) {
   //         T val2 = pDataIn[curIndex];

   //         if (val1 == val2) {
   //            curCount++;
   //            pGroupOut[curIndex] = curGroup + 1;
   //         }
   //         else {
   //            curGroup++;
   //            pCountOut[curGroup] = curCount;
   //            pFirstOut[curGroup] = curIndex;
   //            pGroupOut[curIndex] = curGroup + 1;
   //            val1 = val2;
   //            curCount = 1;
   //         }
   //      }
   //      else {
   //         zeroCount++;
   //         pGroupOut[curIndex] = 0;
   //         if (pFirstOut[0] == invalid) {
   //            pFirstOut[0] = curIndex;
   //         }
   //      }
   //   }
   //   curGroup++;
   //   pCountOut[curGroup] = curCount;

   //   // the zero count tally
   //   pCountOut[0] = zeroCount;

   //   return curGroup;
   //}


    {
      if (base_index == 0) {
         // SHIFT countout

         // Invalid bin init
         // currently nothing filtered out
         pFirstOut[0] = curIndex;
         pGroupOut[curIndex] = 0;

         for (UINDEX i = 1; i < arraySize1; i++) {
            curIndex = pDataIndexIn[i];
            T val2 = pDataIn[curIndex];

            if (val1 == val2) {
               curCount++;
               pGroupOut[curIndex] = curGroup;
            }
            else {
               pCountOut[curGroup] = curCount;
               curGroup++;
               pFirstOut[curGroup] = curIndex;
               pGroupOut[curIndex] = curGroup;
               val1 = val2;
               curCount = 1;
            }
         }
         pCountOut[curGroup] = curCount;
         curGroup++;
      }
      else {
         // Invalid bin init
         // currently nothing filtered out
         pCountOut[0] = 0;

         pFirstOut[0] = curIndex;
         pGroupOut[curIndex] = 1;

         for (UINDEX i = 1; i < arraySize1; i++) {
            curIndex = pDataIndexIn[i];
            T val2 = pDataIn[curIndex];

            if (val1 == val2) {
               curCount++;
               pGroupOut[curIndex] = curGroup + 1;
            }
            else {
               curGroup++;
               pCountOut[curGroup] = curCount;
               pFirstOut[curGroup] = curIndex;
               pGroupOut[curIndex] = curGroup + 1;
               val1 = val2;
               curCount = 1;
            }
         }
         curGroup++;
         pCountOut[curGroup] = curCount;
      }
      return curGroup;
   }
}


template <typename T, typename UINDEX>
static INT64 GroupIndexStep2String(
   void*       pDataIn1,
   UINDEX      arraySize1,
   UINDEX*     pDataIndexIn,
   UINDEX*     pGroupOut,
   UINDEX*     pFirstOut,
   UINDEX*     pCountOut,
   bool*       pFilter,
   INT64       base_index,
   INT64       strlen) {

   T*          pDataIn = (T*)pDataIn1;
   UINDEX      curIndex = pDataIndexIn[0];
   UINDEX      curCount = 1;
   T*          val1 = &pDataIn[curIndex*strlen];
   UINDEX      baseIndex = (UINDEX)base_index;
   UINDEX      curGroup = 0;

   // Invalid bin init when base_index is 1
   pCountOut[0] = 0;

   pFirstOut[0] = curIndex;
   pGroupOut[curIndex] = baseIndex;

   for (UINDEX i = 1; i < arraySize1; i++) {
      curIndex = pDataIndexIn[i];
      T* val2 = &pDataIn[curIndex*strlen];

      if (BINARY_LT(val1, val2, strlen) == 0) {
         curCount++;
         pGroupOut[curIndex] = curGroup + baseIndex;
      }
      else {
         curGroup++;
         pCountOut[curGroup] = curCount;
         pFirstOut[curGroup] = curIndex;
         pGroupOut[curIndex] = curGroup + baseIndex;
         val1 = val2;
         curCount = 1;
      }
   }
   curGroup++;
   pCountOut[curGroup] = curCount;
   return curGroup;
}

typedef INT64(*GROUP_INDEX_FUNC)(
   void*       pDataIn1,
   INT64       arraySize1V,
   void*       pDataIndexInV,
   void*       pGroupOutV,
   void*       pFirstOutV,
   void*       pCountOutV,
   bool*       pFilter,       // optional
   INT64       base_index,
   INT64       strlen);


//------------------------------------------------------------------------------------------
// Internal and can be called from groupby
// caller must allocate the pGroupOut as INT32 or INT64 with size arraySize1
// UINDEX = INT32 or INT64
template <typename UINDEX>
static INT64 GroupIndex(
   void*       pDataIn1,
   INT64       arraySize1V,
   void*       pDataIndexInV,
   void*       pGroupOutV,
   void*       pFirstOutV,
   void*       pCountOutV,
   bool*       pFilter,       // optional
   INT64       base_index,
   INT64       strlen) {

   INT64       uniqueCount = 0;

   UINDEX*     pDataIndexIn = (UINDEX*)pDataIndexInV;
   UINDEX*     pGroupOut = (UINDEX*)pGroupOutV;
   UINDEX*     pFirstOut = (UINDEX*)pFirstOutV;
   UINDEX*     pCountOut = (UINDEX*)pCountOutV;
   UINDEX      arraySize1 = (UINDEX)arraySize1V;

   switch (strlen) {
   case 1:
      uniqueCount =
         GroupIndexStep2<INT8, UINDEX>(pDataIn1, arraySize1, pDataIndexIn, pGroupOut, pFirstOut, pCountOut, pFilter, base_index, 0);
      break;
   case 2:
      uniqueCount =
         GroupIndexStep2<INT16, UINDEX>(pDataIn1, arraySize1, pDataIndexIn, pGroupOut, pFirstOut, pCountOut, pFilter, base_index, 0);
      break;
   case 4:
      uniqueCount =
         GroupIndexStep2<INT32, UINDEX>(pDataIn1, arraySize1, pDataIndexIn, pGroupOut, pFirstOut, pCountOut, pFilter, base_index, 0);
      break;
   case 8:
      uniqueCount =
         GroupIndexStep2<INT64, UINDEX>(pDataIn1, arraySize1, pDataIndexIn, pGroupOut, pFirstOut, pCountOut, pFilter, base_index, 0);
      break;
   default:
      uniqueCount =
         GroupIndexStep2String<const char, UINDEX>(pDataIn1, arraySize1, pDataIndexIn, pGroupOut, pFirstOut, pCountOut, pFilter, base_index, strlen);
      break;
   }

   return uniqueCount;
}


//===============================================================================
// TODO: Need to add checks for some array allocations in this function (to see if they succeeded,
//       and if not, free any other allocated arrays before setting a PyExc_MemoryError and returning).
template<typename UINDEX>
static PyObject* GroupFromLexSortInternal(
   PyObject*   kwargs,
   UINDEX*     pIndex,
   npy_intp    indexLength, 
   npy_intp    indexLengthValues,
   void*       pValues,
   npy_intp    itemSizeValues) {

   INT64    cutOffLength = 0;
   INT64*   pCutOffs = GetCutOffs(kwargs, cutOffLength);

   INT64    filterLength = 0;
   bool*    pFilter = GetFilter(kwargs, filterLength);

   INT64    base_index = GetBaseIndex(kwargs);


   if (pCutOffs && pCutOffs[cutOffLength - 1] != indexLength) {
      return PyErr_Format(PyExc_ValueError, "GroupFromLexSort last cutoff length does not match array length %lld", indexLength);
   }
   if (cutOffLength == -1) {
      return PyErr_Format(PyExc_ValueError, "GroupFromLexSort 'cutoffs' must be an array of type INT64");
   }
   if (pFilter && filterLength != indexLength) {
      return PyErr_Format(PyExc_ValueError, "GroupFromLexSort filter length does not match array length %lld", indexLength);
   }

   // The countout always reserves the zero bin (even for when base_index =0) for filtering out
   // TODO: Change this to use type npy_intp and check for overflow.
   INT64 worstCase = indexLength + 1 + cutOffLength;

   PyArrayObject* const keys = AllocateNumpyArray(1, (npy_intp*)&indexLengthValues, sizeof(UINDEX) == 4 ? NPY_INT32 : NPY_INT64);
   PyArrayObject* const first = AllocateNumpyArray(1, (npy_intp*)&indexLength, sizeof(UINDEX) == 4 ? NPY_INT32 : NPY_INT64);
   PyArrayObject* const count = AllocateNumpyArray(1, (npy_intp*)&worstCase, sizeof(UINDEX) == 4 ? NPY_INT32 : NPY_INT64);

   // Make sure allocations succeeded
   if (!keys || !first || !count)
   {
      // Release/recycle any of the arrays which _were_ successfully allocated so they're not leaked.
      if (keys) { RecycleNumpyInternal(keys); }
      if (first) { RecycleNumpyInternal(first); }

      return PyErr_Format(PyExc_MemoryError, "GroupFromLexSort out of memory length %lld", indexLength);
   }

   UINDEX* pKeyOut = (UINDEX*)PyArray_BYTES(keys);
   UINDEX* pFirstOut = (UINDEX*)PyArray_BYTES(first);
   UINDEX* pCountOut = (UINDEX*)PyArray_BYTES(count);

   INT64 uniqueCount = 0;
   GROUP_INDEX_FUNC  gpfunc = GroupIndex<UINDEX>;

   if (pCutOffs) {
      PyArrayObject* uniqueCounts = AllocateNumpyArray(1, (npy_intp*)&cutOffLength, NPY_INT64);
      INT64*         pUniqueCounts = (INT64*)PyArray_BYTES(uniqueCounts);

      // Turn off caching of large memory allocs
      g_cMathWorker->NoCaching = TRUE;

      PLOGGING("partition version col: %lld  %p  %p  %p\n", cutOffLength, pToSort, pToSort + arrayLength, pValues);

      struct stPGROUP {
         GROUP_INDEX_FUNC  funcSingleGroup;
         INT64*            pUniqueCounts;

         INT64*            pCutOffs;   
         INT64             cutOffLength;

         char*             pValues;
         char*             pIndex;
         char*             pKeyOut;
         char*             pFirstOut;
         char*             pCountOut;
         bool*             pFilter;
         INT64             base_index;
         INT64             strlen;
         INT64             sizeofUINDEX;

      } pgroup;

      pgroup.funcSingleGroup = gpfunc;
      pgroup.pUniqueCounts = pUniqueCounts;

      pgroup.pCutOffs = pCutOffs;
      pgroup.cutOffLength = cutOffLength;
      pgroup.pValues = (char*)pValues;
      pgroup.pIndex = (char*)pIndex;

      pgroup.pKeyOut = (char*)pKeyOut;
      pgroup.pFirstOut = (char*)pFirstOut;
      pgroup.pCountOut = (char*)pCountOut;
      pgroup.pFilter = pFilter;
      pgroup.base_index = base_index;
      pgroup.strlen = itemSizeValues;
      const INT64 INDEX_SIZE = (INT64)sizeof(UINDEX);

      pgroup.sizeofUINDEX = INDEX_SIZE;

      // Use threads per partition
      auto lambdaPSCallback = [](void* callbackArgT, int core, INT64 workIndex) -> BOOL {
         stPGROUP* callbackArg = (stPGROUP*)callbackArgT;
         INT64 t = workIndex;
         INT64 partLength;
         INT64 partStart;

         if (t == 0) {
            partStart = 0;
         }
         else {
            partStart = callbackArg->pCutOffs[t - 1];
         }

         partLength = callbackArg->pCutOffs[t] - partStart;

         PLOGGING("[%lld] start: %lld  length: %lld\n", t, partStart, partLength);

         INT64 shift = 
         // shift the data pointers to match the partition
         // call a single threaded merge
         callbackArg->pUniqueCounts[t]=
         callbackArg->funcSingleGroup(
            callbackArg->pValues + (partStart * callbackArg->strlen),
            partLength,
            callbackArg->pIndex + (partStart * callbackArg->sizeofUINDEX),
            callbackArg->pKeyOut + (partStart * callbackArg->sizeofUINDEX),
            callbackArg->pFirstOut + (partStart * callbackArg->sizeofUINDEX),
            callbackArg->pCountOut + (partStart * callbackArg->sizeofUINDEX),
            callbackArg->pFilter + partStart,
            0,       //callbackArg->base_index, fix for countout
            callbackArg->strlen);

         return TRUE;
      };

      g_cMathWorker->DoMultiThreadedWork((int)cutOffLength, lambdaPSCallback, &pgroup);
      g_cMathWorker->NoCaching = FALSE;

      // TODO: make global routine
      INT64 totalUniques = 0;
      for (int i = 0; i < cutOffLength; i++) {
         totalUniques += pUniqueCounts[i];
         pUniqueCounts[i] = totalUniques;
      }

      //printf("Total uniques %lld\n", totalUniques);

      // TODO: fix up keys
      //parallel add?
      PyArrayObject* firstReduced = AllocateNumpyArray(1, (npy_intp*)&totalUniques, INDEX_SIZE == 4 ? NPY_INT32 : NPY_INT64);

      totalUniques++;
      PyArrayObject* countReduced = AllocateNumpyArray(1, (npy_intp*)&totalUniques, INDEX_SIZE == 4 ? NPY_INT32 : NPY_INT64);

      // ANOTHER PARALEL ROUTINE to copy
      struct stPGROUPADD {
         INT64*            pUniqueCounts;

         INT64*            pCutOffs;       // May be NULL (if so no partitions)
         INT64             cutOffLength;

         char*             pIndex;
         char*             pKeyOut;
         char*             pFirstOut;
         char*             pCountOut;

         char*             pFirstReduced;
         char*             pCountReduced;
         bool*             pFilter;

         INT64             base_index;
         INT64             sizeofUINDEX;

      } pgroupadd;

      pgroupadd.pUniqueCounts = pUniqueCounts;
      pgroupadd.pCutOffs = pCutOffs;
      pgroupadd.cutOffLength = cutOffLength;

      pgroupadd.pIndex = (char*)pIndex;

      pgroupadd.pKeyOut = (char*)pKeyOut;
      pgroupadd.pFirstOut = (char*)pFirstOut;
      pgroupadd.pCountOut = (char*)pCountOut;
      pgroupadd.pFilter = pFilter;

      pgroupadd.pFirstReduced = (char*)PyArray_BYTES(firstReduced);
      pgroupadd.pCountReduced = (char*)PyArray_BYTES(countReduced);

      // skip first value since reserved for zero bin (and assign it 0)
      for (INT64 c = 0; c < INDEX_SIZE; c++) {
         *pgroupadd.pCountReduced++ = 0;
      }

      pgroupadd.base_index = base_index;
      pgroupadd.sizeofUINDEX = sizeof(UINDEX);

      // Use threads per partition
      auto lambdaPGADDCallback = [](void* callbackArgT, int core, INT64 workIndex) -> BOOL {
         stPGROUPADD* callbackArg = (stPGROUPADD*)callbackArgT;
         INT64 t = workIndex;
         INT64 partLength;
         INT64 partStart;
         INT64 uniquesBefore;

         if (t == 0) {
            partStart = 0;
            uniquesBefore = 0;
         }
         else {
            partStart = callbackArg->pCutOffs[t - 1];
            uniquesBefore = callbackArg->pUniqueCounts[t - 1];
         }

         partLength = callbackArg->pCutOffs[t] - partStart;
         //printf("[%lld] start: %lld  length: %lld  ubefore: %lld\n", t, partStart, partLength, uniquesBefore);

         if (callbackArg->sizeofUINDEX == 4) {
            INT32* pKey = (INT32*)callbackArg->pKeyOut;

            // the iGroup is fixed up
            INT32* pIndex = (INT32*)callbackArg->pIndex;
            
            // pFirst is reduced to iFirstKey (only the uniques)
            INT32* pFirst = (INT32*)callbackArg->pFirstOut; 
            INT32* pFirstReduced = (INT32*)callbackArg->pFirstReduced;  

            // becomes nCount and the very first is reserved for zero bin
            // holds all the uniques + 1 for the zero bin.
            INT32* pCount = (INT32*)callbackArg->pCountOut;
            INT32* pCountReduced = (INT32*)callbackArg->pCountReduced;

            INT32  ubefore = (INT32)uniquesBefore;

            if (t != 0) {
               pKey += partStart;
               pIndex += partStart;

               for (INT64 i = 0; i < partLength; i++) {
                  pKey[i] += ((INT32)ubefore + 1); // start at 1 (to reserve zero bin), becomes ikey
                  pIndex[i] += (INT32)partStart;
               }
            }
            else {
               pKey += partStart;

               for (INT64 i = 0; i < partLength; i++) {
                  pKey[i] += ((INT32)partStart + 1); // start at 1, becomes ikey
               }
            }

            INT64 uniqueLength = callbackArg->pUniqueCounts[t] - uniquesBefore;
            //printf("first reduced %d %lld\n", ubefore, uniqueLength);
            pFirst += partStart;
            pFirstReduced += ubefore;

            pCount += partStart;
            pCountReduced += ubefore;

            // very first [0] is for zero bin
            //pCount++;

            for (INT64 i = 0; i < uniqueLength; i++) {
               pFirstReduced[i] = pFirst[i] + (INT32)partStart;
               //printf("setting %lld ", (INT64)pCount[i]);
               pCountReduced[i] = pCount[i];
            }

         }
         else {

            INT64* pKey = (INT64*)callbackArg->pKeyOut;

            // the iGroup is fixed up
            INT64* pIndex = (INT64*)callbackArg->pIndex;

            // pFirst is reduced to iFirstKey (only the uniques)
            INT64* pFirst = (INT64*)callbackArg->pFirstOut;
            INT64* pFirstReduced = (INT64*)callbackArg->pFirstReduced;

            // becomes nCount and the very first is reserved for zero bin
            // holds all the uniques + 1 for the zero bin.
            INT64* pCount = (INT64*)callbackArg->pCountOut;
            INT64* pCountReduced = (INT64*)callbackArg->pCountReduced;

            INT64  ubefore = (INT64)uniquesBefore;

            if (t != 0) {
               pKey += partStart;
               pIndex += partStart;

               for (INT64 i = 0; i < partLength; i++) {
                  pKey[i] += ((INT64)ubefore + 1); // start at 1 (to reserve zero bin), becomes ikey
                  pIndex[i] += (INT64)partStart;
               }
            }
            else {
               pKey += partStart;

               for (INT64 i = 0; i < partLength; i++) {
                  pKey[i] += ((INT64)partStart + 1); // start at 1, becomes ikey
               }
            }

            INT64 uniqueLength = callbackArg->pUniqueCounts[t] - uniquesBefore;
            //printf("first reduced %d %lld\n", ubefore, uniqueLength);
            pFirst += partStart;
            pFirstReduced += ubefore;

            pCount += partStart;
            pCountReduced += ubefore;

            // very first [0] is for zero bin
            //pCount++;

            for (INT64 i = 0; i < uniqueLength; i++) {
               pFirstReduced[i] = pFirst[i] + (INT64)partStart;
               //printf("setting %lld ", (INT64)pCount[i]);
               pCountReduced[i] = pCount[i];
            }

         }

         return TRUE;
      };
      g_cMathWorker->DoMultiThreadedWork((int)cutOffLength, lambdaPGADDCallback, &pgroupadd);

      Py_DecRef((PyObject*)first);
      //Py_DecRef((PyObject*)count);

      PyObject* returnObject = PyList_New(4);
      PyList_SET_ITEM(returnObject, 0, (PyObject*)keys);
      PyList_SET_ITEM(returnObject, 1, (PyObject*)firstReduced); // iFirstKey
      PyList_SET_ITEM(returnObject, 2, (PyObject*)countReduced); // nCountGroup

      PyList_SET_ITEM(returnObject, 3, (PyObject*)uniqueCounts);
      return returnObject;
   }
   else {

      // When multiple arrays are passed, we sort in order of how it is passed
      // Thus, the last array is the last sort, and therefore determines the primary sort order
      uniqueCount =
         gpfunc(
            pValues,
            indexLength,
            pIndex,
            pKeyOut,
            pFirstOut,
            pCountOut,
            pFilter,
            base_index,  
            itemSizeValues);

      // prior we allocate based on worst case
      // now we know the actual unique counts
      // memcpy..
      // also count invalid bin
      PyArrayObject* firstReduced = AllocateNumpyArray(1, (npy_intp*)&uniqueCount, sizeof(UINDEX) == 4 ? NPY_INT32 : NPY_INT64);
      INT64 copySize = sizeof(UINDEX) * uniqueCount;
      memcpy(PyArray_BYTES(firstReduced), pFirstOut, copySize);

      uniqueCount++;
      PyArrayObject* countReduced = AllocateNumpyArray(1, (npy_intp*)&uniqueCount, sizeof(UINDEX) == 4 ? NPY_INT32 : NPY_INT64);
      copySize = sizeof(UINDEX) * uniqueCount;
      // reduced
      memcpy(PyArray_BYTES(countReduced), pCountOut, copySize);

      Py_DecRef((PyObject*)first);
      Py_DecRef((PyObject*)count);

      PyObject* returnObject = PyList_New(3);
      PyList_SET_ITEM(returnObject, 0, (PyObject*)keys);
      PyList_SET_ITEM(returnObject, 1, (PyObject*)firstReduced);
      PyList_SET_ITEM(returnObject, 2, (PyObject*)countReduced);
      return returnObject;
   }
}


//===============================================================================
// Args:
//     Checks for cutoffs
//     If no cutoffs
//     
//     Arg1: lex=INT32/INT64 result from lexsort
//     Arg2: value array that was sorted -- if it came from a list, convert it to a void type
//
// Returns 3 arrays
// iGroup (from lexsort)
// iFirstKey
// nCount -- from which iFirst can be derived
//
// if filtering was used then arrLength != arrLengthValues
// when this happens, the iGroup will have unfilled values for filtered out locations
PyObject* GroupFromLexSort(PyObject *self, PyObject *args, PyObject *kwargs) {

   PyArrayObject *inArrSortIndex = NULL;
   PyArrayObject *inArrValues = NULL;

   if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &inArrSortIndex, &PyArray_Type, &inArrValues))
   {
      return PyErr_Format(PyExc_TypeError, "Invalid argument types and/or count for GroupFromLexSort.");
   }

   const auto arrLength = ArrayLength(inArrSortIndex);
   const auto arrLengthValues = ArrayLength(inArrValues);

   // Due to filtering, now allow a smaller array length which might index the entire
   // size of inArrValues
   if (arrLength > arrLengthValues) {
      return PyErr_Format(PyExc_ValueError, "GroupFromLexSort input array lengths do not match: %lld vs %lld", arrLength, arrLengthValues);
   }

   const auto itemSize = PyArray_ITEMSIZE(inArrValues);

   const auto dtype = PyArray_TYPE(inArrSortIndex);

   void* pIndex = PyArray_BYTES(inArrSortIndex);
   void* pValues = PyArray_BYTES(inArrValues);

   switch (dtype) {
   CASE_NPY_INT32:
      return GroupFromLexSortInternal<INT32>(kwargs, (INT32*)pIndex, arrLength, arrLengthValues,  pValues, itemSize);
      break;

   CASE_NPY_INT64:
      return GroupFromLexSortInternal<INT64>(kwargs, (INT64*)pIndex, arrLength, arrLengthValues, pValues, itemSize);
      break;

   default:
      return PyErr_Format(PyExc_ValueError, "GroupFromLexSort does not support index type of %d", dtype);
   }
}

