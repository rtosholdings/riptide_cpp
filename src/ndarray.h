#pragma once
#include "RipTide.h"

extern Py_ssize_t GetNdArrayLen(PyObject * self);

extern int64_t CopyNdArrayToBuffer(PyObject * self, char * destBuffer, int64_t len);

/**************************************************************************/
/*                             NDArray Object                             */
/**************************************************************************/

static PyTypeObject NDArray_Type;
#define NDArray_Check(v) (Py_TYPE(v) == &NDArray_Type)

#define CHECK_LIST_OR_TUPLE(v) \
    if (! PyList_Check(v) && ! PyTuple_Check(v)) \
    { \
        PyErr_SetString(PyExc_TypeError, #v " must be a list or a tuple"); \
        return NULL; \
    }

#define PyMem_XFree(v) \
    do \
    { \
        if (v) \
            PyMem_Free(v); \
    } \
    while (0)

/* Maximum number of dimensions. */
#define ND_MAX_NDIM (2 * PyBUF_MAX_NDIM)

/* Check for the presence of suboffsets in the first dimension. */
#define HAVE_PTR(suboffsets) (suboffsets && suboffsets[0] >= 0)
/* Adjust ptr if suboffsets are present. */
#define ADJUST_PTR(ptr, suboffsets) (HAVE_PTR(suboffsets) ? *((char **)ptr) + suboffsets[0] : ptr)

/* Default: NumPy style (strides), read-only, no var-export, C-style layout */
#define ND_DEFAULT 0x000
/* User configurable flags for the ndarray */
#define ND_VAREXPORT 0x001 /* change layout while buffers are exported */
/* User configurable flags for each base buffer */
#define ND_WRITABLE 0x002         /* mark base buffer as writable */
#define ND_FORTRAN 0x004          /* Fortran contiguous layout */
#define ND_SCALAR 0x008           /* scalar: ndim = 0 */
#define ND_PIL 0x010              /* convert to PIL-style array (suboffsets) */
#define ND_REDIRECT 0x020         /* redirect buffer requests */
#define ND_GETBUF_FAIL 0x040      /* trigger getbuffer failure */
#define ND_GETBUF_UNDEFINED 0x080 /* undefined view.obj */
/* Internal flags for the base buffer */
#define ND_C 0x100          /* C contiguous layout (default) */
#define ND_OWN_ARRAYS 0x200 /* consumer owns arrays */

/* ndarray properties */
#define ND_IS_CONSUMER(nd) (((NDArrayObject *)nd)->head == &((NDArrayObject *)nd)->staticbuf)

/* ndbuf->flags properties */
#define ND_C_CONTIGUOUS(flags) (! ! (flags & (ND_SCALAR | ND_C)))
#define ND_FORTRAN_CONTIGUOUS(flags) (! ! (flags & (ND_SCALAR | ND_FORTRAN)))
#define ND_ANY_CONTIGUOUS(flags) (! ! (flags & (ND_SCALAR | ND_C | ND_FORTRAN)))

/* getbuffer() requests */
#define REQ_INDIRECT(flags) ((flags & PyBUF_INDIRECT) == PyBUF_INDIRECT)
#define REQ_C_CONTIGUOUS(flags) ((flags & PyBUF_C_CONTIGUOUS) == PyBUF_C_CONTIGUOUS)
#define REQ_F_CONTIGUOUS(flags) ((flags & PyBUF_F_CONTIGUOUS) == PyBUF_F_CONTIGUOUS)
#define REQ_ANY_CONTIGUOUS(flags) ((flags & PyBUF_ANY_CONTIGUOUS) == PyBUF_ANY_CONTIGUOUS)
#define REQ_STRIDES(flags) ((flags & PyBUF_STRIDES) == PyBUF_STRIDES)
#define REQ_SHAPE(flags) ((flags & PyBUF_ND) == PyBUF_ND)
#define REQ_WRITABLE(flags) (flags & PyBUF_WRITABLE)
#define REQ_FORMAT(flags) (flags & PyBUF_FORMAT)

/* Single node of a list of base buffers. The list is needed to implement
changes in memory layout while exported buffers are active. */
// static PyTypeObject NDArray_Type;

struct ndbuf;
typedef struct ndbuf
{
    struct ndbuf * next;
    struct ndbuf * prev;
    Py_ssize_t len;     /* length of data */
    Py_ssize_t offset;  /* start of the array relative to data */
    char * data;        /* raw data */
    int flags;          /* capabilities of the base buffer */
    Py_ssize_t exports; /* number of exports */
    Py_buffer base;     /* base buffer */
} ndbuf_t;

typedef struct
{
    PyObject_HEAD int flags; /* ndarray flags */
    ndbuf_t staticbuf;       /* static buffer for re-exporting mode */
    ndbuf_t * head;          /* currently active base buffer */
} NDArrayObject;
