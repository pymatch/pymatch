#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stddef.h>

// TODO:
// - what to do about data types? (void * everywhere?)
// - allow use as base class?
// - iterable
// - implement buffer protocol
// - implement number protocol
// - https://docs.python.org/3/c-api/structures.html#c.METH_FASTCALL

// TODO: useful macros
// - Py_ALWAYS_INLINE
// - Py_MAX(x, y)
// - Py_MIN(x, y)
// - Py_STRINGIFY(x)
// - PyDoc_STRVAR(name, str)
// - PyDoc_STR(str)

// ----------------------------------------------------------------
//   ▄▄             ▄▄   █
//  █▀▀▌           ▐▛▀   ▀
// ▐▛    ▟█▙ ▐▙██▖▐███  ██   ▟█▟▌
// ▐▌   ▐▛ ▜▌▐▛ ▐▌ ▐▌    █  ▐▛ ▜▌
// ▐▙   ▐▌ ▐▌▐▌ ▐▌ ▐▌    █  ▐▌ ▐▌
//  █▄▄▌▝█▄█▘▐▌ ▐▌ ▐▌  ▗▄█▄▖▝█▄█▌
//   ▀▀  ▝▀▘ ▝▘ ▝▘ ▝▘  ▝▀▀▀▘ ▞▀▐▌
//                           ▜█▛▘
// ----------------------------------------------------------------

// To make it easier to change the data type later
typedef double scalar;

// NOTE: Assume maximum rank of 8
#define MAX_RANK 8
typedef long ShapeType[MAX_RANK];

// ----------------------------------------------------------------
// ▗▄▄▖                       ▄▄       ▄▄▄           ▗▄▖
// ▐▛▀▜▌                     █▀▀▌      ▀█▀           ▝▜▌
// ▐▌ ▐▌ ▟██▖ █▟█▌ ▟█▙      ▐▛          █  ▐█▙█▖▐▙█▙  ▐▌
// ▐███  ▘▄▟▌ █▘  ▐▙▄▟▌     ▐▌          █  ▐▌█▐▌▐▛ ▜▌ ▐▌
// ▐▌ ▐▌▗█▀▜▌ █   ▐▛▀▀▘     ▐▙          █  ▐▌█▐▌▐▌ ▐▌ ▐▌
// ▐▙▄▟▌▐▙▄█▌ █   ▝█▄▄▌      █▄▄▌      ▄█▄ ▐▌█▐▌▐█▄█▘ ▐▙▄
// ▝▀▀▀  ▀▀▝▘ ▀    ▝▀▀        ▀▀       ▀▀▀ ▝▘▀▝▘▐▌▀▘   ▀▀
//                                              ▐▌
// ----------------------------------------------------------------

// Data structure for a tensor or single tensor item
// For a single tensor item
// - numel = 1
// - ndim = 0
// - shape = []
// - strides = ()
// - data: pointer to item (TODO: just store in pointer?)
// For a vector (1D tensor)
// - numel = length of data
// - ndim = 1
// - shape = [length of data]
// - strides = (1,)
// - data: pointer to data
typedef struct
{
    long numel;        // Number of elements (length of data array)
    long ndim;         // Number of dimensions
    ShapeType shape;   // Shape of tensor
    ShapeType strides; // Strides of tensor
    scalar *data;      // Tensor data
} BareTensorData;

static int BareTensorData_init(BareTensorData *td, ShapeType shape)
{
    td->numel = 1;
    td->ndim = 0;

    // NOTE: not necessary, but good for self-documentation
    memset(td->shape, 0, sizeof(ShapeType));
    memset(td->strides, 0, sizeof(ShapeType));

    long stride = td->numel;
    for (long i = 0; i < MAX_RANK; i++)
    {
        long dim = shape[i];
        if (dim == 0)
        {
            break;
        }
        td->numel *= dim;
        td->ndim += 1;
        td->shape[i] = dim;

        stride /= dim;
        td->strides[i] = stride;
    }

    td->data = malloc(td->numel * sizeof(scalar));

    return td->data == NULL ? -1 : 0;
}

void BareTensorData_dealloc(BareTensorData *td)
{
    free(td->data);
    td->data = NULL;
    td->numel = 0;
    td->ndim = 0;
    memset(td->shape, 0, sizeof(ShapeType));
    memset(td->strides, 0, sizeof(ShapeType));
}

long BareTensorData_indices_to_index(BareTensorData *td, ShapeType indices)
{
    long index = 0;
    for (long i = 0; i < td->ndim; i++)
    {
        index += indices[i] * td->strides[i];
    }
    return index;
}

void print_vector(scalar *v, long n)
{
    for (long i = 0; i < n; i++)
    {
        const char *sep = i < n - 1 ? ", " : "";
        printf("%f%s", v[i], sep);
    }
}

void print_tensor(BareTensorData *td, long current_dim, ShapeType indices)
{
    printf("[");
    if (current_dim == td->ndim - 1)
    {
        long index = BareTensorData_indices_to_index(td, indices);
        print_vector(&td->data[index], td->shape[current_dim]);
    }
    else
    {
        for (long dim = 0; dim < td->shape[current_dim]; dim++)
        {
            ShapeType new_indices = {0};
            memcpy(new_indices, indices, sizeof(ShapeType));
            new_indices[current_dim] = dim;
            print_tensor(td, current_dim + 1, new_indices);
        }
    }
    printf("] ");
}

// TODO: change to accept output pointer?
void BareTensorData_stringify(BareTensorData *td)
{
    ShapeType indices = {0};
    print_tensor(td, 0, indices);
}

// methods
// - item
// - reshape
// - broadcast, unbroadcast
// - sum, mean, max, min, etc.
// - sigmoid, relu, etc.
// - add, sub, mul, div, etc.
// - matmul, dot, etc.
// - in-place operators

// TODO: add should delegate to appropriate version based on shapes and types
void add(scalar *a, scalar *b, scalar *out, long n)
{
    for (long i = 0; i < n; i++)
    {
        out[i] = a[i] + b[i];
    }
}

// ----------------------------------------------------------------
// ▗▄ ▄▖          ▗▖
// ▐█ █▌          ▐▌
// ▐███▌ ▟█▙ ▐█▙█▖▐▙█▙  ▟█▙  █▟█▌▗▟██▖
// ▐▌█▐▌▐▙▄▟▌▐▌█▐▌▐▛ ▜▌▐▙▄▟▌ █▘  ▐▙▄▖▘
// ▐▌▀▐▌▐▛▀▀▘▐▌█▐▌▐▌ ▐▌▐▛▀▀▘ █    ▀▀█▖
// ▐▌ ▐▌▝█▄▄▌▐▌█▐▌▐█▄█▘▝█▄▄▌ █   ▐▄▄▟▌
// ▝▘ ▝▘ ▝▀▀ ▝▘▀▝▘▝▘▀▘  ▝▀▀  ▀    ▀▀▀
// ----------------------------------------------------------------

// clang-format off
typedef struct
{
    PyObject_HEAD
    long ndim;
    PyObject *shape;
    BareTensorData td;
} TDMembers;
// clang-format on

static PyMemberDef TensorData_members[] = {
    {"ndim", Py_T_LONG, offsetof(TDMembers, ndim), Py_READONLY, "TODO: docs"},
    {"shape", Py_T_OBJECT_EX, offsetof(TDMembers, shape), Py_READONLY, "TODO: docs"},
    {NULL} /* Sentinel */
};

// ----------------------------------------------------------------
// ▗▄ ▄▖          ▗▖           ▗▖
// ▐█ █▌      ▐▌  ▐▌           ▐▌
// ▐███▌ ▟█▙ ▐███ ▐▙██▖ ▟█▙  ▟█▟▌▗▟██▖
// ▐▌█▐▌▐▙▄▟▌ ▐▌  ▐▛ ▐▌▐▛ ▜▌▐▛ ▜▌▐▙▄▖▘
// ▐▌▀▐▌▐▛▀▀▘ ▐▌  ▐▌ ▐▌▐▌ ▐▌▐▌ ▐▌ ▀▀█▖
// ▐▌ ▐▌▝█▄▄▌ ▐▙▄ ▐▌ ▐▌▝█▄█▘▝█▄█▌▐▄▄▟▌
// ▝▘ ▝▘ ▝▀▀   ▀▀ ▝▘ ▝▘ ▝▀▘  ▝▀▝▘ ▀▀▀
// ----------------------------------------------------------------

// TODO: methods
// add, etc.

static PyObject *TensorData_numel(TDMembers *self, PyObject *Py_UNUSED(ignored))
{
    return PyLong_FromLong(self->td.numel);
}

static PyObject *TensorData_stride(TDMembers *self, PyObject *Py_UNUSED(ignored))
{
    PyObject *stride = PyTuple_New(self->td.ndim);

    for (long i = 0; i < self->td.ndim; i++)
    {
        PyTuple_SetItem(stride, i, PyLong_FromLong(self->td.strides[i]));
    }

    return stride;
}

static PyObject *TensorData_print(TDMembers *self, PyObject *Py_UNUSED(ignored))
{
    BareTensorData_stringify(&self->td);
    Py_RETURN_NONE;
}

static PyMethodDef TensorData_methods[] = {
    {"numel", (PyCFunction)TensorData_numel, METH_NOARGS, "TODO: docs"},
    {"stride", (PyCFunction)TensorData_stride, METH_NOARGS, "TODO: docs"},
    {"print", (PyCFunction)TensorData_print, METH_NOARGS, "TODO: docs"},
    {NULL} /* Sentinel */
};

// ----------------------------------------------------------------
//  ▗▄▖ ▗▖     █
//  █▀█ ▐▌     ▀             ▐▌
// ▐▌ ▐▌▐▙█▙  ██   ▟█▙  ▟██▖▐███
// ▐▌ ▐▌▐▛ ▜▌  █  ▐▙▄▟▌▐▛  ▘ ▐▌
// ▐▌ ▐▌▐▌ ▐▌  █  ▐▛▀▀▘▐▌    ▐▌
//  █▄█ ▐█▄█▘  █  ▝█▄▄▌▝█▄▄▌ ▐▙▄
//  ▝▀▘ ▝▘▀▘   █   ▝▀▀  ▝▀▀   ▀▀
//           ▐█▛
// ----------------------------------------------------------------

static int TensorData_init(TDMembers *self, PyObject *args, PyObject *kwds)
{
    // Parse args as tuple of dimensions (or tuple of tuple of dimensions)
    Py_ssize_t tuple_len = PyTuple_Size(args);

    if (tuple_len == 0)
    {
        PyErr_SetString(PyExc_ValueError, "Tensor must have at least one value.");
        return -1;
    }

    if (tuple_len > MAX_RANK)
    {
        PyErr_SetString(PyExc_ValueError, "Tensor rank exceeds maximum allowed.");
        return -1;
    }

    self->ndim = tuple_len;
    self->shape = PyTuple_New(tuple_len);

    ShapeType td_shape = {0};

    for (long i = 0; i < tuple_len; i++)
    {
        PyObject *item = PyTuple_GetItem(args, i);
        td_shape[i] = PyLong_AsLong(item);

        if (!PyLong_Check(item))
        {
            PyErr_SetString(PyExc_ValueError, "Tensor dimensions must be integers.");
            return -1;
        }

        if (PyTuple_SetItem(self->shape, i, item))
        {
            PyErr_SetString(PyExc_RuntimeError, "Failed to set shape item.");
            return -1;
        }
    }

    if (BareTensorData_init(&self->td, td_shape))
    {
        PyErr_SetString(PyExc_RuntimeError, "Failed to initialize tensor data.");
        return -1;
    }

    return 0;
}

static void TensorData_dealloc(TDMembers *self)
{
    Py_DECREF(self->shape);
    BareTensorData_dealloc(&self->td);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

// TODO
// static PyObject *
// TensorData_str(TDMembers *obj)
// {
//     // TODO: call BareTensorData_stringify (how to handle allocation?)
//     // return PyUnicode_FromFormat("Stringified_newdatatype{{size:%d}}",
//     //                             obj->obj_UnderlyingDatatypePtr->size);
// }

// clang-format off
// NOTE: disabling formatter due to PyVarObject_HEAD_INIT macro
static PyTypeObject TensorData = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "match.TensorData",
    .tp_doc = PyDoc_STR("TODO: docs"),
    // .tp_repr = TensorData_str,
    // .tp_str = TensorData_str,
    .tp_basicsize = sizeof(TDMembers),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)TensorData_init,
    .tp_dealloc = (destructor)TensorData_dealloc,
    .tp_members = TensorData_members,
    .tp_methods = TensorData_methods,
    // TODO: .tp_getset = TensorData_getset,
    // TODO: .tp_getattr/tp_setattr?
    // TODO: .tp_getattro/tp_setattro?
    // TODO: .tp_richcompare?
    // TODO: .tp_iter/tp_iternext?
};
// clang-format on

// ----------------------------------------------------------------
// ▗▄ ▄▖        ▗▖     ▗▄▖
// ▐█ █▌        ▐▌     ▝▜▌
// ▐███▌ ▟█▙  ▟█▟▌▐▌ ▐▌ ▐▌   ▟█▙
// ▐▌█▐▌▐▛ ▜▌▐▛ ▜▌▐▌ ▐▌ ▐▌  ▐▙▄▟▌
// ▐▌▀▐▌▐▌ ▐▌▐▌ ▐▌▐▌ ▐▌ ▐▌  ▐▛▀▀▘
// ▐▌ ▐▌▝█▄█▘▝█▄█▌▐▙▄█▌ ▐▙▄ ▝█▄▄▌
// ▝▘ ▝▘ ▝▀▘  ▝▀▝▘ ▀▀▝▘  ▀▀  ▝▀▀
// ----------------------------------------------------------------

static PyModuleDef tensordata = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "match",
    .m_doc = "TODO: docs",
    .m_size = -1,
};

PyMODINIT_FUNC
PyInit_tensordata(void)
{
    PyObject *m;

    if (PyType_Ready(&TensorData) < 0)
        return NULL;

    m = PyModule_Create(&tensordata);
    if (m == NULL)
        return NULL;

    Py_INCREF(&TensorData);
    if (PyModule_AddObject(m, "TensorData", (PyObject *)&TensorData) < 0)
    {
        Py_DECREF(&TensorData);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
