#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stddef.h>

#include "tensorbase.c"

// TODO:
// - what to do about data types? (void * everywhere?)
// - allow use as base class?
// - iterable
// - implement buffer protocol?
//      - http://jakevdp.github.io/blog/2014/05/05/introduction-to-the-python-buffer-protocol/
//      - https://docs.python.org/3/c-api/buffer.html
// - implement number protocol
// - implement sequence protocol
// - https://docs.python.org/3/c-api/structures.html#c.METH_FASTCALL
//  Py_TPFLAGS_CHECKTYPES
// - indexing and slicing
// - multiple tensors can share the same data
// - squeeze, reshape, resize, transpose, etc.
// - len, getitem, setitem
// - handle case of single item tensors (ndim = 0)
// - be consistent about taking ptr or value

// TODO: useful macros
// - Py_ALWAYS_INLINE
// - Py_MAX(x, y)
// - Py_MIN(x, y)
// - Py_STRINGIFY(x)
// - PyDoc_STRVAR(name, str)
// - PyDoc_STR(str)

// ----------------------------------------------------------------
// ▗▄▄▖           ▗▄▄▄▖                         ▗▄▄
// ▐▛▀▜▖          ▝▀█▀▘                         ▐▛▀█       ▐▌
// ▐▌ ▐▌▝█ █▌       █   ▟█▙ ▐▙██▖▗▟██▖ ▟█▙  █▟█▌▐▌ ▐▌ ▟██▖▐███  ▟██▖
// ▐██▛  █▖█        █  ▐▙▄▟▌▐▛ ▐▌▐▙▄▖▘▐▛ ▜▌ █▘  ▐▌ ▐▌ ▘▄▟▌ ▐▌   ▘▄▟▌
// ▐▌    ▐█▛        █  ▐▛▀▀▘▐▌ ▐▌ ▀▀█▖▐▌ ▐▌ █   ▐▌ ▐▌▗█▀▜▌ ▐▌  ▗█▀▜▌
// ▐▌     █▌        █  ▝█▄▄▌▐▌ ▐▌▐▄▄▟▌▝█▄█▘ █   ▐▙▄█ ▐▙▄█▌ ▐▙▄ ▐▙▄█▌
// ▝▘     █         ▀   ▝▀▀ ▝▘ ▝▘ ▀▀▀  ▝▀▘  ▀   ▝▀▀   ▀▀▝▘  ▀▀  ▀▀▝▘
//       █▌
// ----------------------------------------------------------------

// clang-format off
typedef struct
{
    PyObject_HEAD
    TensorBase td;
} TensorData;
// clang-format on

static PyObject *TensorData_get_ndim(TensorData *self, PyObject *Py_UNUSED(ignored))
{
    return PyLong_FromLong(self->td.ndim);
}

static PyObject *TensorData_get_shape(TensorData *self, PyObject *Py_UNUSED(ignored))
{
    PyObject *shape = PyTuple_New(self->td.ndim);

    for (long i = 0; i < self->td.ndim; i++)
    {
        if (PyTuple_SetItem(shape, i, PyLong_FromLong(self->td.shape[i])))
        {
            PyErr_SetString(PyExc_RuntimeError, "Failed to set shape item.");
            return NULL;
        }
    }

    return shape;
}

static PyGetSetDef TensorData_getset[] = {
    {"ndim", (getter)TensorData_get_ndim, NULL, "TODO: docs", NULL},
    {"shape", (getter)TensorData_get_shape, NULL, "TODO: docs", NULL},
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

static PyObject *TensorData_numel(TensorData *self, PyObject *Py_UNUSED(ignored))
{
    return PyLong_FromLong(self->td.numel);
}

static PyObject *TensorData_stride(TensorData *self, PyObject *Py_UNUSED(ignored))
{
    PyObject *stride = PyTuple_New(self->td.ndim);

    for (long i = 0; i < self->td.ndim; i++)
    {
        if (PyTuple_SetItem(stride, i, PyLong_FromLong(self->td.strides[i])))
        {
            PyErr_SetString(PyExc_RuntimeError, "Failed to set shape item.");
            return NULL;
        }
    }

    return stride;
}

// TODO: eventually remove this
static PyObject *TensorData_print(TensorData *self, PyObject *Py_UNUSED(ignored))
{
    TensorBase_stringify(&self->td);
    Py_RETURN_NONE;
}

static PyMethodDef TensorData_methods[] = {
    {"numel", (PyCFunction)TensorData_numel, METH_NOARGS, "TODO: docs"},
    {"stride", (PyCFunction)TensorData_stride, METH_NOARGS, "TODO: docs"},
    {"print", (PyCFunction)TensorData_print, METH_NOARGS, "TODO: docs"},
    {NULL} /* Sentinel */
};

// ----------------------------------------------------------------
// ▗▄ ▗▖          ▗▖
// ▐█ ▐▌          ▐▌
// ▐▛▌▐▌▐▌ ▐▌▐█▙█▖▐▙█▙  ▟█▙  █▟█▌
// ▐▌█▐▌▐▌ ▐▌▐▌█▐▌▐▛ ▜▌▐▙▄▟▌ █▘
// ▐▌▐▟▌▐▌ ▐▌▐▌█▐▌▐▌ ▐▌▐▛▀▀▘ █
// ▐▌ █▌▐▙▄█▌▐▌█▐▌▐█▄█▘▝█▄▄▌ █
// ▝▘ ▀▘ ▀▀▝▘▝▘▀▝▘▝▘▀▘  ▝▀▀  ▀
// ----------------------------------------------------------------

// Implementation of Number Protocol

static PyObject *TensorData_add(PyObject *self, PyObject *other);

static PyNumberMethods TensorData_as_number = {
    .nb_add = (binaryfunc)TensorData_add,
    // .nb_subtract = (binaryfunc)TensorData_as_number_subtract,
    // .nb_multiply = (binaryfunc)TensorData_as_number_multiply,
    // .nb_true_divide = (binaryfunc)TensorData_as_number_true_divide,
    // .nb_floor_divide = (binaryfunc)TensorData_as_number_floor_divide,
    // .nb_remainder = (binaryfunc)TensorData_as_number_remainder,
    // .nb_power = (ternaryfunc)TensorData_as_number_power,
    // .nb_negative = (unaryfunc)TensorData_as_number_negative,
    // .nb_positive = (unaryfunc)TensorData_as_number_positive,
    // .nb_absolute = (unaryfunc)TensorData_as_number_absolute,
    // .nb_invert = (unaryfunc)TensorData_as_number_invert,
    // .nb_lshift = (binaryfunc)TensorData_as_number_lshift,
    // .nb_rshift = (binaryfunc)TensorData_as_number_rshift,
    // .nb_and = (binaryfunc)TensorData_as_number_and,
    // .nb_xor = (binaryfunc)TensorData_as_number_xor,
    // .nb_or = (binaryfunc)TensorData_as_number_or,
    // .nb_int = (unaryfunc)TensorData_as_number_int,
    // .nb_float = (unaryfunc)TensorData_as_number_float,
    // .nb_inplace_add = (binaryfunc)TensorData_as_number_inplace_add,
    // .nb_inplace_subtract = (binaryfunc)TensorData_as_number_inplace_subtract,
    // .nb_inplace_multiply = (binaryfunc)TensorData_as_number_inplace_multiply,
    // .nb_inplace_true_divide = (binaryfunc)TensorData_as_number_inplace_true_divide,
    // .nb_inplace_floor_divide = (binaryfunc)TensorData_as_number_inplace_floor_divide,
    // .nb_inplace_remainder = (binaryfunc)TensorData_as_number_inplace_remainder,
    // .nb_inplace_power = (ternaryfunc)TensorData_as_number_inplace_power,
    // .nb_inplace_lshift = (binaryfunc)TensorData_as_number_inplace_lshift,
    // .nb_inplace_rshift = (binaryfunc
};

; // ----------------------------------------------------------------
// ▗▄▄▄▖                         ▗▄▄                      ▗▄▄▄▖
// ▝▀█▀▘                         ▐▛▀█       ▐▌            ▝▀█▀▘
//   █   ▟█▙ ▐▙██▖▗▟██▖ ▟█▙  █▟█▌▐▌ ▐▌ ▟██▖▐███  ▟██▖       █  ▝█ █▌▐▙█▙  ▟█▙
//   █  ▐▙▄▟▌▐▛ ▐▌▐▙▄▖▘▐▛ ▜▌ █▘  ▐▌ ▐▌ ▘▄▟▌ ▐▌   ▘▄▟▌       █   █▖█ ▐▛ ▜▌▐▙▄▟▌
//   █  ▐▛▀▀▘▐▌ ▐▌ ▀▀█▖▐▌ ▐▌ █   ▐▌ ▐▌▗█▀▜▌ ▐▌  ▗█▀▜▌       █   ▐█▛ ▐▌ ▐▌▐▛▀▀▘
//   █  ▝█▄▄▌▐▌ ▐▌▐▄▄▟▌▝█▄█▘ █   ▐▙▄█ ▐▙▄█▌ ▐▙▄ ▐▙▄█▌       █    █▌ ▐█▄█▘▝█▄▄▌
//   ▀   ▝▀▀ ▝▘ ▝▘ ▀▀▀  ▝▀▘  ▀   ▝▀▀   ▀▀▝▘  ▀▀  ▀▀▝▘       ▀    █  ▐▌▀▘  ▝▀▀
//                                                              █▌  ▐▌
// ----------------------------------------------------------------

static int TensorData_init(TensorData *self, PyObject *args, PyObject *kwds);
static void TensorData_dealloc(TensorData *self);

// clang-format off
// NOTE: disabling formatter due to PyVarObject_HEAD_INIT macro
static PyTypeObject TensorDataType = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "match.TensorData",
    .tp_doc = PyDoc_STR("TODO: docs"),
    // .tp_repr = TensorData_str,
    // .tp_str = TensorData_str,
    .tp_basicsize = sizeof(TensorData),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)TensorData_init,
    .tp_dealloc = (destructor)TensorData_dealloc,
    // .tp_members = TensorData_members,
    .tp_methods = TensorData_methods,
    .tp_getset = TensorData_getset,
    // TODO: .tp_getattr/tp_setattr?
    // TODO: .tp_getattro/tp_setattro?
    // TODO: .tp_richcompare?
    // TODO: .tp_iter/tp_iternext?
    .tp_as_number = &TensorData_as_number,
};
// clang-format on

static int TensorData_init(TensorData *self, PyObject *args, PyObject *kwds)
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

    ShapeArray td_shape = {0};

    for (long i = 0; i < tuple_len; i++)
    {
        PyObject *item = PyTuple_GetItem(args, i);
        if (!PyLong_Check(item))
        {
            PyErr_SetString(PyExc_ValueError, "Tensor dimensions must be integers.");
            return -1;
        }

        td_shape[i] = PyLong_AsLong(item);
    }

    if (TensorBase_init(&self->td, td_shape) < 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "Failed to initialize tensor data.");
        return -1;
    }

    return 0;
}

static void TensorData_dealloc(TensorData *self)
{
    TensorBase_dealloc(&self->td);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

// TODO
// static PyObject *
// TensorData_str(TensorData *obj)
// {
//     // TODO: call TensorBase_stringify (how to handle allocation?)
//     // return PyUnicode_FromFormat("Stringified_newdatatype{{size:%d}}",
//     //                             obj->obj_UnderlyingDatatypePtr->size);
// }

// ----------------------------------------------------------------
// ▗▄▄▄▖                              ▗▖ ▗▖       █  ▗▄▖
// ▝▀█▀▘                              ▐▌ ▐▌ ▐▌    ▀  ▝▜▌
//   █   ▟█▙ ▐▙██▖▗▟██▖ ▟█▙  █▟█▌     ▐▌ ▐▌▐███  ██   ▐▌  ▗▟██▖
//   █  ▐▙▄▟▌▐▛ ▐▌▐▙▄▖▘▐▛ ▜▌ █▘       ▐▌ ▐▌ ▐▌    █   ▐▌  ▐▙▄▖▘
//   █  ▐▛▀▀▘▐▌ ▐▌ ▀▀█▖▐▌ ▐▌ █        ▐▌ ▐▌ ▐▌    █   ▐▌   ▀▀█▖
//   █  ▝█▄▄▌▐▌ ▐▌▐▄▄▟▌▝█▄█▘ █        ▝█▄█▘ ▐▙▄ ▗▄█▄▖ ▐▙▄ ▐▄▄▟▌
//   ▀   ▝▀▀ ▝▘ ▝▘ ▀▀▀  ▝▀▘  ▀         ▝▀▘   ▀▀ ▝▀▀▀▘  ▀▀  ▀▀▀
// ----------------------------------------------------------------

static long PyTensorData_Check(PyObject *obj)
{
    return PyObject_IsInstance(obj, (PyObject *)&TensorDataType);
}

static long PyFloatOrLong_Check(PyObject *obj)
{
    return PyLong_Check(obj) || PyFloat_Check(obj);
}

static long can_math(PyObject *obj)
{
    return PyFloatOrLong_Check(obj) || PyTensorData_Check(obj);
}

static scalar PyFloatOrLong_asDouble(PyObject *obj)
{
    if (PyLong_Check(obj))
    {
        return PyLong_AsDouble(obj);
    }
    return PyFloat_AsDouble(obj);
}

static TensorData *TensorData_create(ShapeArray shape)
{
    TensorData *result = (TensorData *)PyObject_New(TensorData, &TensorDataType);
    if (result == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create new TensorData object.");
        return NULL;
    }

    if (TensorBase_init(&result->td, shape) < 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "Failed to initialize tensor data.");
        return NULL;
    }

    return result;
}

// static TensorData *TensorData_shallow_broadcast(TensorData *t, ShapeArray shape)
// {
//     TensorData *result = (TensorData *)PyObject_New(TensorData, &TensorDataType);
//     if (result == NULL)
//     {
//         PyErr_SetString(PyExc_RuntimeError, "Failed to create new TensorData object.");
//         return NULL;
//     }

//     result->td = t->td;

//     // Muck with strides...

//     return result;
// }

// ----------------------------------------------------------------
// ▗▄▄▄▖                              ▗▄ ▄▖          ▗▖
// ▝▀█▀▘                              ▐█ █▌      ▐▌  ▐▌
//   █   ▟█▙ ▐▙██▖▗▟██▖ ▟█▙  █▟█▌     ▐███▌ ▟██▖▐███ ▐▙██▖
//   █  ▐▙▄▟▌▐▛ ▐▌▐▙▄▖▘▐▛ ▜▌ █▘       ▐▌█▐▌ ▘▄▟▌ ▐▌  ▐▛ ▐▌
//   █  ▐▛▀▀▘▐▌ ▐▌ ▀▀█▖▐▌ ▐▌ █        ▐▌▀▐▌▗█▀▜▌ ▐▌  ▐▌ ▐▌
//   █  ▝█▄▄▌▐▌ ▐▌▐▄▄▟▌▝█▄█▘ █        ▐▌ ▐▌▐▙▄█▌ ▐▙▄ ▐▌ ▐▌
//   ▀   ▝▀▀ ▝▘ ▝▘ ▀▀▀  ▝▀▘  ▀        ▝▘ ▝▘ ▀▀▝▘  ▀▀ ▝▘ ▝▘
// ----------------------------------------------------------------

static PyObject *TensorData_add_tensor_scalar(TensorData *t, scalar s)
{
    TensorData *result = TensorData_create(t->td.shape);
    if (!result)
    {
        // NOTE: error string set in TensorData_create
        return NULL;
    }

    TensorBase_add_tensor_scalar(&t->td, s, &result->td);
    return (PyObject *)result;
}

static PyObject *TensorData_add_tensor_tensor(TensorData *a, TensorData *b)
{
    TensorData *result = TensorData_create(a->td.shape);
    if (!result)
    {
        // NOTE: error string set in TensorData_create
        return NULL;
    }

    TensorBase a_temp;
    TensorBase b_temp;

    if (TensorBase_broadcast_for_binop(&a->td, &b->td, &a_temp, &b_temp) < 0)
    {
        PyErr_SetString(PyExc_ValueError, "Incompatible shapes for addition.");
        return NULL;
    }

    TensorBase_add_tensor_tensor(&a_temp, &b_temp, &result->td);
    return (PyObject *)result;
}

static PyObject *TensorData_add(PyObject *a, PyObject *b)
{
    // Valid types: TensorData (with broadcastable dimensions), integers, floats
    // TODO: just get types and compare?

    if (!(can_math(a) && can_math(b)))
    {
        PyErr_SetString(PyExc_ValueError, "Invalid types for addition.");
        return NULL;
    }

    // TensorData + (Long | Float)
    if (PyTensorData_Check(a) && PyFloatOrLong_Check(b))
    {
        return TensorData_add_tensor_scalar((TensorData *)a, PyFloatOrLong_asDouble(b));
    }
    // (Long | Float) + TensorData
    else if (PyFloatOrLong_Check(a) && PyTensorData_Check(b))
    {
        return TensorData_add_tensor_scalar((TensorData *)b, PyFloatOrLong_asDouble(a));
    }
    // TensorData + TensorData
    else if (PyTensorData_Check(a) && PyTensorData_Check(b))
    {
        return TensorData_add_tensor_tensor((TensorData *)a, (TensorData *)b);
    }
    // Else invalid
    else
    {
        PyErr_SetString(PyExc_ValueError, "Invalid types for addition.");
        return NULL;
    }
}

// ----------------------------------------------------------------
// ▗▄▄▄▖                      █
// ▐▛▀▀▘                ▐▌    ▀
// ▐▌   ▐▌ ▐▌▐▙██▖ ▟██▖▐███  ██   ▟█▙ ▐▙██▖▗▟██▖
// ▐███ ▐▌ ▐▌▐▛ ▐▌▐▛  ▘ ▐▌    █  ▐▛ ▜▌▐▛ ▐▌▐▙▄▖▘
// ▐▌   ▐▌ ▐▌▐▌ ▐▌▐▌    ▐▌    █  ▐▌ ▐▌▐▌ ▐▌ ▀▀█▖
// ▐▌   ▐▙▄█▌▐▌ ▐▌▝█▄▄▌ ▐▙▄ ▗▄█▄▖▝█▄█▘▐▌ ▐▌▐▄▄▟▌
// ▝▘    ▀▀▝▘▝▘ ▝▘ ▝▀▀   ▀▀ ▝▀▀▀▘ ▝▀▘ ▝▘ ▝▘ ▀▀▀
// ----------------------------------------------------------------

static PyObject *TensorData_ones(TensorData *self, PyObject *args)
{
    // TODO: should probably use create
    if (!TensorData_init(self, args, NULL))
    {
        return NULL;
    }

    for (long i = 0; i < self->td.numel; i++)
    {
        self->td.data[i] = 1;
    }

    return self;
}

static PyMethodDef TensorData_functions[] = {
    {"ones", (PyCFunction)TensorData_ones, METH_NOARGS, "TODO: docs"},
    {NULL} /* Sentinel */
};

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
    .m_methods = TensorData_functions,
};

PyMODINIT_FUNC
PyInit_tensordata(void)
{
    PyObject *m;

    if (PyType_Ready(&TensorDataType) < 0)
        return NULL;

    m = PyModule_Create(&tensordata);
    if (m == NULL)
        return NULL;

    Py_INCREF(&TensorDataType);
    if (PyModule_AddObject(m, "TensorData", (PyObject *)&TensorDataType) < 0)
    {
        Py_DECREF(&TensorDataType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
