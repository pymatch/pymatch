#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stddef.h>

#include "tensorbase.c"

// https://getcode.substack.com/p/fun-and-hackable-tensors-in-rust
// https://jessicastringham.net/2017/12/31/stride-tricks/
// https://github.com/abeschneider/tensor/blob/master/include/tensor_ops.hpp
// https://github.com/kurtschelfthout/tensorken
// https://www.google.com/search?q=as%20strided%20broadcasting&ie=utf-8&oe=utf-8&client=firefox-b-1-m
// https://www.google.com/search?client=firefox-b-1-m&sca_esv=da8adb7374804231&q=as+strided+convolution&oq=as+strided+convolution&aqs=heirloom-srp..

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
// ▗▄▄▖           ▗▄▄▄▖                         ▗▄▄▖
// ▐▛▀▜▖          ▝▀█▀▘                         ▐▛▀▜▌
// ▐▌ ▐▌▝█ █▌       █   ▟█▙ ▐▙██▖▗▟██▖ ▟█▙  █▟█▌▐▌ ▐▌ ▟██▖▗▟██▖ ▟█▙
// ▐██▛  █▖█        █  ▐▙▄▟▌▐▛ ▐▌▐▙▄▖▘▐▛ ▜▌ █▘  ▐███  ▘▄▟▌▐▙▄▖▘▐▙▄▟▌
// ▐▌    ▐█▛        █  ▐▛▀▀▘▐▌ ▐▌ ▀▀█▖▐▌ ▐▌ █   ▐▌ ▐▌▗█▀▜▌ ▀▀█▖▐▛▀▀▘
// ▐▌     █▌        █  ▝█▄▄▌▐▌ ▐▌▐▄▄▟▌▝█▄█▘ █   ▐▙▄▟▌▐▙▄█▌▐▄▄▟▌▝█▄▄▌
// ▝▘     █         ▀   ▝▀▀ ▝▘ ▝▘ ▀▀▀  ▝▀▘  ▀   ▝▀▀▀  ▀▀▝▘ ▀▀▀  ▝▀▀
//       █▌
// ----------------------------------------------------------------

//
// A wrapper around TensorBase enabling use as a Python object.
// We try to match the pytorch tensor API as closely as possible.
//

// clang-format off
typedef struct
{
    PyObject_HEAD
    TensorBase tb;
} PyTensorBase;
// clang-format on

// ----------------------------------------------------------------
// ▗▄▄▖                                 █
// ▐▛▀▜▖                          ▐▌    ▀
// ▐▌ ▐▌ █▟█▌ ▟█▙ ▐▙█▙  ▟█▙  █▟█▌▐███  ██   ▟█▙ ▗▟██▖
// ▐██▛  █▘  ▐▛ ▜▌▐▛ ▜▌▐▙▄▟▌ █▘   ▐▌    █  ▐▙▄▟▌▐▙▄▖▘
// ▐▌    █   ▐▌ ▐▌▐▌ ▐▌▐▛▀▀▘ █    ▐▌    █  ▐▛▀▀▘ ▀▀█▖
// ▐▌    █   ▝█▄█▘▐█▄█▘▝█▄▄▌ █    ▐▙▄ ▗▄█▄▖▝█▄▄▌▐▄▄▟▌
// ▝▘    ▀    ▝▀▘ ▐▌▀▘  ▝▀▀  ▀     ▀▀ ▝▀▀▀▘ ▝▀▀  ▀▀▀
//                ▐▌
// ----------------------------------------------------------------

static PyObject *PyTensorBase_get_ndim(PyTensorBase *self, PyObject *Py_UNUSED(ignored))
{
    return PyLong_FromLong(self->tb.ndim);
}

static PyObject *PyTensorBase_get_shape(PyTensorBase *self, PyObject *Py_UNUSED(ignored))
{
    PyObject *shape = PyTuple_New(self->tb.ndim);

    for (long i = 0; i < self->tb.ndim; i++)
    {
        if (PyTuple_SetItem(shape, i, PyLong_FromLong(self->tb.shape[i])))
        {
            PyErr_SetString(PyExc_RuntimeError, "Failed to set shape item.");
            return NULL;
        }
    }

    return shape;
}

static PyGetSetDef PyTensorBase_getset[] = {
    {"ndim", (getter)PyTensorBase_get_ndim, NULL, "TODO: docs", NULL},
    {"shape", (getter)PyTensorBase_get_shape, NULL, "TODO: docs", NULL},
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

static PyObject *PyTensorBase_numel(PyTensorBase *self, PyObject *Py_UNUSED(ignored))
{
    return PyLong_FromLong(self->tb.numel);
}

static PyObject *PyTensorBase_stride(PyTensorBase *self, PyObject *Py_UNUSED(ignored))
{
    PyObject *stride = PyTuple_New(self->tb.ndim);

    for (long i = 0; i < self->tb.ndim; i++)
    {
        if (PyTuple_SetItem(stride, i, PyLong_FromLong(self->tb.strides[i])))
        {
            PyErr_SetString(PyExc_RuntimeError, "Failed to set shape item.");
            return NULL;
        }
    }

    return stride;
}

// TODO: eventually remove this
static PyObject *PyTensorBase_print(PyTensorBase *self, PyObject *Py_UNUSED(ignored))
{
    TensorBase_stringify(&self->tb);
    Py_RETURN_NONE;
}

static PyMethodDef PyTensorBase_methods[] = {
    {"numel", (PyCFunction)PyTensorBase_numel, METH_NOARGS, "TODO: docs"},
    {"stride", (PyCFunction)PyTensorBase_stride, METH_NOARGS, "TODO: docs"},
    {"print", (PyCFunction)PyTensorBase_print, METH_NOARGS, "TODO: docs"},
    {NULL} /* Sentinel */
};

// ----------------------------------------------------------------
// ▗▄ ▗▖          ▗▖                  ▗▄▄▖                               ▗▄▖
// ▐█ ▐▌          ▐▌                  ▐▛▀▜▖           ▐▌                 ▝▜▌
// ▐▛▌▐▌▐▌ ▐▌▐█▙█▖▐▙█▙  ▟█▙  █▟█▌     ▐▌ ▐▌ █▟█▌ ▟█▙ ▐███  ▟█▙  ▟██▖ ▟█▙  ▐▌
// ▐▌█▐▌▐▌ ▐▌▐▌█▐▌▐▛ ▜▌▐▙▄▟▌ █▘       ▐██▛  █▘  ▐▛ ▜▌ ▐▌  ▐▛ ▜▌▐▛  ▘▐▛ ▜▌ ▐▌
// ▐▌▐▟▌▐▌ ▐▌▐▌█▐▌▐▌ ▐▌▐▛▀▀▘ █        ▐▌    █   ▐▌ ▐▌ ▐▌  ▐▌ ▐▌▐▌   ▐▌ ▐▌ ▐▌
// ▐▌ █▌▐▙▄█▌▐▌█▐▌▐█▄█▘▝█▄▄▌ █        ▐▌    █   ▝█▄█▘ ▐▙▄ ▝█▄█▘▝█▄▄▌▝█▄█▘ ▐▙▄
// ▝▘ ▀▘ ▀▀▝▘▝▘▀▝▘▝▘▀▘  ▝▀▀  ▀        ▝▘    ▀    ▝▀▘   ▀▀  ▝▀▘  ▝▀▀  ▝▀▘   ▀▀
// ----------------------------------------------------------------

// These methods are implemented below
// TODO: move implementations here?
static PyObject *PyTensorBase_add(PyObject *self, PyObject *other);

static PyNumberMethods PyTensorBase_as_number = {
    .nb_add = (binaryfunc)PyTensorBase_add,
    // .nb_subtract = (binaryfunc)PyTensorBase_as_number_subtract,
    // .nb_multiply = (binaryfunc)PyTensorBase_as_number_multiply,
    // .nb_true_divide = (binaryfunc)PyTensorBase_as_number_true_divide,
    // .nb_floor_divide = (binaryfunc)PyTensorBase_as_number_floor_divide,
    // .nb_remainder = (binaryfunc)PyTensorBase_as_number_remainder,
    // .nb_power = (ternaryfunc)PyTensorBase_as_number_power,
    // .nb_negative = (unaryfunc)PyTensorBase_as_number_negative,
    // .nb_positive = (unaryfunc)PyTensorBase_as_number_positive,
    // .nb_absolute = (unaryfunc)PyTensorBase_as_number_absolute,
    // .nb_invert = (unaryfunc)PyTensorBase_as_number_invert,
    // .nb_lshift = (binaryfunc)PyTensorBase_as_number_lshift,
    // .nb_rshift = (binaryfunc)PyTensorBase_as_number_rshift,
    // .nb_and = (binaryfunc)PyTensorBase_as_number_and,
    // .nb_xor = (binaryfunc)PyTensorBase_as_number_xor,
    // .nb_or = (binaryfunc)PyTensorBase_as_number_or,
    // .nb_int = (unaryfunc)PyTensorBase_as_number_int,
    // .nb_float = (unaryfunc)PyTensorBase_as_number_float,
    // .nb_inplace_add = (binaryfunc)PyTensorBase_as_number_inplace_add,
    // .nb_inplace_subtract = (binaryfunc)PyTensorBase_as_number_inplace_subtract,
    // .nb_inplace_multiply = (binaryfunc)PyTensorBase_as_number_inplace_multiply,
    // .nb_inplace_true_divide = (binaryfunc)PyTensorBase_as_number_inplace_true_divide,
    // .nb_inplace_floor_divide = (binaryfunc)PyTensorBase_as_number_inplace_floor_divide,
    // .nb_inplace_remainder = (binaryfunc)PyTensorBase_as_number_inplace_remainder,
    // .nb_inplace_power = (ternaryfunc)PyTensorBase_as_number_inplace_power,
    // .nb_inplace_lshift = (binaryfunc)PyTensorBase_as_number_inplace_lshift,
    // .nb_inplace_rshift = (binaryfunc
};

// ----------------------------------------------------------------
// ▗▄▄▖            ▗▄▖ ▗▖     █                      ▗▄▄         ▄▄
// ▐▛▀▜▖           █▀█ ▐▌     ▀             ▐▌       ▐▛▀█       ▐▛▀
// ▐▌ ▐▌▝█ █▌     ▐▌ ▐▌▐▙█▙  ██   ▟█▙  ▟██▖▐███      ▐▌ ▐▌ ▟█▙ ▐███
// ▐██▛  █▖█      ▐▌ ▐▌▐▛ ▜▌  █  ▐▙▄▟▌▐▛  ▘ ▐▌       ▐▌ ▐▌▐▙▄▟▌ ▐▌
// ▐▌    ▐█▛      ▐▌ ▐▌▐▌ ▐▌  █  ▐▛▀▀▘▐▌    ▐▌       ▐▌ ▐▌▐▛▀▀▘ ▐▌
// ▐▌     █▌       █▄█ ▐█▄█▘  █  ▝█▄▄▌▝█▄▄▌ ▐▙▄      ▐▙▄█ ▝█▄▄▌ ▐▌
// ▝▘     █        ▝▀▘ ▝▘▀▘   █   ▝▀▀  ▝▀▀   ▀▀      ▝▀▀   ▝▀▀  ▝▘
//       █▌                 ▐█▛
// ----------------------------------------------------------------

static int PyTensorBase_init(PyTensorBase *self, PyObject *args, PyObject *kwds)
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

    if (TensorBase_init(&self->tb, td_shape) < 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "Failed to initialize tensor data.");
        return -1;
    }

    return 0;
}

static void PyTensorBase_dealloc(PyTensorBase *self)
{
    TensorBase_dealloc(&self->tb);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

// TODO
// static PyObject *
// PyTensorBase_str(PyTensorBase *obj)
// {
//     // TODO: call TensorBase_stringify (how to handle allocation?)
//     // return PyUnicode_FromFormat("Stringified_newdatatype{{size:%d}}",
//     //                             obj->obj_UnderlyingDatatypePtr->size);
// }

// clang-format off
// NOTE: disabling formatter due to PyVarObject_HEAD_INIT macro
static PyTypeObject PyTensorBaseType = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "match.tensorbase.TensorBase",
    .tp_doc = PyDoc_STR("TODO: docs"),
    // .tp_repr = PyTensorBase_str,
    // .tp_str = PyTensorBase_str,
    .tp_basicsize = sizeof(PyTensorBase),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)PyTensorBase_init,
    .tp_dealloc = (destructor)PyTensorBase_dealloc,
    // .tp_members = PyTensorBase_members,
    .tp_methods = PyTensorBase_methods,
    .tp_getset = PyTensorBase_getset,
    // TODO: .tp_getattr/tp_setattr?
    // TODO: .tp_getattro/tp_setattro?
    // TODO: .tp_richcompare?
    // TODO: .tp_iter/tp_iternext?
    .tp_as_number = &PyTensorBase_as_number,
};
// clang-format on

// ----------------------------------------------------------------
// ▗▖ ▗▖       █  ▗▄▖    █         █
// ▐▌ ▐▌ ▐▌    ▀  ▝▜▌    ▀   ▐▌    ▀
// ▐▌ ▐▌▐███  ██   ▐▌   ██  ▐███  ██   ▟█▙ ▗▟██▖
// ▐▌ ▐▌ ▐▌    █   ▐▌    █   ▐▌    █  ▐▙▄▟▌▐▙▄▖▘
// ▐▌ ▐▌ ▐▌    █   ▐▌    █   ▐▌    █  ▐▛▀▀▘ ▀▀█▖
// ▝█▄█▘ ▐▙▄ ▗▄█▄▖ ▐▙▄ ▗▄█▄▖ ▐▙▄ ▗▄█▄▖▝█▄▄▌▐▄▄▟▌
//  ▝▀▘   ▀▀ ▝▀▀▀▘  ▀▀ ▝▀▀▀▘  ▀▀ ▝▀▀▀▘ ▝▀▀  ▀▀▀
// ----------------------------------------------------------------

static long PyTensorBase_Check(PyObject *obj)
{
    return PyObject_IsInstance(obj, (PyObject *)&PyTensorBaseType);
}

static long PyFloatOrLong_Check(PyObject *obj)
{
    return PyLong_Check(obj) || PyFloat_Check(obj);
}

static long can_math(PyObject *obj)
{
    return PyFloatOrLong_Check(obj) || PyTensorBase_Check(obj);
}

static scalar PyFloatOrLong_asDouble(PyObject *obj)
{
    if (PyLong_Check(obj))
    {
        return PyLong_AsDouble(obj);
    }
    return PyFloat_AsDouble(obj);
}

static PyTensorBase *PyTensorBase_create(ShapeArray shape)
{
    PyTensorBase *result = (PyTensorBase *)PyObject_New(PyTensorBase, &PyTensorBaseType);
    if (result == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create new TensorBase object.");
        return NULL;
    }

    if (TensorBase_init(&result->tb, shape) < 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "Failed to initialize tensor base.");
        return NULL;
    }

    return result;
}

// static PyTensorBase *PyTensorBase_shallow_broadcast(PyTensorBase *t, ShapeArray shape)
// {
//     PyTensorBase *result = (PyTensorBase *)PyObject_New(PyTensorBase, &PyTensorBaseType);
//     if (result == NULL)
//     {
//         PyErr_SetString(PyExc_RuntimeError, "Failed to create new PyTensorBase object.");
//         return NULL;
//     }

//     result->tb = t->tb;

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

static PyObject *PyTensorBase_add_tensor_scalar(PyTensorBase *t, scalar s)
{
    PyTensorBase *result = PyTensorBase_create(t->tb.shape);
    if (!result)
    {
        // NOTE: error string set in PyTensorBase_create
        return NULL;
    }

    TensorBase_add_tensor_scalar(&t->tb, s, &result->tb);
    return (PyObject *)result;
}

static PyObject *PyTensorBase_add_tensor_tensor(PyTensorBase *a, PyTensorBase *b)
{
    PyTensorBase *result = PyTensorBase_create(a->tb.shape);
    if (!result)
    {
        // NOTE: error string set in PyTensorBase_create
        return NULL;
    }

    TensorBase a_temp;
    TensorBase b_temp;

    if (TensorBase_broadcast_for_binop(&a->tb, &b->tb, &a_temp, &b_temp) < 0)
    {
        PyErr_SetString(PyExc_ValueError, "Incompatible shapes for addition.");
        return NULL;
    }

    TensorBase_add_tensor_tensor(&a_temp, &b_temp, &result->tb);
    return (PyObject *)result;
}

static PyObject *PyTensorBase_add(PyObject *a, PyObject *b)
{
    // Valid types: PyTensorBase (with broadcastable dimensions), integers, floats
    // TODO: just get types and compare?

    if (!(can_math(a) && can_math(b)))
    {
        PyErr_SetString(PyExc_ValueError, "Invalid types for addition.");
        return NULL;
    }

    // PyTensorBase + (Long | Float)
    if (PyTensorBase_Check(a) && PyFloatOrLong_Check(b))
    {
        return PyTensorBase_add_tensor_scalar((PyTensorBase *)a, PyFloatOrLong_asDouble(b));
    }
    // (Long | Float) + PyTensorBase
    else if (PyFloatOrLong_Check(a) && PyTensorBase_Check(b))
    {
        return PyTensorBase_add_tensor_scalar((PyTensorBase *)b, PyFloatOrLong_asDouble(a));
    }
    // PyTensorBase + PyTensorBase
    else if (PyTensorBase_Check(a) && PyTensorBase_Check(b))
    {
        return PyTensorBase_add_tensor_tensor((PyTensorBase *)a, (PyTensorBase *)b);
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

static PyObject *PyTensorBase_ones(PyTensorBase *self, PyObject *args)
{
    // TODO: should probably use create
    if (!PyTensorBase_init(self, args, NULL))
    {
        return NULL;
    }

    for (long i = 0; i < self->tb.numel; i++)
    {
        self->tb.data[i] = 1;
    }

    return self;
}

static PyMethodDef PyTensorBase_functions[] = {
    {"ones", (PyCFunction)PyTensorBase_ones, METH_NOARGS, "TODO: docs"},
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

static PyModuleDef TensorBaseModule = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "match.tensorbase",
    .m_doc = PyDoc_STR("TODO: docs"),
    .m_size = -1,
    .m_methods = PyTensorBase_functions,
};

PyMODINIT_FUNC
PyInit_tensorbase(void)
{

    if (PyType_Ready(&PyTensorBaseType) < 0)
        return NULL;

    PyObject *m = PyModule_Create(&TensorBaseModule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&PyTensorBaseType);
    if (PyModule_AddObject(m, "TensorBase", (PyObject *)&PyTensorBaseType) < 0)
    {
        Py_DECREF(&PyTensorBaseType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
