#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include <math.h>

static int double_cmp(const void *a, const void *b)
{
  double da = *(const double *)a;
  double db = *(const double *)b;
  if (da < db)
    return -1;
  if (da > db)
    return 1;
  return 0;
}

static PyObject *py_fmedian(PyObject *self, PyObject *args)
{
  PyObject *in_obj, *out_obj;
  int xsize, ysize, exclude_center;
  if (!PyArg_ParseTuple(args, "OOiii", &in_obj, &out_obj, &xsize, &ysize, &exclude_center))
  {
    return NULL;
  }

  PyArrayObject *in_arr = (PyArrayObject *)PyArray_FROM_OTF(in_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_C_CONTIGUOUS);
  PyArrayObject *out_arr = (PyArrayObject *)PyArray_FROM_OTF(out_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY | NPY_ARRAY_C_CONTIGUOUS);
  if (!in_arr || !out_arr)
  {
    Py_XDECREF(in_arr);
    Py_XDECREF(out_arr);
    PyErr_SetString(PyExc_TypeError, "fmedian: input and output must be numpy arrays of dtype float64");
    return NULL;
  }

  if (PyArray_NDIM(in_arr) != 2 || PyArray_NDIM(out_arr) != 2)
  {
    Py_DECREF(in_arr);
    Py_DECREF(out_arr);
    PyErr_SetString(PyExc_ValueError, "fmedian: arrays must be 2-dimensional");
    return NULL;
  }

  npy_intp *dims = PyArray_DIMS(in_arr);
  npy_intp ny = dims[0];
  npy_intp nx = dims[1];

  npy_intp *odims = PyArray_DIMS(out_arr);
  if (odims[0] != ny || odims[1] != nx)
  {
    Py_DECREF(in_arr);
    Py_DECREF(out_arr);
    PyErr_SetString(PyExc_ValueError, "fmedian: input and output must have the same shape");
    return NULL;
  }

  double *in = (double *)PyArray_DATA(in_arr);
  double *out = (double *)PyArray_DATA(out_arr);

  int wx = 2 * xsize + 1;
  int wy = 2 * ysize + 1;
  int maxbuf = wx * wy;
  double *buf = (double *)malloc(sizeof(double) * maxbuf);
  if (!buf)
  {
    Py_DECREF(in_arr);
    Py_DECREF(out_arr);
    PyErr_NoMemory();
    return NULL;
  }

  for (npy_intp iy = 0; iy < ny; ++iy)
  {
    for (npy_intp ix = 0; ix < nx; ++ix)
    {
      int count = 0;
      for (int dy = -ysize; dy <= ysize; ++dy)
      {
        npy_intp yy = iy + dy;
        if (yy < 0 || yy >= ny)
          continue;
        for (int dx = -xsize; dx <= xsize; ++dx)
        {
          npy_intp xx = ix + dx;
          if (xx < 0 || xx >= nx)
            continue;
          if (exclude_center && dx == 0 && dy == 0)
            continue;
          buf[count++] = in[yy * nx + xx];
        }
      }
      if (count == 0)
      {
        out[iy * nx + ix] = 0.0;
      }
      else
      {
        qsort(buf, count, sizeof(double), double_cmp);
        if (count % 2 == 1)
        {
          out[iy * nx + ix] = buf[count / 2];
        }
        else
        {
          double a = buf[(count / 2) - 1];
          double b = buf[count / 2];
          out[iy * nx + ix] = 0.5 * (a + b);
        }
      }
    }
  }

  free(buf);
  Py_DECREF(in_arr);
  PyArray_ResolveWritebackIfCopy(out_arr);
  Py_DECREF(out_arr);

  Py_RETURN_NONE;
}

static PyMethodDef FmedianMethods[] = {
    {"fmedian", py_fmedian, METH_VARARGS, "Compute filtered median over local window"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef fmedianmodule = {
    PyModuleDef_HEAD_INIT,
    "_fmedian",
    "fmedian C extension",
    -1,
    FmedianMethods};

PyMODINIT_FUNC PyInit__fmedian(void)
{
  PyObject *m;
  m = PyModule_Create(&fmedianmodule);
  if (m == NULL)
    return NULL;
  import_array();
  return m;
}
