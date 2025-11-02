#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include <math.h>

static PyObject *py_fsigma(PyObject *self, PyObject *args)
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
    PyErr_SetString(PyExc_TypeError, "fsigma: input and output must be numpy arrays of dtype float64");
    return NULL;
  }

  if (PyArray_NDIM(in_arr) != 2 || PyArray_NDIM(out_arr) != 2)
  {
    Py_DECREF(in_arr);
    Py_DECREF(out_arr);
    PyErr_SetString(PyExc_ValueError, "fsigma: arrays must be 2-dimensional");
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
    PyErr_SetString(PyExc_ValueError, "fsigma: input and output must have the same shape");
    return NULL;
  }

  double *in = (double *)PyArray_DATA(in_arr);
  double *out = (double *)PyArray_DATA(out_arr);

  for (npy_intp iy = 0; iy < ny; ++iy)
  {
    for (npy_intp ix = 0; ix < nx; ++ix)
    {
      int count = 0;
      double sum = 0.0;
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
          double v = in[yy * nx + xx];
          sum += v;
          count += 1;
        }
      }
      if (count == 0)
      {
        out[iy * nx + ix] = 0.0;
      }
      else
      {
        double mean = sum / count;
        double ss = 0.0;
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
            double v = in[yy * nx + xx];
            double d = v - mean;
            ss += d * d;
          }
        }
        out[iy * nx + ix] = sqrt(ss / count);
      }
    }
  }

  Py_DECREF(in_arr);
  PyArray_ResolveWritebackIfCopy(out_arr);
  Py_DECREF(out_arr);

  Py_RETURN_NONE;
}

static PyMethodDef FsigmaMethods[] = {
    {"fsigma", py_fsigma, METH_VARARGS, "Compute local standard deviation over window"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef fsigmamodule = {
    PyModuleDef_HEAD_INIT,
    "_fsigma",
    "fsigma C extension",
    -1,
    FsigmaMethods};

PyMODINIT_FUNC PyInit__fsigma(void)
{
  PyObject *m;
  m = PyModule_Create(&fsigmamodule);
  if (m == NULL)
    return NULL;
  import_array();
  return m;
}
