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

  /* If user passed a NumPy array with wrong dtype, raise a TypeError rather than silently casting. */
  if (PyArray_Check(in_obj))
  {
    PyArrayObject *tmp = (PyArrayObject *)in_obj;
    if (PyArray_TYPE(tmp) != NPY_DOUBLE)
    {
      PyErr_SetString(PyExc_TypeError, "fmedian: input must be numpy array of dtype float64");
      return NULL;
    }
  }
  if (PyArray_Check(out_obj))
  {
    PyArrayObject *tmp = (PyArrayObject *)out_obj;
    if (PyArray_TYPE(tmp) != NPY_DOUBLE)
    {
      PyErr_SetString(PyExc_TypeError, "fmedian: output must be numpy array of dtype float64");
      return NULL;
    }
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
          double v = in[yy * nx + xx];
          if (!isnan(v))
          {
            buf[count++] = v;
          }
        }
      }
      if (count == 0)
      {
        /* No finite neighbors found. Behavior:
         * - For exclude_center==1: if center is finite, use center value; else write NaN.
         * - For exclude_center==0: no finite values in window -> write NaN.
         */
        double center = in[iy * nx + ix];
        if (exclude_center && !isnan(center))
        {
          out[iy * nx + ix] = center;
        }
        else
        {
          out[iy * nx + ix] = NAN;
        }
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
