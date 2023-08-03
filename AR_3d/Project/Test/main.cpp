#include <Python.h>
#include <iostream>
#include <numpy/arrayobject.h>
#include <ctime> //time
#include <stdio.h>
#include <opencv2/opencv.hpp>//5-1
using namespace std;
using namespace cv;
int main(int argc, char *argv[])
{
    wchar_t *program = Py_DecodeLocale(argv[0], NULL);
    cout <<argv[0]<<endl;
    if (program == NULL) {
        fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
        exit(1);
    }
    Py_SetProgramName(program);  /* optional but recommended */

    Py_Initialize();    // 使用Python系统前，必须使用Py_Initialize对其进行初始化
    if ( !Py_IsInitialized() )  //检查初始化是否成功
    {
        cout << "Py is not initialized" << endl;
        return -1;
    }

    import_array();

    cout << "---------------------添加路径--------------------" << endl;
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('/home/ylr/下载/Test/')");
    //PyRun_SimpleString("print(sys.path)");
    cout << "---------------------脚本获取--------------------" << endl;

    PyObject *pName,*pModule,*pDict,*pFunc,*pArgs;
    pName = PyUnicode_FromString("test1");  //python3中用这个
    pModule = PyImport_Import(pName);

    if ( !pModule ) {
        cout <<"can't find test.py"<<endl;
        return -1;
    }
    pDict = PyModule_GetDict(pModule);
    if ( !pDict ) {
        cout<<"can't find dict"<<endl;
        return -1;
    }

    cout << "---------------------函数获取--------------------" << endl;
    // 找出函数名为display的函数
    pFunc = PyDict_GetItemString(pDict, "object_recognize_3D");
    if ( !pFunc || !PyCallable_Check(pFunc) ) {
        cout<<"can't find function [object_recognize_3D]"<<endl;
        return -1;
    }

    cout << "---------------------构造参数对象--------------------" << endl;

    //将参数传进去。1代表一个参数。
    pArgs = PyTuple_New(1);
    float pc[20000][3];
    npy_intp dims[2];//2 dims
    dims[0]=20000;//20000*3 matrix
    dims[1]=3;

    for(int i=0;i<20000;i++)
    {
        for(int n=0;n<3;n++)
        {
            if(n==0){pc[i][n]=1;}
            if(n==1){pc[i][n]=2;}
            if(n==2){pc[i][n]=3;}
        }
    }
    PyObject *pValue = PyArray_SimpleNewFromData(2,dims,NPY_FLOAT,pc);
    PyTuple_SetItem(pArgs,0,pValue);
    PyObject *pResult = PyObject_CallObject(pFunc, pArgs);

    if (pResult == NULL)
    {
        cout <<"Return value is NULL."<<endl;
        Py_Finalize();
        return -1;
    }

    if(PyList_Check(pResult))
    {
        //int SizeOfList = PyList_Size(pResult);
        //cout<<"----"<<SizeOfList<<"----"<<endl;//==1
        PyObject *ListItem1 = PyList_GetItem(pResult, 0);//获取List对象中的每一个元素
        int NumOfItems1 = PyList_Size(ListItem1);//List对象子元素的大小
        //cout<<"----"<<NumOfItems1<<"----"<<endl;//==2
        for(int Index1 = 0;Index1<NumOfItems1;Index1++)//num of class
        {
            PyObject *ListItem2 = PyList_GetItem(ListItem1, Index1);
            if(PyTuple_Check(ListItem2))//index of a class
            {
                //int NumOfItems2 = PyTuple_Size(ListItem2);
                //cout<<"----"<<NumOfItems2<<"----"<<endl;//==2
                PyObject *pInt = PyTuple_GetItem(ListItem2, 0);
                PyObject *pArray = PyTuple_GetItem(ListItem2, 1);

                int result1;//class name
                PyArg_Parse(pInt, "i", &result1);
                //cout<<"----"<<result<<"----"<<endl;

                if(PyArray_Check(pArray))
                {
                    //int NumOfItems3 = PyArray_Size(pArray);
                    //cout<<"----"<<NumOfItems3<<"----"<<endl;//==6 and 3
                    PyArrayObject *result2;
                    PyArray_OutputConverter(pArray, &result2);
                    int Row = result2->dimensions[0], Col = result2->dimensions[1];
                    cout<<Row<<"*"<<Col<<endl;
                    for(int Index2 = 0;Index2 < Row;Index2++)
                    {
                        for(int Index3 =0;Index3 < Col;Index3++)
                        {
                            //访问数据，Index_m 和 Index_n 分别是数组元素的坐标，乘上相应维度的步长，即可以访问数组元素
                            float box = *(double *)(result2->data + Index2 * result2->strides[0] + Index3 * result2->strides[1]);
                            cout << "----" << box << "----"<< endl;
                        }
                    }
                    //Py_DECREF(result2);
                }
                Py_DECREF(pInt);
             }
             Py_DECREF(ListItem2);
        }
        Py_DECREF(ListItem1);
    }
    // 关闭Python
    Py_Finalize();
    return 0;
}
