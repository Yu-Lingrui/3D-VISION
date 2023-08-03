#include <iostream>
#include <stdio.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <iomanip>
#include <time.h>
#include <signal.h>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <string>
#include <sstream>
#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/registration.h>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/logger.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <cmath>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#include "Leap.h"
#include <thread>
#include <future>
#include <Eigen/Dense>

#define PI 3.1415926

using namespace Leap;
using namespace std;
using namespace cv;
using namespace Eigen;

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

enum
{
    Processor_cl,
    Processor_gl,
    Processor_cpu
};

bool protonect_shutdown = false; // Whether the running application should shut down.
bool save_state = false;  //S保存按键标志位
bool find_state = true;   //D键标志位,按一次锁定指线长度,再按一次解锁,同时解锁保存按键
bool model_state = false;  //
bool recog_state = false;//R to recognize
// angle of rotation for the camera direction
float angle = 0.0f;

// actual vector representing the camera's direction
float lx=0.0f,lz=-1.0f;

// XZ position of the camera
float x=0.0f, z=5.0f;

// the key states. These variables will be zero
// when no key is being presses
float deltaAngle = 0.0f;
float deltaMove = 0;
int   xOrigin = -1;

// Converting ratio from mm to unit in OpenGL
float real2gl = 0.025;

Controller controller;//
const Frame frame;//

// Pointer for drawing quadratoc objects
GLUquadricObj *quadratic;

//float palmX;
//float palmY;
//float palmZ;
//Vector4f palm;
Eigen::Matrix<float,4,17> leap_xyz;  //存储手势原始坐标,为4*17齐次矩阵
Matrix4f lp_rgb;  //leap 到 rgb 的矩阵
Matrix4f rgb_lp;  //rgb 到 leap 的矩阵
Matrix4f rgb_depth;  //rgb 到 depth 的矩阵
Eigen::Matrix<float,4,17> cloud_xyz;  //存储转换后的手势坐标
double colors[17][3] = {{0.0,1.0,1.0},{1.0,0.0,1.0}, //palm,wrist
                        {0.2,0.0,0.3},{0.2,0.0,0.3},{0.0,0.0,1.0}, //bone
                        {0.2,0.0,0.3},{0.2,0.0,0.3},{0.0,0.0,1.0},
                        {0.2,0.0,0.3},{0.2,0.0,0.3},{0.0,0.0,1.0},
                        {0.2,0.0,0.3},{0.2,0.0,0.3},{0.0,0.0,1.0},
                        {0.2,0.0,0.3},{0.2,0.0,0.3},{0.0,0.0,1.0},};  //存储手势节点的颜色

void sigint_handler(int s)
{
    protonect_shutdown = true;
}

void changeSize(int w, int h) {

    // Prevent a divide by zero, when window is too short
    // (you cant make a window of zero width).
    if (h == 0)
        h = 1;

    float ratio =  w * 1.0 / h;

    // Use the Projection Matrix
    glMatrixMode(GL_PROJECTION);

    // Reset Matrix
    glLoadIdentity();

    // Set the viewport to be the entire window
    glViewport(0, 0, w, h);

    // Set the correct perspective.
    gluPerspective(45.0f, ratio, 0.1f, 100.0f);

    // Get Back to the Modelview
    glMatrixMode(GL_MODELVIEW);
}

void computePos(float deltaMove) {

    x += deltaMove * lx * 0.1f;
    z += deltaMove * lz * 0.1f;
}

void renderScene(void) {
    if (deltaMove)
        computePos(deltaMove);

    //std::cout << "In!" << std::endl;//test 4.5.2021

    // Clear Color and Depth Buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Reset transformations
    glLoadIdentity();

    // Set the camera
    gluLookAt(  x, 1.0f, z,
             x+lx, 1.0f,  z+lz,
             0.0f, 1.0f,  0.0f);
    gluLookAt( 0,  8.0f,  9.0f,
            0.0f,  2.0f,  0.0f,
            0.0f,  1.0f,  0.0f);

    // Draw ground
    glColor3f(0.5f, 0.5f, 0.5f);//gray
    //glColor3f(1.0f, 0.0f, 0.0f);
    glBegin(GL_QUADS);
        glVertex3f(-100.0f, 0.0f, -100.0f);
        glVertex3f(-100.0f, 0.0f,  100.0f);
        glVertex3f( 100.0f, 0.0f,  100.0f);
        glVertex3f( 100.0f, 0.0f, -100.0f);
    glEnd();

    const Frame frame = controller.frame();
    HandList hands = frame.hands();
    for (HandList::const_iterator hl = hands.begin(); hl != hands.end(); ++hl) {

        // Draw palm
        const Hand hand = *hl;
        float palmX = hand.palmPosition().x;
        float palmY = hand.palmPosition().y;
        float palmZ = hand.palmPosition().z;

        //std::cout << "In: " << palmX << std::endl;

        leap_xyz.block<4,1>(0,0) << palmX,
                                    palmY,
                                    palmZ,
                                    1;      //给leap_xyz赋值

        glColor3f(0.0f, 1.0f, 1.0f);//white
        glPushMatrix();
        glTranslatef((palmX)*real2gl, (palmY)*real2gl, (palmZ)*real2gl);
        glutSolidSphere(0.3f,100,100);
        glPopMatrix();

        // Draw wrist
        Arm arm = hand.arm();
        float wristX = arm.wristPosition().x;
        float wristY = arm.wristPosition().y;
        float wristZ = arm.wristPosition().z;

        leap_xyz.block<4,1>(0,1) << wristX,
                                    wristY,
                                    wristZ,
                                    1;  //给leap_xyz赋值

        glColor3f(1.0f, 0.0f, 1.0f);//pingRED
        glPushMatrix();
        glTranslatef(wristX*real2gl, wristY*real2gl, wristZ*real2gl);
        glutSolidSphere(0.2f,100,100);
        glPopMatrix();


        // Draw fingers
        const FingerList fingers = hand.fingers();
        int count = 0;
        for (FingerList::const_iterator fl = fingers.begin(); fl != fingers.end(); ++fl){
            const Finger finger = *fl;

             float tipX = finger.tipPosition().x;  //指尖点
             float tipY = finger.tipPosition().y;
             float tipZ = finger.tipPosition().z;

             leap_xyz.block<4,1>(0,4+count*3) << tipX,
                                                 tipY,
                                                 tipZ,
                                                 1;  //给leap_xyz赋值

             glColor3f(0.0f, 0.0f, 1.0f);//blue
             glPushMatrix();
             glTranslatef(tipX*real2gl, tipY*real2gl, tipZ*real2gl);
             glutSolidSphere(0.1f,100,100);
             glPopMatrix();

             count++;  //计数记录手指

            // Draw joints & bones

            for (int b = 0; b < 4; ++b) {
                Bone::Type boneType = static_cast<Bone::Type>(b);
                Bone bone = finger.bone(boneType);
                float boneStartX = bone.prevJoint().x;
                float boneStartY = bone.prevJoint().y;
                float boneStartZ = bone.prevJoint().z;
                float boneEndX   = bone.nextJoint().x;
                float boneEndY   = bone.nextJoint().y;
                float boneEndZ   = bone.nextJoint().z;

                if(b!=0&&b!=3)  //只选择部分节点转入点云
                {
                    leap_xyz.block<4,1>(0,count*3+b-2) << boneStartX,
                                                          boneStartY,
                                                          boneStartZ,
                                                          1;  //给leap_xyz赋值
                }

                // Draw joints
                glColor3f(0.0f, 1.0f, 0.0f);//green
                glPushMatrix();
                glTranslatef(boneStartX*real2gl, boneStartY*real2gl, boneStartZ*real2gl);
                glutSolidSphere(0.08f,100,100);
                glPopMatrix();

                // Draw joints
                glColor3f(0.2f, 0.0f, 0.3f);
                glPushMatrix();
                glTranslatef(boneEndX*real2gl, boneEndY*real2gl, boneEndZ*real2gl);
                glutSolidSphere(0.08f,100,100);
                glPopMatrix();

                // Draw bones
                float boneVectorX = boneEndX - boneStartX;
                float boneVectorY = boneEndY - boneStartY;
                float boneVectorZ = boneEndZ - boneStartZ;
                float phi = atan2(boneVectorX, boneVectorZ) * 180 / PI;
                float theta = (-1) * atan2(boneVectorY, hypot(boneVectorX, boneVectorZ)) * 180 / PI;
                glColor3f(0.6f, 0.6f, 0.0f);
                glPushMatrix();
                glTranslatef(boneStartX*real2gl, boneStartY*real2gl, boneStartZ*real2gl);
                glRotatef(phi, 0.0f, 1.0f, 0.0f);
                glRotatef(theta, 1.0f, 0.0f, 0.0f);
                quadratic = gluNewQuadric();
                gluCylinder(quadratic,0.05f,0.05f,bone.length()*real2gl,32,32);
                glPopMatrix();

                gluDeleteQuadric(quadratic);
            }
        }
    }
    glutSwapBuffers();
}

void processNormalKeys(unsigned char key, int xx, int yy) {

        if (key == 27)//esc
              exit(0);
}

void pressKey(int key, int xx, int yy) {

       switch (key) {
             case GLUT_KEY_UP : deltaMove = 0.5f; break;
             case GLUT_KEY_DOWN : deltaMove = -0.5f; break;
       }
}

void releaseKey(int key, int x, int y) {

        switch (key) {
             case GLUT_KEY_UP :
             case GLUT_KEY_DOWN : deltaMove = 0;break;
        }
}

void mouseMove(int x, int y) {

         // this will only be true when the left button is down
         if (xOrigin >= 0) {

        // update deltaAngle
        deltaAngle = (x - xOrigin) * 0.001f;

        // update camera's direction
        lx = sin(angle + deltaAngle);
        lz = -cos(angle + deltaAngle);
    }
}

void mouseButton(int button, int state, int x, int y) {

    // only start motion if the left button is pressed
    if (button == GLUT_LEFT_BUTTON) {

        // when the button is released
        if (state == GLUT_UP) {
            angle += deltaAngle;
            xOrigin = -1;
        }
        else  {// state = GLUT_DOWN
            xOrigin = x;
        }
    }
}

void keyboardEventOccurred(const pcl::visualization::KeyboardEvent& event, void* nothing)  //pcl 的键盘监听函数
{
    if(event.keyDown())
    {
        if(event.getKeySym() == "d")
        {
            find_state = !find_state;   //按D键切换状态
        }
        if(event.getKeySym() == "s")
        {
            save_state = true;          //按S键保存选择的点云
            model_state = true;
        }
        if(event.getKeySym() == "w")
        {
            recog_state = true;   //press "w" key to start recognize
        }
    }
}

float distance(PointT A, PointT B)  //计算两点距离的函数
{
    return sqrt(pow(A.x-B.x,2)+pow(A.y-B.y,2)+pow(A.z-B.z,2));
}

int main(int argc, char *argv[])
{
    wchar_t *program = Py_DecodeLocale(argv[0], NULL);
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

    cout << "----添加路径----" << endl;
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('/home/ylr/下载/Test/')");
    //PyRun_SimpleString("print(sys.path)");

    cout << "----脚本获取----" << endl;
    PyObject *pName,*pModule,*pDict,*pFunc;
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

    cout << "----函数获取----" << endl;
    // 找出函数名为display的函数
    pFunc = PyDict_GetItemString(pDict, "object_recognize_3D");
    if ( !pFunc || !PyCallable_Check(pFunc) ) {
        cout<<"can't find function [object_recognize_3D]"<<endl;
        return -1;
    }
    //end

//定义变量
    std::cout << "start!" << std::endl;
    libfreenect2::Freenect2 freenect2;
    libfreenect2::Freenect2Device *dev = 0;
    libfreenect2::PacketPipeline  *pipeline = 0;

//搜寻并初始化传感器
    if(freenect2.enumerateDevices() == 0)
    {
        std::cout << "no device connected!" << std::endl;
        return -1;
    }
    string serial = freenect2.getDefaultDeviceSerialNumber();
    std::cout << "SERIAL: " << serial << std::endl;

//配置传输格式
#if 1 // sean
    int depthProcessor = Processor_cl;
    if(depthProcessor == Processor_cpu)
    {
        if(!pipeline)
            //! [pipeline]
            pipeline = new libfreenect2::CpuPacketPipeline();
        //! [pipeline]
    }
    else if (depthProcessor == Processor_gl) // if support gl
    {
#ifdef LIBFREENECT2_WITH_OPENGL_SUPPORT
        if(!pipeline)
        {
            pipeline = new libfreenect2::OpenGLPacketPipeline();
        }
#else
        std::cout << "OpenGL pipeline is not supported!" << std::endl;
#endif
    }
    else if (depthProcessor == Processor_cl) // if support cl
    {
#ifdef LIBFREENECT2_WITH_OPENCL_SUPPORT
        if(!pipeline)
            pipeline = new libfreenect2::OpenCLPacketPipeline();
#else
        std::cout << "OpenCL pipeline is not supported!" << std::endl;//?4.5.2021
#endif
    }

//启动设备
    if(pipeline)
    {
        dev = freenect2.openDevice(serial, pipeline);
    }
    else
    {
        dev = freenect2.openDevice(serial);
    }
    if(dev == 0)
    {
        std::cout << "failure opening device!" << std::endl;
        return -1;
    }
    signal(SIGINT, sigint_handler);
    protonect_shutdown = false;
    libfreenect2::SyncMultiFrameListener listener(
            libfreenect2::Frame::Color |
            libfreenect2::Frame::Depth |
            libfreenect2::Frame::Ir);
    libfreenect2::FrameMap frames;
    dev->setColorFrameListener(&listener);
    dev->setIrAndDepthFrameListener(&listener);


//启动数据传输
    dev->start();

    std::cout << "device serial: " << dev->getSerialNumber() << std::endl;
    std::cout << "device firmware: " << dev->getFirmwareVersion() << std::endl;

//循环接收
    libfreenect2::Registration* registration = new libfreenect2::Registration(dev->getIrCameraParams(), dev->getColorCameraParams());
    libfreenect2::Frame undistorted(512, 424, 4), registered(512, 424, 4), depth2rgb(1920, 1080 + 2, 4);

    Mat rgbmat, depthmat, rgbd, dst;
    float x, y, z, color;

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Viewer"));
    viewer->registerKeyboardCallback(&keyboardEventOccurred,(void*)NULL);  //键盘监听
    viewer->setCameraPosition(0,0,1000,0,0,-1,0,-1,0,0);   //设定初始视角

    pcl::KdTreeFLANN<PointT> kdtree;  //创建kd树

    std::vector<int> pointIdxRadiusSearch;  //存储kd树找到的点的索引
    std::vector<float>  pointRadiusSquaredDistance;

    // init GLUT and create window
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(100,100);
    glutInitWindowSize(320,320);
    glutCreateWindow("Leap Motion");//

    // register callbacks
    glutDisplayFunc(renderScene);//
    glutReshapeFunc(changeSize);
    glutIdleFunc(renderScene);

    glutIgnoreKeyRepeat(1);
    glutKeyboardFunc(processNormalKeys);
    glutSpecialFunc(pressKey);
    glutSpecialUpFunc(releaseKey);

    // here are the two new functions
    glutMouseFunc(mouseButton);
    glutMotionFunc(mouseMove);

    glEnable(GL_DEPTH_TEST);
    // glShadeModel(GL_SMOOTH);
    // glLightfv(GL_LIGHT1, GL_AMBIENT, LightAmbient); // Setup The Ambient Light
    // glLightfv(GL_LIGHT1, GL_DIFFUSE, LightDiffuse); // Setup The Diffuse Light
    // glLightfv(GL_LIGHT1, GL_POSITION,LightPosition);// Position The Light
    // glEnable(GL_LIGHT1); // Enable Light One

    // enter GLUT event processing cycle
    std::thread t(glutMainLoop);   //将leap的主循环函数放入线程
    //std::thread t_m;
    PointT o_find;  //存储指线到点云的交点
    float K;  //存储交点到指线原点的距离倍数
    set<int> s;  //存储涂色的点的索引
    //int targetnum = 0;  //记录保存的点云的计数
    /*lp_rgb <<-0.9999,0.0003,0.0185,-148.6363,
             -0.0006,-0.9998,-0.0190,284.6143,
              0.0186,-0.0192,0.9996,117.3620,
              0,0,0,1.0000;
    rgb_lp <<-0.999771131689191, -0.0006720187786706647, 0.01846283797450896, -150.5778226905179,
              0.0002758035497362674, -0.9998190485483229, -0.01904786269504187, 286.8393230496208,
              0.01856249023849071, -0.01915245326382596, 0.9997019467992146, -109.1169457655665,
              0, 0, 0, 1;*/

    lp_rgb <<-0.9999,0.0003,0.0185,-194.2927,
                 -0.0006,-0.9998,-0.0190,292.0596,
                  0.0186,-0.0192,0.9996,-99.1805,
                  0,0,0,1.0000;
    rgb_lp <<-0.999771131689191, -0.0006720187786706647, 0.01846283797450896, -190.2773,
                  0.0002758035497362674, -0.9998190485483229, -0.01904786269504187, 290.7160,
                  0.01856249023849071, -0.01915245326382596, 0.9997019467992146, -110.2810,
                  0, 0, 0, 1;
    rgb_depth << 0.999957,-0.00863626,0.00330734,-50.5963541066607,
                 0.00864393,0.99996,-0.00231205,-0.343015879849346,
                -0.00328724,0.00234054,0.999992,2.27473117895580,
                 0,0,0,1;
    float pc[217088][3];//5-3
    float boxPoint[8][3];
    while(!protonect_shutdown)
    {
        //float boxPoint[8][3];
        listener.waitForNewFrame(frames);
        libfreenect2::Frame *rgb = frames[libfreenect2::Frame::Color];
        libfreenect2::Frame *depth = frames[libfreenect2::Frame::Depth];
        registration->apply(rgb, depth, &undistorted, &registered, true, &depth2rgb);

        cloud_xyz = rgb_depth*rgb_lp*leap_xyz;  //leap -> depth的转换计算

        PointCloud::Ptr cloud ( new PointCloud ); //使用智能指针，创建一个空点云。这种指针用完会自动释放。
        //float pc[217088][3];//5-3

        for (int m = 0;  m < 512 ; m++)//point_cloud 3d position 4-22
        {
            for (int n = 0 ; n < 424 ; n++)
            {
                PointT p;
                registration->getPointXYZRGB(&undistorted, &registered, n, m, x, y, z, color);
                const uint8_t *c = reinterpret_cast<uint8_t*>(&color);
                uint8_t b = c[0];
                uint8_t g = c[1];
                uint8_t r = c[2];

                //save point cloud data to recognize
                if(isnan(x)||isnan(y)||isnan(z))
                {
                    pc[m*424+n][0]=1;//x
                    pc[m*424+n][1]=1;//y
                    pc[m*424+n][2]=1;//z
                }
                else
                {
                    pc[m*424+n][0]=(float)x;//x
                    pc[m*424+n][1]=(float)y;//y
                    pc[m*424+n][2]=(float)z;//z  --------------------------------------------------------------------------5-8 -z?
                }

                //if (z<1 && y<0.5)  //暂时先通过限定xyz来除去不需要的点，点云分割还在学习中。。。
                if (z<5 && y<1)   //change z<4 y<0.5 4-22
                {
                    p.z = -z*1000;//??-z 5-8
                    p.x = x*1000;
                    p.y = y*1000;   //乘以1000与手势坐标统一 m->mm

                    p.b = b;
                    p.g = g;
                    p.r = r;
                }
                cloud->points.push_back(p);
            }
        }
        set<int>::iterator iter = s.begin();
        while(iter!=s.end()&&save_state==false)
        {
            cloud->points[*iter].r=255;//RED
            cloud->points[*iter].g=0;
            cloud->points[*iter].b=0;
            iter++;
        }  //根据s的索引信息涂色

        kdtree.setInputCloud(cloud);  //设置kd树的输入点云
        for(int h = 0;h<17;h++)
        {
            PointT o;
            int radiu;
            o.x=cloud_xyz(0,h);
            o.y=cloud_xyz(1,h);
            o.z=cloud_xyz(2,h);
            if(h == 0)
                radiu = 15;  //palm点的半径15
            else if(h == 1)
                radiu = 9;   //wrist点的半径9
            else
                radiu = 5;   //普通节点半径5
            stringstream oss;
            oss<<"o"<<h;
            viewer->addSphere(o,radiu,oss.str(),0);
            viewer->updateSphere(o,radiu,colors[h][0],colors[h][1],colors[h][2],oss.str());
        }  //添加手势节点

        for(int x = 0;x<5;x++)  //5根手指
        {
            for(int y = 0;y<2;y++)  //两根骨骼
            {
                PointT p1;
                PointT p2;
                p1.x=cloud_xyz(0,2+x*3+y);
                p1.y=cloud_xyz(1,2+x*3+y);
                p1.z=cloud_xyz(2,2+x*3+y);
                p2.x=cloud_xyz(0,3+x*3+y);
                p2.y=cloud_xyz(1,3+x*3+y);
                p2.z=cloud_xyz(2,3+x*3+y);
                stringstream oss;
                oss<<"l"<<(x*2+y);
                viewer->addLine(p1,p2,0.6,0.6,0.0,oss.str(),0);
            }
        }  //添加骨骼
        PointT palm;
        PointT tip;
        PointT far;

        palm.x=cloud_xyz(0,5);
        palm.y=cloud_xyz(1,5);
        palm.z=cloud_xyz(2,5);
        tip.x=cloud_xyz(0,7);
        tip.y=cloud_xyz(1,7);
        tip.z=cloud_xyz(2,7);  //确定指线的两点

        float radius = distance(tip,palm);
        if(find_state == true)  //为true时，未锁定交点
        {
            for(int n = 0;n<80;n++)//5-7 n<40----------------------------------------------------------------------------
            {
                o_find.x=palm.x+(n+1)*(tip.x-palm.x)/2.0;
                o_find.y=palm.y+(n+1)*(tip.y-palm.y)/2.0;
                o_find.z=palm.z+(n+1)*(tip.z-palm.z)/2.0;
                if(kdtree.radiusSearch(o_find,radius/4.0,pointIdxRadiusSearch,pointRadiusSquaredDistance)>0)
                {
                    K = (o_find.x-palm.x)/(tip.x-palm.x)+0.5;
                    break;  //找到最近的后退出寻找
                }
            }  //从指线原点出发，每隔(tip.x-palm.x)/2.0的距离，寻找一次交点

            far.x=palm.x+40*(tip.x-palm.x);//5-7 20-----------------------------------------------------------
            far.y=palm.y+40*(tip.y-palm.y);
            far.z=palm.z+40*(tip.z-palm.z);
            viewer->addLine(palm,far,1.0,0.0,0.0,"l",0);
            viewer->addSphere(o_find,10,"o_find",0);  //绘制指线!!5-3-------------------------------------------------------
            /*
            float Point_x1=(palm.x+K*(tip.x-palm.x))/1000;
            float Point_y1=(palm.y+K*(tip.y-palm.y))/1000;//5-7!!y
            float Point_z1=(palm.z+K*(tip.z-palm.z))/1000;//z
            if(Point_x1 != 0 && Point_y1 !=0 && Point_z1 !=0)
            {
                cout << "拾取点坐标: " << "x:" << Point_x1 << " " << "y:" << Point_y1 <<" "<< "z:" << Point_z1 <<endl;
                Point_x1 = 0;
                Point_y1 = 0;
                Point_z1 = 0;
            }
            */

            //judge if or not to recognize and use votenet and judge object class
            if(recog_state == true)//press "w" key
            {
                clock_t time_stt = clock();
                //cout << "----构造参数对象----" << endl;
                //将参数传进去。1代表一个参数。
                PyObject *pArgs = PyTuple_New(1);
                npy_intp dims[2];//2 dims
                dims[0]=217088;//217088*3 matrix
                dims[1]=3;

                PyObject *pValue = PyArray_SimpleNewFromData(2,dims,NPY_FLOAT,pc);
                PyTuple_SetItem(pArgs,0,pValue);
                PyObject *pResult = PyObject_CallObject(pFunc, pArgs);

                //Py_DECREF(pArgs);//5-3!!
                //Py_DECREF(pValue);//5-7

                //x,y,z of gesture point
                float Point_x=(palm.x+K*(tip.x-palm.x))/1000;
                float Point_z=(palm.y+K*(tip.y-palm.y))/1000;//5-7!!y
                float Point_y=(palm.z+K*(tip.z-palm.z))/1000;//z
                cout << "拾取点坐标: " <<"x:"<<Point_x<< " " << "y:" << Point_y <<" "<< "z:" << Point_z <<endl;

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

                    //Py_DECREF(pResult);//5-7 5-3!!-------------------------------------------------------------------------------------------

                    int NumOfItems1 = PyList_Size(ListItem1);//List对象子元素的大小
                    //cout<<"----"<<NumOfItems1<<"----"<<endl;//==2
                    for(int Index1 = 0;Index1<NumOfItems1;Index1++)//num of objects
                    {
                        PyObject *ListItem2 = PyList_GetItem(ListItem1, Index1);

                        //Py_DECREF(ListItem1);//5-7 5-3!!--------------------------------------------------------------------------------------

                        if(PyTuple_Check(ListItem2))//index of a class
                        {
                            //int NumOfItems2 = PyTuple_Size(ListItem2);
                            //cout<<"----"<<NumOfItems2<<"----"<<endl;//==2

                            PyObject *pInt = PyTuple_GetItem(ListItem2, 0);//class name
                            PyObject *pArray = PyTuple_GetItem(ListItem2, 1);//8 box point
                            PyObject *pConf = PyTuple_GetItem(ListItem2, 2);//6-6 confidence value

                            //int NumOfArray = PyArray_Size(pArray);//Array对象子元素的大小
                            //cout << "-----------------" << NumOfArray <<"------------------"<<endl;
                           // Py_DECREF(ListItem2);//5-3!!!!---------------------------------------------------------------------------------

                            int result1;//class name
                            PyArg_Parse(pInt, "i", &result1);

                            float Conf;//6-6 confidence value
                            /*
                            if(PyFloat_Check(pConf))
                            {
                               // PyArrayObject *Conf_result;
                               // PyArray_OutputConverter(pConf, &Conf_result);
                                cout<< "yes" <<endl;
                            }
                            */
                            PyArg_Parse(pConf, "f", &Conf);

                            //cout<<"----"<<result<<"----"<<endl;

                            //Py_DECREF(pInt);//5-3!!



                            if(PyArray_Check(pArray))
                            {
                                //int NumOfItems3 = PyArray_Size(pArray);
                                //cout<<"----"<<NumOfItems3<<"----"<<endl;//==6 and 3
                                PyArrayObject *result2;
                                PyArray_OutputConverter(pArray, &result2);

                                //Py_DECREF(pArray);//5-7 5-3!!
                                //float boxPoint[8][3];
                                int Row = result2->dimensions[0], Col = result2->dimensions[1];
                                //cout<<Row<<"*"<<Col<<endl;//5-7
                                for(int Index2 = 0; Index2 < Row; Index2++)//8 box point
                                {
                                    //x,y,z of one box point
                                    for(int Index3 =0;Index3 < Col;Index3++)//x,y,z
                                    {
                                        //访问数据，Index_m 和 Index_n 分别是数组元素的坐标，乘上相应维度的步长，即可以访问数组元素
                                        float box = *(double *)(result2->data + Index2 * result2->strides[0] + Index3 * result2->strides[1]);
                                        //cout << "----" << box << "----"<< endl;
                                        boxPoint[Index2][Index3] = box;
                                    }

                                }
                                //Py_DECREF(result2);//5-3!!

                                //parameter of 6 plains
                                float a_1 = (boxPoint[1][1]-boxPoint[0][1])*(boxPoint[2][2]-boxPoint[0][2])-(boxPoint[2][1]-boxPoint[0][1])*(boxPoint[1][2]-boxPoint[0][2]);
                                float b_1 = (boxPoint[1][2]-boxPoint[0][2])*(boxPoint[2][0]-boxPoint[0][0])-(boxPoint[2][2]-boxPoint[0][2])*(boxPoint[1][0]-boxPoint[0][0]);//0-2
                                float c_1 = (boxPoint[1][0]-boxPoint[0][0])*(boxPoint[2][1]-boxPoint[0][1])-(boxPoint[2][0]-boxPoint[0][0])*(boxPoint[1][1]-boxPoint[0][1]);
                                float d_1 = -(a_1*boxPoint[0][0]+b_1*boxPoint[0][1]+c_1*boxPoint[0][2]);

                                float a_2 = (boxPoint[5][1]-boxPoint[4][1])*(boxPoint[6][2]-boxPoint[4][2])-(boxPoint[6][1]-boxPoint[4][1])*(boxPoint[5][2]-boxPoint[4][2]);
                                float b_2 = (boxPoint[5][2]-boxPoint[4][2])*(boxPoint[6][0]-boxPoint[4][0])-(boxPoint[6][2]-boxPoint[4][2])*(boxPoint[5][0]-boxPoint[4][0]);
                                float c_2 = (boxPoint[5][0]-boxPoint[4][0])*(boxPoint[6][1]-boxPoint[4][1])-(boxPoint[6][0]-boxPoint[4][0])*(boxPoint[5][1]-boxPoint[4][1]);
                                float d_2 = -(a_2*boxPoint[4][0]+b_2*boxPoint[4][1]+c_2*boxPoint[4][2]);

                                float a_3 = (boxPoint[3][1]-boxPoint[2][1])*(boxPoint[6][2]-boxPoint[2][2])-(boxPoint[6][1]-boxPoint[2][1])*(boxPoint[3][2]-boxPoint[2][2]);
                                float b_3 = (boxPoint[3][2]-boxPoint[2][2])*(boxPoint[6][0]-boxPoint[2][0])-(boxPoint[6][2]-boxPoint[2][2])*(boxPoint[3][0]-boxPoint[2][0]);
                                float c_3 = (boxPoint[3][0]-boxPoint[2][0])*(boxPoint[6][1]-boxPoint[2][1])-(boxPoint[6][0]-boxPoint[2][0])*(boxPoint[3][1]-boxPoint[2][1]);
                                float d_3 = -(a_3*boxPoint[2][0]+b_3*boxPoint[2][1]+c_3*boxPoint[2][2]);

                                float a_4 = (boxPoint[1][1]-boxPoint[0][1])*(boxPoint[4][2]-boxPoint[0][2])-(boxPoint[4][1]-boxPoint[0][1])*(boxPoint[1][2]-boxPoint[0][2]);
                                float b_4 = (boxPoint[1][2]-boxPoint[0][2])*(boxPoint[4][0]-boxPoint[0][0])-(boxPoint[4][2]-boxPoint[0][2])*(boxPoint[1][0]-boxPoint[0][0]);
                                float c_4 = (boxPoint[1][0]-boxPoint[0][0])*(boxPoint[4][1]-boxPoint[0][1])-(boxPoint[4][0]-boxPoint[0][0])*(boxPoint[1][1]-boxPoint[0][1]);
                                float d_4 = -(a_4*boxPoint[0][0]+b_4*boxPoint[0][1]+c_4*boxPoint[0][2]);

                                float a_5 = (boxPoint[2][1]-boxPoint[1][1])*(boxPoint[5][2]-boxPoint[1][2])-(boxPoint[5][1]-boxPoint[1][1])*(boxPoint[2][2]-boxPoint[1][2]);
                                float b_5 = (boxPoint[2][2]-boxPoint[1][2])*(boxPoint[5][0]-boxPoint[1][0])-(boxPoint[5][2]-boxPoint[1][2])*(boxPoint[2][0]-boxPoint[1][0]);
                                float c_5 = (boxPoint[2][0]-boxPoint[1][0])*(boxPoint[5][1]-boxPoint[1][1])-(boxPoint[5][0]-boxPoint[1][0])*(boxPoint[2][1]-boxPoint[1][1]);
                                float d_5 = -(a_5*boxPoint[1][0]+b_5*boxPoint[1][1]+c_5*boxPoint[1][2]);

                                float a_6 = (boxPoint[3][1]-boxPoint[0][1])*(boxPoint[4][2]-boxPoint[0][2])-(boxPoint[4][1]-boxPoint[0][1])*(boxPoint[3][2]-boxPoint[0][2]);
                                float b_6 = (boxPoint[3][2]-boxPoint[0][2])*(boxPoint[4][0]-boxPoint[0][0])-(boxPoint[4][2]-boxPoint[0][2])*(boxPoint[3][0]-boxPoint[0][0]);
                                float c_6 = (boxPoint[3][0]-boxPoint[0][0])*(boxPoint[4][1]-boxPoint[0][1])-(boxPoint[4][0]-boxPoint[0][0])*(boxPoint[3][1]-boxPoint[0][1]);
                                float d_6 = -(a_6*boxPoint[0][0]+b_6*boxPoint[0][1]+c_6*boxPoint[0][2]);

                                /*
                                //x,y,z of gesture point
                                float Point_x=(palm.x+K*(tip.x-palm.x))/1000;
                                float Point_z=(palm.y+K*(tip.y-palm.y))/1000;//5-7!!y
                                float Point_y=(palm.z+K*(tip.z-palm.z))/1000;//z
                                //cout << "拾取点坐标: " <<"x:"<<Point_x<< " " << "y:" << Point_y <<" "<< "z:" << Point_z <<endl;
                                */

                                //judge if point in box
                                float D1 = (float)fabs(a_1*boxPoint[4][0]+b_1*boxPoint[4][1]+c_1*boxPoint[4][2]+d_1)/sqrt(pow(a_1,2)+pow(b_1,2)+pow(c_1,2));
                                float D2 = (float)fabs(a_3*boxPoint[0][0]+b_3*boxPoint[0][1]+c_3*boxPoint[0][2]+d_3)/sqrt(pow(a_3,2)+pow(b_3,2)+pow(c_3,2));
                                float D3 = (float)fabs(a_5*boxPoint[0][0]+b_5*boxPoint[0][1]+c_5*boxPoint[0][2]+d_5)/sqrt(pow(a_5,2)+pow(b_5,2)+pow(c_5,2));

                                float d1 = (float)fabs(a_1*Point_x+b_1*Point_y+c_1*Point_z+d_1)/sqrt(pow(a_1,2)+pow(b_1,2)+pow(c_1,2));
                                float d2 = (float)fabs(a_2*Point_x+b_2*Point_y+c_2*Point_z+d_2)/sqrt(pow(a_2,2)+pow(b_2,2)+pow(c_2,2));
                                float d3 = (float)fabs(a_3*Point_x+b_3*Point_y+c_3*Point_z+d_3)/sqrt(pow(a_3,2)+pow(b_3,2)+pow(c_3,2));
                                float d4 = (float)fabs(a_4*Point_x+b_4*Point_y+c_4*Point_z+d_4)/sqrt(pow(a_4,2)+pow(b_4,2)+pow(c_4,2));
                                float d5 = (float)fabs(a_5*Point_x+b_5*Point_y+c_5*Point_z+d_5)/sqrt(pow(a_5,2)+pow(b_5,2)+pow(c_5,2));
                                float d6 = (float)fabs(a_6*Point_x+b_6*Point_y+c_6*Point_z+d_6)/sqrt(pow(a_6,2)+pow(b_6,2)+pow(c_6,2));

                                //cout << d1 << "-"<<d2 << "-"<<D1 << "--"<<d3 << "-"<<d4 <<"-"<< D2 <<"--"<< d5 <<"-"<< d6 <<"-"<< D3 << endl;
                                if(((int&)D1-(int&)d1)>0 && ((int&)D1-(int&)d2)>0 && ((int&)D2-(int&)d3)>0 && ((int&)D2-(int&)d4)>0 && ((int&)D3-(int&)d5)>0 && ((int&)D3-(int&)d6)>0)
                                {
                                    string label;
                                    switch(result1)
                                    {
                                        case 0:  label = "床(bed)";
                                                 break;
                                        case 1:  label = "桌子(table)";
                                                 break;
                                        case 2:  label = "沙发(sofa)";
                                                 break;
                                        case 3:  label = "椅子(chair)";
                                                 break;
                                        case 4:  label = "马桶(toilet)";
                                                 break;
                                        case 5:  label = "办公桌(desk)";
                                                 break;
                                        case 6:  label = "柜子(dresser)";
                                                 break;
                                        case 7:  label = "床头柜(night stand)";
                                                 break;
                                        case 8:  label = "书架(bookshelf)";
                                                 break;
                                        case 9:  label = "浴缸(bathtub)";
                                                 break;
                                    }
                                    cout << "---------------" <<endl;
                                    cout << "指向的物品是 : " << label <<endl;//output the result of detection  
                                    cout << "置信度 : " << Conf <<endl;//6-6
                                    cout << "3D包围框坐标 : "<<endl;
                                    for(int a=0; a<8 ; a++)
                                    {
                                            cout << "[" << boxPoint[a][0] << "," << boxPoint[a][1] << "," << boxPoint[a][2]<< "]"<<endl;
                                    }
                                    model_state = true;
                                    recog_state = false;
                                    break;
                                }

                            }
                            Py_DECREF(pInt);//5-3
                         }
                         //Py_DECREF(ListItem2);//5-3
                    }
                    //Py_DECREF(ListItem1);//5-3
                }
                if(recog_state == true)
                {
                    cout << "---------------" <<endl;
                    cout <<"无结果！"<< endl;
                    recog_state = false;
                    model_state = false;
                }
                cout << "检测所用时间为 : " << 1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC <<"ms"<<endl;
                cout << "---------------"<< endl;
            }

            if(save_state == true)  //涂色完毕，切换find_state为true时，可以按S键保存点云
            {
                set<int>::iterator iter = s.begin();
                PointCloud::Ptr target ( new PointCloud );
                while(iter!=s.end())
                {
                    PointT tp;
                    tp.x = cloud->points[*iter].x;
                    tp.y = cloud->points[*iter].y;
                    tp.z = cloud->points[*iter].z;
                    tp.r = cloud->points[*iter].r;
                    tp.g = cloud->points[*iter].g;
                    tp.b = cloud->points[*iter].b;
                    target->points.push_back(tp);
                    iter++;
                }  //给涂色的点创建新点云
                target->height = 1;     //设置点云高度
                target->width = target->points.size();  //设置点云宽度
                target->is_dense = false;           //非密集型
                stringstream oss;
                //oss<<"target"<<targetnum<<".pcd";
                oss<<"/home/ylr/桌面/model/build-src-Desktop-Default/data/"<<"target.pcd";   //save pcd
                pcl::io::savePCDFileASCII(oss.str(), *target);
                save_state = false;
                //targetnum++;
                s.clear();
            }
        }
        else
        {  //为false时，锁定了交点，开始涂色
            far.x=palm.x+K*(tip.x-palm.x);
            far.y=palm.y+K*(tip.y-palm.y);
            far.z=palm.z+K*(tip.z-palm.z);
            viewer->addLine(palm,far,1.0,0.0,0.0,"l",0);
            viewer->addSphere(far,10,"o_find",0);
            if(kdtree.radiusSearch(far,40,pointIdxRadiusSearch,pointRadiusSquaredDistance)>0)//
            {
                for(size_t i = 0;i<pointIdxRadiusSearch.size();++i)
                {
                    s.insert(pointIdxRadiusSearch[i]);
                }
            }  //涂色
        }
    //    if(save_state == false && model_state == true)
        if(model_state == true)
        {
            //t_m = std::thread(sysmodel);
            //system(" gnome-terminal --working-directory='/home/ylr/桌面/model/build-src-Desktop-Default' -x ./model data/object_templates.txt data/target_templates.txt");//
            //model_state = false;
            PointT box_1,box_2,box_3,box_4,box_5,box_6,box_7,box_8;
            box_1.x=boxPoint[0][0]*1000;
            box_1.y=boxPoint[0][2]*1000;
            box_1.z=boxPoint[0][1]*1000;

            box_2.x=boxPoint[1][0]*1000;
            box_2.y=boxPoint[1][2]*1000;
            box_2.z=boxPoint[1][1]*1000;

            box_3.x=boxPoint[2][0]*1000;
            box_3.y=boxPoint[2][2]*1000;
            box_3.z=boxPoint[2][1]*1000;

            box_4.x=boxPoint[3][0]*1000;
            box_4.y=boxPoint[3][2]*1000;
            box_4.z=boxPoint[3][1]*1000;

            box_5.x=boxPoint[4][0]*1000;
            box_5.y=boxPoint[4][2]*1000;
            box_5.z=boxPoint[4][1]*1000;

            box_6.x=boxPoint[5][0]*1000;
            box_6.y=boxPoint[5][2]*1000;
            box_6.z=boxPoint[5][1]*1000;

            box_7.x=boxPoint[6][0]*1000;
            box_7.y=boxPoint[6][2]*1000;
            box_7.z=boxPoint[6][1]*1000;

            box_8.x=boxPoint[7][0]*1000;
            box_8.y=boxPoint[7][2]*1000;
            box_8.z=boxPoint[7][1]*1000;

            viewer->addLine(box_1,box_2,0.0,1.0,0.0,"box1");
            viewer->addLine(box_2,box_3,0.0,1.0,0.0,"box2");
            viewer->addLine(box_3,box_4,0.0,1.0,0.0,"box3");
            viewer->addLine(box_4,box_1,0.0,1.0,0.0,"box4");
            viewer->addLine(box_5,box_6,0.0,1.0,0.0,"box5");
            viewer->addLine(box_6,box_7,0.0,1.0,0.0,"box6");
            viewer->addLine(box_7,box_8,0.0,1.0,0.0,"box7");
            viewer->addLine(box_8,box_5,0.0,1.0,0.0,"box8");
            viewer->addLine(box_1,box_5,0.0,1.0,0.0,"box9");
            viewer->addLine(box_2,box_6,0.0,1.0,0.0,"box10");
            viewer->addLine(box_3,box_7,0.0,1.0,0.0,"box11");
            viewer->addLine(box_4,box_8,0.0,1.0,0.0,"box12");
        }

        viewer->addPointCloud(cloud,"cloud");
        viewer->spinOnce(0.01);//0.01

        viewer->removeAllPointClouds();
        viewer->removeAllShapes();

        int key = cv::waitKey(1);
        protonect_shutdown = protonect_shutdown || (key > 0 && ((key & 0xFF) == 27)); // shutdown on escape
        listener.release(frames);
    }

    dev->stop();
    dev->close();

    delete registration;

#endif
    t.join();  //线程阻塞
    //t_m.join();
    std::cout << "stop!" << std::endl;
    // 关闭Python
    Py_DECREF(pName);
    Py_DECREF(pModule);
    Py_DECREF(pDict);
    Py_DECREF(pFunc);
    Py_Finalize();
    return 0;

}
