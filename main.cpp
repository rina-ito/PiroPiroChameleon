//コンパイルコマンド
//g++ -O3 main.cpp -framework OpenGL -framework GLUT -I/usr/include/ni -lOpenNI `pkg-config --cflags opencv --libs opencv` -I/usr/local/include -lalut -framework OpenAL -Wno-deprecated
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <XnOpenNI.h>  //Kinect
#include <XnCodecIDs.h>  //Kinect
#include <XnCppWrapper.h>  //Kinect
#include <XnPropNames.h>  //Kinect
//#include <map>  //Kinect
#include <GLUT/glut.h>  //OpenGL
#include <cv.h>  //OpenCV
#include <cxcore.h>  //OpenCV
#include <highgui.h>  //OpenCV
#include <AL/alut.h>  //OpenAL

using namespace xn;

//定数の宣言
#define GL_WIN_SIZE_X 640
#define GL_WIN_SIZE_Y 480
#define GL_WIN_TOTAL 307200
#define MAX_DEPTH 10000
#define MAX_USER 10
#define XN_CALIBRATION_FILE_NAME "UserCalibration.bin"
#define MUSHIMAX 15
#define FRUITMAX 3
#define TIMEMAX 930
#define GOTIME 450
#define OBJMAX 200
#define G -900.0

#define SAMPLE_XML_PATH "SamplesConfig.xml"

#define CHECK_RC(nRetVal, what)						\
if (nRetVal != XN_STATUS_OK)							\
{																\
printf("%s failed: %s\n", what, xnGetStatusString(nRetVal));\
return nRetVal;											\
}

//三次元ベクトル構造体: Vec_3D
typedef struct _Vec_3D
{
    double x, y, z;
    double w, h, d;
    double vx, vy, vz;
    double rx, ry, rz;
    double sx, sy, sz, sd;
    double ax, ay, az;
    double r, g, b, a;
    int status;
    int status2;
    int id;
    double theta;
    double value;
    int fruit;
    int cnt;
    int score;
} Vec_3D;

//関数名宣言
//Kinect関係
//std::map<XnUInt32, std::pair<XnCalibrationStatus, XnPoseDetectionStatus> > m_Errors;
int initKinect();
//void XN_CALLBACK_TYPE MyCalibrationInProgress(SkeletonCapability& capability, XnUserID id, XnCalibrationStatus calibrationError, void* pCookie);
//void XN_CALLBACK_TYPE MyPoseInProgress(PoseDetectionCapability& capability, const XnChar* strPose, XnUserID id, XnPoseDetectionStatus poseError, void* pCookie);
void CleanupExit();
void XN_CALLBACK_TYPE User_NewUser(UserGenerator& generator, XnUserID nId, void* pCookie);
void XN_CALLBACK_TYPE User_LostUser(UserGenerator& generator, XnUserID nId, void* pCookie);
void XN_CALLBACK_TYPE UserPose_PoseDetected(PoseDetectionCapability& capability, const XnChar* strPose, XnUserID nId, void* pCookie);
void XN_CALLBACK_TYPE UserCalibration_CalibrationStart(SkeletonCapability& capability, XnUserID nId, void* pCookie);
void XN_CALLBACK_TYPE UserCalibration_CalibrationComplete(SkeletonCapability& capability, XnUserID nId, XnCalibrationStatus eStatus, void* pCookie);
int getJoint();
void getDepthImage();

void display();
void initGL();
void resize(int w, int h);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void timer(int value);
void keyboard(unsigned char key, int x, int y);
Vec_3D vectorNormalize(Vec_3D v0);
void initCV();
void resetInsect(int i, int mode);
void resetFruit();
void dispNumber(int number);

//グローバル変数
//Kinect関連
Context g_Context;
ScriptNode g_scriptNode;
DepthGenerator g_DepthGenerator;
UserGenerator g_UserGenerator;
ImageGenerator g_ImageGenerator;
Player g_Player;
XnBool g_bNeedPose = FALSE;
XnChar g_strPose[20] = "";
XnBool g_bDrawBackground = TRUE;
XnBool g_bDrawPixels = TRUE;
XnBool g_bDrawSkeleton = TRUE;
XnBool g_bPrintID = TRUE;
XnBool g_bPrintState = TRUE;
XnBool g_bPause = false;
XnBool g_bRecord = false;
XnBool g_bQuit = false;
XnRGB24Pixel* g_pTexMap = NULL;
unsigned int g_nTexMapX = 0;
unsigned int g_nTexMapY = 0;
SceneMetaData sceneMD;
DepthMetaData depthMD;
ImageMetaData imageMD;

int winW, winH;  //ウィンドウサイズ
Vec_3D e, tg;  //視点，目標点
double eDegY, eDegX, eDist;  //視点の水平角，垂直角，距離
int mX, mY, mState, mButton;  //マウスクリック位置格納用
double fLate = 30;  //フレームレート
int jointNum = 15;  //Kinectで取得する関節の数
XnPoint3D t1[GL_WIN_TOTAL], pointPos[GL_WIN_TOTAL];  //深度情報，座標
XnRGB24Pixel pointCol[GL_WIN_TOTAL];  //色情報
XnPoint3D jointPos[MAX_USER][24];  //関節座標
double jointConf[MAX_USER][24];  //関節座標信頼度
XnLabel pointLabel[GL_WIN_TOTAL];  //人ラベル情報

IplImage *depthImage, *depthDispImage;
IplImage *textureImage;

Vec_3D fPoint[640][480];  //床頂点座標
GLdouble modelM[16], projM[16];  //変換用行列
GLint viewM[4];  //変換用行列
int faceW = 100, faceT = 50, faceB = 150;
Vec_3D piroPoint[2][4];
double piroLen = 2.0;
int animeNum[] = {4,10,9};

//虫
Vec_3D catchArea;
Vec_3D mushiPos[MUSHIMAX];
int mushiID[MUSHIMAX];
Vec_3D mushiPosReset;

Vec_3D fruitPos[FRUITMAX];

int gameMode = 0;  //0:開始前，1:プレイ中，2:Gameover
int startTime = TIMEMAX;
int score[] = {0,0};
int highScore = 0;
int scoreX[] = {0,0};
int piroCnt[] = {0,0};
int goTime = GOTIME;
int piroStatus[] = {0,0};

//スタートボタン
Vec_3D startPos;

//AL
void initAL();  //OpenAL初期化(OpenAL)
ALuint sourceMushi[MUSHIMAX];  //音源データ
ALuint bufferMushi[4];
ALuint sourceEat;

int kabeMushi = 50;

//20190702
double detectDist = 2500.0;  //人検出最長距離
int playerNum;

//破裂
Vec_3D objPos[OBJMAX];
int objNum = 0;

//メイン関数
int main(int argc, char **argv)
{
    //乱数初期化
    srand((unsigned)time(NULL));
    
    //Kinect関連初期化
    if (initKinect()>0)
        return 1;
    
    //初期化処理
    alutInit(&argc,argv);
    initAL();
    glutInit(&argc, argv);
    initGL();
    initCV();
    
    //イベント待ち無限ループ
    glutMainLoop();
    return 0;
}

//------------------------------------------------------- 関数 --------------------------------------------------------
void initAL()
{
    ALuint buffer;
    ALuint source;

    //初期化
    alListener3f(AL_POSITION, 0, 0, 0);  //マイク位置

    alGenBuffers(1, &buffer);  //バッファ生成
    buffer = alutCreateBufferFromFile("./wav/jungle.wav");
    alGenSources(1, &source);
    alSourcei(source, AL_BUFFER, buffer);  //音源にバッファを結び付け
    alSourcei(source, AL_LOOPING, AL_TRUE);
    alSourcef(source, AL_PITCH, 1.0);  //ピッチ設定
    alSourcef(source, AL_GAIN, 0.3);  //音量設定(0〜1)
    alSource3f(source, AL_POSITION, 0.0, 0.0, 0.0);
    alSourcePlay(source);
    alSourcef(source, AL_SEC_OFFSET, 0.0);  //再生開始位置

    buffer = alutCreateBufferFromFile("./wav/eat.wav");
    alGenSources(1, &sourceEat);
    alSourcei(sourceEat, AL_BUFFER, buffer);  //音源にバッファを結び付け
    alSourcei(sourceEat, AL_LOOPING, AL_FALSE);
    alSourcef(sourceEat, AL_PITCH, 3.0);  //ピッチ設定
    alSourcef(sourceEat, AL_GAIN, 1.0);  //音量設定(0〜1)
    alSource3f(sourceEat, AL_POSITION, 0.0, 0.0, 0.0);

    alGenBuffers(1, &bufferMushi[0]);  //バッファ生成
    alGenBuffers(1, &bufferMushi[1]);  //バッファ生成
    alGenBuffers(1, &bufferMushi[2]);  //バッファ生成
    
    bufferMushi[0] = alutCreateBufferFromFile("./wav/bee2.wav");
    bufferMushi[1] = alutCreateBufferFromFile("./wav/kumo2.wav");
    bufferMushi[2] = alutCreateBufferFromFile("./wav/kumo.wav");
    bufferMushi[3] = alutCreateBufferFromFile("./wav/imo.wav");

}

void initCV()
{
    //画像
    depthImage = cvCreateImage(cvSize(GL_WIN_SIZE_X, GL_WIN_SIZE_Y), IPL_DEPTH_32F, 1);  //深度値生データ
    depthDispImage = cvCreateImage(cvSize(GL_WIN_SIZE_X, GL_WIN_SIZE_Y), IPL_DEPTH_8U, 3);  //深度値表示用
}

void resetInsect(int i, int mode)
{
    mushiPos[i] = mushiPosReset;
    
    mushiPos[i].status = 1;  //0：食べられて消滅，1：飛行中，2：食べられ途中，3：フルーツ食べ中
    mushiPos[i].status2 = -1;  //どのプレイヤに食べられたか
    mushiID[i] = rand()%mode;
    mushiPos[i].sd = 1.0;
    mushiPos[i].cnt = rand()%animeNum[mushiID[i]];
    if (mushiID[i]==0) {
        mushiPos[i].x = catchArea.h*0.8*rand()/RAND_MAX-catchArea.h*0.8*0.5;
        mushiPos[i].y = catchArea.h;
        mushiPos[i].z = catchArea.z-10.0;
        mushiPos[i].sx = 50.0; mushiPos[i].sy = 50.0; mushiPos[i].sz = 1.0;
        mushiPos[i].id = pow(-1, rand()%2);
        mushiPos[i].ax = 1.0*rand()/RAND_MAX-0.5;
        mushiPos[i].ay = 2.0*rand()/RAND_MAX+1.0;
        mushiPos[i].az = 0.1*rand()/RAND_MAX+0.03;
        double len = 4.0*rand()/RAND_MAX+4.0;
        double theta = 2.0*M_PI*rand()/RAND_MAX;
        mushiPos[i].vx = len*cos(theta)*2;
        mushiPos[i].vy = len*sin(theta)*2;
        mushiPos[i].score = 5;
    }
    if (mushiID[i]==1) {
        mushiPos[i].x = catchArea.h*rand()/RAND_MAX-catchArea.h*0.5;
        mushiPos[i].y = catchArea.h;
        mushiPos[i].z = catchArea.z-10.0;
        mushiPos[i].sx = 120.0; mushiPos[i].sy = 120.0; mushiPos[i].sz = 1.0;
        mushiPos[i].rx = mushiPos[i].x;
        mushiPos[i].ry = mushiPos[i].y;
        mushiPos[i].rz = mushiPos[i].z;
        mushiPos[i].az = 10.0*rand()/RAND_MAX+70.0;
        mushiPos[i].ay = 4.5*rand()/RAND_MAX+1.5;
        mushiPos[i].ax = 50.0*rand()/RAND_MAX+50.0;
        mushiPos[i].x = mushiPos[i].rx+mushiPos[i].ax*sin(M_PI*mushiPos[i].az/180.0);
        mushiPos[i].y = mushiPos[i].ry-mushiPos[i].ax*sin(M_PI*mushiPos[i].az/180.0);
        mushiPos[i].id = 1;
        mushiPos[i].score = 15;
    }
    if (mushiID[i]==2) {
        mushiPos[i].sx = 120.0; mushiPos[i].sy = 120.0; mushiPos[i].sz = 1.0;
        if (rand()%2==0) {
            mushiPos[i].x = -catchArea.h;
            mushiPos[i].y = fruitPos[0].y+mushiPos[i].sy*0.35;
            mushiPos[i].z = fruitPos[0].z-1.0;
            mushiPos[i].vx = 1.0*rand()/RAND_MAX+0.5;
            mushiPos[i].id = 1;
        }
        else {
            mushiPos[i].x = catchArea.h;
            mushiPos[i].y = fruitPos[1].y+mushiPos[i].sy*0.35;
            mushiPos[i].z = fruitPos[1].z-1.0;
            mushiPos[i].vx = -(1.0*rand()/RAND_MAX+0.5);
            mushiPos[i].id = -1;
        }
        mushiPos[i].score = 10;
    }
    
    if (mode==3) {
        alDeleteSources(1, &sourceMushi[i]);
        alGenSources(1, &sourceMushi[i]);
        alSourcei(sourceMushi[i], AL_BUFFER, bufferMushi[mushiID[i]]);  //音源にバッファを結び付け
        alSourcei(sourceMushi[i], AL_LOOPING, AL_TRUE);
        alSourcef(sourceMushi[i], AL_PITCH, 0.4*rand()/RAND_MAX+0.8);  //ピッチ設定
        //alSourcef(sourceMushi[i], AL_BYTE_OFFSET, 1.0*rand()/RAND_MAX);  //再生開始位置
        alSourcef(sourceMushi[i], AL_GAIN, 0.3);  //音量設定(0〜1)
        alSource3f(sourceMushi[i], AL_POSITION, 0.0, 0.0, 0.0);
        alSourcePlay(sourceMushi[i]);
    }
    if (mode==1) {
        alDeleteSources(1, &sourceMushi[i]);
    }

}

void resetFruit()
{
    //フルーツの初期位置
    fruitPos[0].x = -0.2*catchArea.w;
    fruitPos[0].y = -0.35*catchArea.h;
    fruitPos[0].z = catchArea.z-5.0;
    fruitPos[0].sx = 120.0; fruitPos[0].sy = 120.0; fruitPos[0].sz = 1.0;
    fruitPos[0].value = 100.0;
    fruitPos[1].x = 0.05*catchArea.w;
    fruitPos[1].y = -0.25*catchArea.h;
    fruitPos[1].z = catchArea.z-5.0;
    fruitPos[1].sx = 110.0; fruitPos[1].sy = 110.0; fruitPos[1].sz = 1.0;
    fruitPos[1].value = 100.0;
    fruitPos[2].x = 0.3*catchArea.w;
    fruitPos[2].y = -0.3*catchArea.h;
    fruitPos[2].z = catchArea.z-5.0;
    fruitPos[2].sx = 100.0; fruitPos[2].sy = 100.0; fruitPos[2].sz = 1.0;
    fruitPos[2].value = 100.0;
}

void initGL()
{
    char fileName[100];
    
    //視点
    eDegY = 180.0; eDegX = 0.0; eDist = 2000.0;
    //目標点
    tg.x = 0.0; tg.y = 0.0; tg.z = 2000;
    
    //ウィンドウ
    winW = 1920*0.5; winH = 1200*0.5;
    glutInitWindowSize(winW, winH);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
    glutCreateWindow("KINECT");
    
    //コールバック関数の指定
    glutDisplayFunc(display);
    glutReshapeFunc(resize);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutTimerFunc(1000.0/fLate, timer, 0);
    glutKeyboardFunc(keyboard);
    
    //その他設定
    glClearColor(0.0, 0.0, 0.2, 1.0);
    glEnable(GL_DEPTH_TEST);  //Zバッファの有効化
    glEnable(GL_BLEND);
    glEnable(GL_NORMALIZE);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_ALPHA_TEST);  //アルファテスト有効化
    glAlphaFunc(GL_GREATER, 0.1);
    
    //光源
    glEnable(GL_LIGHTING);  //陰影付けの有効化
    glEnable(GL_LIGHT0);  //光源0の有効化
    
    //光源0の各種パラメータ設定
    GLfloat col[4];  //色指定用配列
    col[0] = 0.8; col[1] = 0.8; col[2] = 0.8; col[3] = 1.0;
    glLightfv(GL_LIGHT0, GL_DIFFUSE, col);  //拡散反射光
    col[0] = 0.2; col[1] = 0.2; col[2] = 0.2; col[3] = 1.0;
    glLightfv(GL_LIGHT0, GL_AMBIENT, col);  //環境光
    col[0] = 1.0; col[1] = 1.0; col[2] = 1.0; col[3] = 1.0;
    glLightfv(GL_LIGHT0, GL_SPECULAR, col);  //鏡面反射光
    
    //キャッチエリア
    catchArea.z = 1000.0;
    catchArea.w = 2.0*catchArea.z*atan(0.5*57.0*M_PI/180.0);
    catchArea.h = 2.0*catchArea.z*atan(0.5*43.0*M_PI/180.0);
    
    //フルーツリセット
    resetFruit();
    
    //虫の初期位置
    for (int i=0; i<MUSHIMAX; i++) {
        resetInsect(i, 1);
    }
    
    //スタートボタン
    startPos.x = 0.0; startPos.y = -270.0; startPos.z = catchArea.z-20.0;
    
    //破裂
    for (int i=0; i<OBJMAX; i++) {
        objPos[i].x = 0.0; objPos[i].y = 0.0; objPos[i].z = catchArea.z;
        objPos[i].vx = 0.0; objPos[i].vy = 0.0; objPos[i].vz = catchArea.z;
        objPos[i].r = 1.0; objPos[i].g = 1.0; objPos[i].b = 1.0; objPos[i].a = 1.0;
        objPos[i].sx = 15.0*rand()/RAND_MAX+15.0; objPos[i].sy = 15.0*rand()/RAND_MAX+15.0; objPos[i].vz = 1.0;
    }
    
    //テクスチャ
    textureImage = cvLoadImage("./png/tongue.png", CV_LOAD_IMAGE_UNCHANGED);
    glBindTexture(GL_TEXTURE_2D, 0);  //テクスチャオブジェクト生成
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);  //パラメータ設定
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);  //パラメータ設定
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, textureImage->width, textureImage->height, 0, GL_BGRA, GL_UNSIGNED_BYTE, textureImage->imageData);
    
    textureImage = cvLoadImage("./png/eda0.png", CV_LOAD_IMAGE_UNCHANGED);
    glBindTexture(GL_TEXTURE_2D, 1);  //テクスチャオブジェクト生成
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);  //パラメータ設定
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);  //パラメータ設定
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, textureImage->width, textureImage->height, 0, GL_BGRA, GL_UNSIGNED_BYTE, textureImage->imageData);
    
    textureImage = cvLoadImage("./png/eda1.png", CV_LOAD_IMAGE_UNCHANGED);
    glBindTexture(GL_TEXTURE_2D, 2);  //テクスチャオブジェクト生成
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);  //パラメータ設定
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);  //パラメータ設定
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, textureImage->width, textureImage->height, 0, GL_BGRA, GL_UNSIGNED_BYTE, textureImage->imageData);
    
    textureImage = cvLoadImage("./png/back1.png", CV_LOAD_IMAGE_UNCHANGED);
    glBindTexture(GL_TEXTURE_2D, 3);  //テクスチャオブジェクト生成
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);  //パラメータ設定
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);  //パラメータ設定
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, textureImage->width, textureImage->height, 0, GL_BGRA, GL_UNSIGNED_BYTE, textureImage->imageData);
    
    textureImage = cvLoadImage("./png/logo2.png", CV_LOAD_IMAGE_UNCHANGED);
    glBindTexture(GL_TEXTURE_2D, 4);  //テクスチャオブジェクト生成
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);  //パラメータ設定
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);  //パラメータ設定
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, textureImage->width, textureImage->height, 0, GL_BGRA, GL_UNSIGNED_BYTE, textureImage->imageData);

    textureImage = cvLoadImage("./png/go.png", CV_LOAD_IMAGE_UNCHANGED);
    glBindTexture(GL_TEXTURE_2D, 5);  //テクスチャオブジェクト生成
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);  //パラメータ設定
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);  //パラメータ設定
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, textureImage->width, textureImage->height, 0, GL_BGRA, GL_UNSIGNED_BYTE, textureImage->imageData);

    textureImage = cvLoadImage("./png/sc.png", CV_LOAD_IMAGE_UNCHANGED);
    glBindTexture(GL_TEXTURE_2D, 6);  //テクスチャオブジェクト生成
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);  //パラメータ設定
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);  //パラメータ設定
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, textureImage->width, textureImage->height, 0, GL_BGRA, GL_UNSIGNED_BYTE, textureImage->imageData);

    textureImage = cvLoadImage("./png/hs.png", CV_LOAD_IMAGE_UNCHANGED);
    glBindTexture(GL_TEXTURE_2D, 7);  //テクスチャオブジェクト生成
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);  //パラメータ設定
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);  //パラメータ設定
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, textureImage->width, textureImage->height, 0, GL_BGRA, GL_UNSIGNED_BYTE, textureImage->imageData);

    textureImage = cvLoadImage("./png/chame.png", CV_LOAD_IMAGE_UNCHANGED);
    glBindTexture(GL_TEXTURE_2D, 200);  //テクスチャオブジェクト生成
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);  //パラメータ設定
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);  //パラメータ設定
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, textureImage->width, textureImage->height, 0, GL_BGRA, GL_UNSIGNED_BYTE, textureImage->imageData);
    textureImage = cvLoadImage("./png/chame4.png", CV_LOAD_IMAGE_UNCHANGED);
    glBindTexture(GL_TEXTURE_2D, 201);  //テクスチャオブジェクト生成
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);  //パラメータ設定
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);  //パラメータ設定
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, textureImage->width, textureImage->height, 0, GL_BGRA, GL_UNSIGNED_BYTE, textureImage->imageData);

    textureImage = cvLoadImage("./png/start.png", CV_LOAD_IMAGE_UNCHANGED);
    glBindTexture(GL_TEXTURE_2D, 9);  //テクスチャオブジェクト生成
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);  //パラメータ設定
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);  //パラメータ設定
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, textureImage->width, textureImage->height, 0, GL_BGRA, GL_UNSIGNED_BYTE, textureImage->imageData);

    textureImage = cvLoadImage("./png/pi.png", CV_LOAD_IMAGE_UNCHANGED);
    glBindTexture(GL_TEXTURE_2D, 90);  //テクスチャオブジェクト生成
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);  //パラメータ設定
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);  //パラメータ設定
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, textureImage->width, textureImage->height, 0, GL_BGRA, GL_UNSIGNED_BYTE, textureImage->imageData);

    textureImage = cvLoadImage("./png/haretsu.png", CV_LOAD_IMAGE_UNCHANGED);
    glBindTexture(GL_TEXTURE_2D, 91);  //テクスチャオブジェクト生成
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);  //パラメータ設定
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);  //パラメータ設定
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, textureImage->width, textureImage->height, 0, GL_BGRA, GL_UNSIGNED_BYTE, textureImage->imageData);

    //ハエ
    for (int i=0; i<animeNum[0]; i++) {
        sprintf(fileName, "./png/f%02d.png", i);
        textureImage = cvLoadImage(fileName, CV_LOAD_IMAGE_UNCHANGED);
        glBindTexture(GL_TEXTURE_2D, 10+i);  //テクスチャオブジェクト生成
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);  //パラメータ設定
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);  //パラメータ設定
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, textureImage->width, textureImage->height, 0, GL_BGRA, GL_UNSIGNED_BYTE, textureImage->imageData);
    }
    
    //蜘蛛
    for (int i=0; i<animeNum[1]; i++) {
        sprintf(fileName, "./png/s%02d.png", i);
        textureImage = cvLoadImage(fileName, CV_LOAD_IMAGE_UNCHANGED);
        glBindTexture(GL_TEXTURE_2D, 20+i);  //テクスチャオブジェクト生成
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);  //パラメータ設定
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);  //パラメータ設定
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, textureImage->width, textureImage->height, 0, GL_BGRA, GL_UNSIGNED_BYTE, textureImage->imageData);
    }
    
    //イモムシ
    for (int i=0; i<animeNum[2]; i++) {
        sprintf(fileName, "./png/i%02d.png", i);
        textureImage = cvLoadImage(fileName, CV_LOAD_IMAGE_UNCHANGED);
        glBindTexture(GL_TEXTURE_2D, 30+i);  //テクスチャオブジェクト生成
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);  //パラメータ設定
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);  //パラメータ設定
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, textureImage->width, textureImage->height, 0, GL_BGRA, GL_UNSIGNED_BYTE, textureImage->imageData);
    }
    
    textureImage = cvLoadImage("./png/fruit0.png", CV_LOAD_IMAGE_UNCHANGED);
    glBindTexture(GL_TEXTURE_2D, 60);  //テクスチャオブジェクト生成
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);  //パラメータ設定
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);  //パラメータ設定
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, textureImage->width, textureImage->height, 0, GL_BGRA, GL_UNSIGNED_BYTE, textureImage->imageData);
    
    textureImage = cvLoadImage("./png/fruit1.png", CV_LOAD_IMAGE_UNCHANGED);
    glBindTexture(GL_TEXTURE_2D, 70);  //テクスチャオブジェクト生成
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);  //パラメータ設定
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);  //パラメータ設定
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, textureImage->width, textureImage->height, 0, GL_BGRA, GL_UNSIGNED_BYTE, textureImage->imageData);
    
    textureImage = cvLoadImage("./png/fruit2.png", CV_LOAD_IMAGE_UNCHANGED);
    glBindTexture(GL_TEXTURE_2D, 80);  //テクスチャオブジェクト生成
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);  //パラメータ設定
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);  //パラメータ設定
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, textureImage->width, textureImage->height, 0, GL_BGRA, GL_UNSIGNED_BYTE, textureImage->imageData);
    
    //数字
    for (int i=0; i<10; i++) {
        sprintf(fileName, "./png/num%d.png", i);
        textureImage = cvLoadImage(fileName, CV_LOAD_IMAGE_UNCHANGED);
        glBindTexture(GL_TEXTURE_2D, 100+i);  //テクスチャオブジェクト生成
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);  //パラメータ設定
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);  //パラメータ設定
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, textureImage->width, textureImage->height, 0, GL_BGRA, GL_UNSIGNED_BYTE, textureImage->imageData);
    }
}

//ディスプレイコールバック関数
void display()
{
    static double logoPhi = 0.0;
    
    //描画用バッファおよびZバッファの消去
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    //投影変換行列の設定
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();  //行列初期化
    gluPerspective(43.0, (double)winW/(double)winH, 50.0, 20000);
    
    //行列初期化(モデルビュー変換行列)
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    
    //視点視線の設定
    double eRad_y = M_PI*eDegY/180.0;
    double eRad_x = M_PI*eDegX/180.0;  //角度からラジアンに変換
    e.x = eDist*cos(eRad_x)*sin(eRad_y)+tg.x;
    e.y = eDist*sin(eRad_x)+tg.y;
    e.z = eDist*cos(eRad_x)*cos(eRad_y)+tg.z;
    gluLookAt(e.x, e.y, e.z, tg.x, tg.y, tg.z, 0.0, 1.0, 0.0);
    
    //モデルビュー変換行列・透視変換行列・ビューポート変換行列を取り出す
    glGetDoublev(GL_MODELVIEW_MATRIX, modelM);
    glGetDoublev(GL_PROJECTION_MATRIX, projM);
    glGetIntegerv(GL_VIEWPORT, viewM);
    
    GLfloat pos[4];  //座標指定用配列
    
    //光源0の位置指定
    pos[0] = 1300.0; pos[1] = 1800.0; pos[2] = -1200.0; pos[3] = 1.0;
    glLightfv(GL_LIGHT0, GL_POSITION, pos);
    
    GLfloat col[4], spe[4], shi[1];
    
    //------------------------------Kinectデータ取得，処理，表示------------------------------
    //データ取得準備
    XnStatus rc = XN_STATUS_OK;
    
    //フレーム読み込み
    rc = g_Context.WaitAnyUpdateAll();
    if (rc != XN_STATUS_OK) {
        printf("Read failed: %s\n", xnGetStatusString(rc));
        return;
    }
    
    //データ処理
    g_DepthGenerator.GetMetaData(depthMD);
    g_UserGenerator.GetUserPixels(0, sceneMD);
    g_ImageGenerator.GetMetaData(imageMD);
    
    //関節点"jointPos[]"取得
    int rtn = getJoint();
    //int rtn;
    
    //スキャン点"pointPos[GL_WIN_TOTAL]"と色"pointCol[]"取得
    getDepthImage();
    
    //pointPos[]のデータからdepthImage生成
    for (int i=0; i<GL_WIN_TOTAL; i++) {
        int N = i/GL_WIN_SIZE_X;
        int M = GL_WIN_SIZE_X-(i-N*GL_WIN_SIZE_X)-1;
        
        double s = pointPos[i].Z;
        
        if (s<50 || s>20000) s = 20000.0;
        
        CvScalar s2;
        s2.val[0]= s2.val[1]= s2.val[2]= 255-s/10.0;
        
        cvSetReal2D(depthImage, N, M, s);
        cvSet2D(depthDispImage, N, M, s2);
        
        //M = i-640*N;
        fPoint[M][N].x = -pointPos[i].X;
        fPoint[M][N].y = pointPos[i].Y;
        fPoint[M][N].z = s;
        fPoint[M][N].d = pointLabel[i];
        fPoint[M][N].r = pointCol[i].nRed/255.0;
        fPoint[M][N].g = pointCol[i].nGreen/255.0;
        fPoint[M][N].b = pointCol[i].nBlue/255.0;
        fPoint[M][N].a = 1.0;
    }
    
    //プレイヤー判定
    int playerID[] = {-1, -1};
    double distX[] = {detectDist, detectDist};
    int playerNumTemp = 0;
    for (int i=0; i<rtn; i++) {
        double dist = sqrt(pow(jointPos[i][0].Z,2)+pow(jointPos[i][0].X*2.0,2));
        if (jointConf[i][0]>0.1 && dist<distX[0]) {
            playerID[1] = playerID[0];
            playerID[0] = i;
            distX[1] = distX[0];
            distX[0] = dist;
            playerNumTemp++;
        }
        else if (jointConf[i][0]>0.1 && dist<distX[1]) {
            playerID[1] = i;
            distX[1] = dist;
            playerNumTemp++;
        }
    }
    
    if (playerNumTemp>2) playerNumTemp = 2;
    
    if (gameMode!=1) {
        playerNum = playerNumTemp;
    }

    if (playerNum==2 && playerNumTemp==2) {  //x座標が小さい方をpleyerID[0]
        if (jointPos[playerID[0]][0].X>jointPos[playerID[1]][0].X) {
            int tempID = playerID[0];
            playerID[0] = playerID[1];
            playerID[1] = tempID;
        }
    }
    
    for (int i=0; i<MIN(playerNum,playerNumTemp); i++) {
        piroPoint[i][0].x = -jointPos[playerID[i]][0].X; piroPoint[i][0].y = jointPos[playerID[i]][0].Y; piroPoint[i][0].z = jointPos[playerID[i]][0].Z;
    }
    
    //スキャン点描画
    glDisable(GL_LIGHTING);
    glBegin(GL_QUADS);
    for (int j=0; j<GL_WIN_SIZE_Y-1; j=j+1) {
        for (int i=0; i<GL_WIN_SIZE_X-1; i=i+1) {
            double aveZ = (fPoint[i][j].z+fPoint[i+1][j].z+fPoint[i+1][j+1].z+fPoint[i][j+1].z)*0.25;
            double varZ = (pow(aveZ-fPoint[i][j].z,2)+pow(aveZ-fPoint[i+1][j].z,2)+pow(aveZ-fPoint[i+1][j+1].z,2)+pow(aveZ-fPoint[i][j+1].z,2))*0.25;
            if (varZ<2000 && fPoint[i][j].d>0 && fPoint[i+1][j].d>0 && fPoint[i+1][j+1].d>0 && fPoint[i][j+1].d>0) {
                //if (varZ<2000) {
                glColor4d(fPoint[i][j].r, fPoint[i][j].g, fPoint[i][j].b, 0.7);
                glVertex3d(fPoint[i][j].x, fPoint[i][j].y, fPoint[i][j].z);
                glColor4d(fPoint[i+1][j].r, fPoint[i+1][j].g, fPoint[i+1][j].b, 0.7);
                glVertex3d(fPoint[i+1][j].x, fPoint[i+1][j].y, fPoint[i+1][j].z);
                glColor4d(fPoint[i+1][j+1].r, fPoint[i+1][j+1].g, fPoint[i+1][j+1].b, 0.7);
                glVertex3d(fPoint[i+1][j+1].x, fPoint[i+1][j+1].y, fPoint[i+1][j+1].z);
                glColor4d(fPoint[i][j+1].r, fPoint[i][j+1].g, fPoint[i][j+1].b, 0.7);
                glVertex3d(fPoint[i][j+1].x, fPoint[i][j+1].y, fPoint[i][j+1].z);
            }
        }
    }
    glEnd();
    
    //背景
    glDisable(GL_LIGHTING);
    glColor4d(1.0, 1.0, 1.0, 1.0);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 3);  //テクスチャオブジェクト生成
    glPushMatrix();
    glTranslated(0.0, 0.0, 5000.0);
    glScaled(1920.0*4, 1080.0*4, 1.0);
    glBegin(GL_QUADS);
    glTexCoord2d(0.0, 0.0);
    glVertex3d(0.5, 0.5, 0.0);
    glTexCoord2d(1.0, 0.0);
    glVertex3d(-0.5, 0.5, 0.0);
    glTexCoord2d(1.0, 1.0);
    glVertex3d(-0.5, -0.5, 0.0);
    glTexCoord2d(0.0, 1.0);
    glVertex3d(0.5, -0.5, 0.0);
    glEnd();
    glPopMatrix();
    glDisable(GL_TEXTURE_2D);
    
    //フルーツ
    glDisable(GL_LIGHTING);
    glEnable(GL_TEXTURE_2D);
    for (int i=0; i<FRUITMAX; i++) {
        glColor4d(fruitPos[i].value/100.0, fruitPos[i].value/100.0, 1.0, 1.0);
        glBindTexture(GL_TEXTURE_2D, (i+6)*10);  //テクスチャオブジェクト生成
        glPushMatrix();
        glTranslated(fruitPos[i].x, fruitPos[i].y, fruitPos[i].z);
        glScaled(fruitPos[i].sx, fruitPos[i].sy, fruitPos[i].sz);
        glBegin(GL_QUADS);
        glTexCoord2d(0.0, 0.0);
        glVertex3d(0.5, 0.5, 0.0);
        glTexCoord2d(1.0, 0.0);
        glVertex3d(-0.5, 0.5, 0.0);
        glTexCoord2d(1.0, 1.0);
        glVertex3d(-0.5, -0.5, 0.0);
        glTexCoord2d(0.0, 1.0);
        glVertex3d(0.5, -0.5, 0.0);
        glEnd();
        glPopMatrix();
    }
    glDisable(GL_TEXTURE_2D);
    
    //カメレオン
    if (gameMode>0) {
        for (int i=0; i<MIN(playerNum,playerNumTemp); i++) {
            glDisable(GL_LIGHTING);
            glEnable(GL_TEXTURE_2D);
            glColor4d(1.0, 1.0, 1.0, 1.0);
            glBindTexture(GL_TEXTURE_2D, 200+i);  //テクスチャオブジェクト生成
            glPushMatrix();
            glTranslated(piroPoint[i][0].x*(piroPoint[i][0].z-200.0)/piroPoint[i][0].z, (piroPoint[i][0].y+80.0)*(piroPoint[i][0].z-200.0)/piroPoint[i][0].z+50.0, piroPoint[i][0].z-200.0);
            glScaled(360.0*0.8, 520.0*0.8, 1.0);
            glBegin(GL_QUADS);
            glTexCoord2d(0.0, 0.0);
            glVertex3d(0.5, 0.5, 0.0);
            glTexCoord2d(1.0, 0.0);
            glVertex3d(-0.5, 0.5, 0.0);
            glTexCoord2d(1.0, 1.0);
            glVertex3d(-0.5, -0.5, 0.0);
            glTexCoord2d(0.0, 1.0);
            glVertex3d(0.5, -0.5, 0.0);
            glEnd();
            glPopMatrix();
            glDisable(GL_TEXTURE_2D);
        }
    }

    //枝
    glDisable(GL_LIGHTING);
    glEnable(GL_TEXTURE_2D);
    glColor4d(1.0, 1.0, 1.0, 1.0);
    glPushMatrix();
    glTranslated((fruitPos[0].x-catchArea.w*0.5*1.5)*0.5, fruitPos[0].y+30, fruitPos[0].z+1);
    glScaled(catchArea.w*0.5*1.5+fruitPos[0].x, 300.0, 1.0);
    glBindTexture(GL_TEXTURE_2D, 1);
    glBegin(GL_QUADS);
    glTexCoord2d(0.0, 0.0);
    glVertex3d(0.5, 0.5, 0.0);
    glTexCoord2d(1.0, 0.0);
    glVertex3d(-0.5, 0.5, 0.0);
    glTexCoord2d(1.0, 1.0);
    glVertex3d(-0.5, -0.5, 0.0);
    glTexCoord2d(0.0, 1.0);
    glVertex3d(0.5, -0.5, 0.0);
    glEnd();
    glPopMatrix();
    glPushMatrix();
    glTranslated((fruitPos[1].x+catchArea.w*0.5*1.5)*0.5, fruitPos[1].y+30, fruitPos[1].z+1);
    glScaled(catchArea.w*0.5*1.5-fruitPos[1].x, 300.0, 1.0);
    glBindTexture(GL_TEXTURE_2D, 2);
    glBegin(GL_QUADS);
    glTexCoord2d(0.0, 0.0);
    glVertex3d(0.5, 0.5, 0.0);
    glTexCoord2d(1.0, 0.0);
    glVertex3d(-0.5, 0.5, 0.0);
    glTexCoord2d(1.0, 1.0);
    glVertex3d(-0.5, -0.5, 0.0);
    glTexCoord2d(0.0, 1.0);
    glVertex3d(0.5, -0.5, 0.0);
    glEnd();
    glPopMatrix();
    glDisable(GL_TEXTURE_2D);
    
    
    //笛先端検出
    for (int i=0; i<MIN(playerNum,playerNumTemp); i++) {
        GLdouble winX, winY, winZ;
        gluProject(jointPos[playerID[i]][0].X, jointPos[playerID[i]][0].Y, jointPos[playerID[i]][0].Z, modelM, projM, viewM, &winX, &winY, &winZ);
        //printf("win = (%f, %f, %f)\n", winX, winY, winZ);
        winX = GL_WIN_SIZE_X*winX/winW;
        winY = GL_WIN_SIZE_Y*(winH-winY)/winH;
        
        int searchL = fmax(0+kabeMushi, winX-faceW);
        int searchR = fmin(GL_WIN_SIZE_X-kabeMushi, winX+faceW);
        int searchT = fmax(0, winY-faceT);
        int searchB = fmin(GL_WIN_SIZE_Y, winY+faceB);
        
        double distMin = 20000.0;
        int iMin = 0, jMin = 0;
        for (int j=searchT; j<searchB; j++) {
            for (int i=searchL; i<searchR; i++) {
                if (fPoint[i][j].z<distMin) {
                    iMin = i; jMin = j;
                    distMin = fPoint[i][j].z;
                }
            }
        }
        
        //printf("distMin = %f\n", distMin);
        cvRectangle(depthDispImage, cvPoint(searchL, searchT), cvPoint(searchR, searchB), cv::Scalar(0,0,255), 1, 8);
        cvCircle(depthDispImage, cvPoint(iMin, jMin), 2, cv::Scalar(0,255,0), -1, 8);
        piroPoint[i][1] = fPoint[iMin][jMin];
    }
    
    
    //舌
    for (int i=0; i<MIN(playerNum,playerNumTemp); i++) {
        piroPoint[i][2].x = piroPoint[i][1].x-piroPoint[i][0].x; piroPoint[i][2].y = piroPoint[i][1].y-piroPoint[i][0].y; piroPoint[i][2].z = piroPoint[i][1].z-piroPoint[i][0].z;
        piroLen = fmax(sqrt(pow(piroPoint[i][2].x,2.0)+pow(piroPoint[i][2].y,2.0)+pow(piroPoint[i][2].z,2.0))-350.0, 0.0)*5.0;

        piroPoint[i][2] = vectorNormalize(piroPoint[i][2]);
        piroPoint[i][3].x = piroPoint[i][1].x+piroLen*piroPoint[i][2].x; piroPoint[i][3].y = piroPoint[i][1].y+piroLen*piroPoint[i][2].y; piroPoint[i][3].z = piroPoint[i][1].z+piroLen*piroPoint[i][2].z;
        
        if (piroPoint[i][3].z<catchArea.z && piroPoint[i][3].z>catchArea.z*0.5)
            piroStatus[i] = 1;
        else if (piroPoint[i][3].z>catchArea.z && piroStatus[i]==1 && gameMode==1) {
            piroStatus[i] = 2;
            piroCnt[i]++;
        }
        else
            piroStatus[i] = 0;

        double tVal = (catchArea.z-piroPoint[i][1].z)/piroPoint[i][2].z;
        Vec_3D markPos;
        markPos.x = piroPoint[i][1].x+tVal*piroPoint[i][2].x;
        markPos.y = piroPoint[i][1].y+tVal*piroPoint[i][2].y;
        markPos.z = piroPoint[i][1].z+tVal*piroPoint[i][2].z;
       
        glDisable(GL_TEXTURE_2D);
        glDisable(GL_LIGHTING);
        glColor4d(1.0, 1.0, 1.0, 1.0);
        glPushMatrix();
        glTranslated(markPos.x, markPos.y, markPos.z);
        glBegin(GL_LINE_LOOP);
        glVertex3d(-50.0, -50.0, 0.0);
        glVertex3d(50.0, -50.0, 0.0);
        glVertex3d(50.0, 50.0, 0.0);
        glVertex3d(-50.0, 50.0, 0.0);
        glEnd();
        glPopMatrix();
        
        glEnable(GL_TEXTURE_2D);  //テクスチャ有効化
        glBindTexture(GL_TEXTURE_2D, 0);  //0830:テクスチャオブジェクト生成
        glDisable(GL_LIGHTING);
        glColor4d(1.0, 1.0, 1.0, 1.0);
        glBegin(GL_QUADS);
        glTexCoord2d(0.0, 0.0);
        glVertex3d(piroPoint[i][1].x-30.0, piroPoint[i][1].y, piroPoint[i][1].z);
        glTexCoord2d(1.0, 0.0);
        glVertex3d(piroPoint[i][1].x+30.0, piroPoint[i][1].y, piroPoint[i][1].z);
        glTexCoord2d(1.0, 1.0);
        glVertex3d(piroPoint[i][3].x+50.0, piroPoint[i][3].y, piroPoint[i][3].z);
        glTexCoord2d(0.0, 1.0);
        glVertex3d(piroPoint[i][3].x-50.0, piroPoint[i][3].y, piroPoint[i][3].z);
        glEnd();
        glDisable(GL_TEXTURE_2D);  //テクスチャ有効化
    }
    
    //破裂オブジェクト
    glDisable(GL_LIGHTING);
    glEnable(GL_TEXTURE_2D);  //テクスチャ有効化
    glBindTexture(GL_TEXTURE_2D, 91);  //0830:テクスチャオブジェクト生成
    for (int i=0; i<OBJMAX; i++) {
        glColor4d(objPos[i].r, objPos[i].g, objPos[i].b, objPos[i].a);
        glPushMatrix();
        glTranslated(objPos[i].x, objPos[i].y, objPos[i].z);
        glScaled(objPos[i].sx, objPos[i].sy, objPos[i].sz);
        glBegin(GL_QUADS);
        glTexCoord2d(0.0, 0.0);
        glVertex3d(0.5, 0.5, 0.0);
        glTexCoord2d(1.0, 0.0);
        glVertex3d(-0.5, 0.5, 0.0);
        glTexCoord2d(1.0, 1.0);
        glVertex3d(-0.5, -0.5, 0.0);
        glTexCoord2d(0.0, 1.0);
        glVertex3d(0.5, -0.5, 0.0);
        glEnd();
        glPopMatrix();
        
        objPos[i].x += objPos[i].vx/fLate; objPos[i].y += objPos[i].vy/fLate; objPos[i].z += objPos[i].vz/fLate;
        if (objPos[i].y<-5000.0) objPos[i].y = -5000.0;
        objPos[i].vy += G/fLate;
        objPos[i].a *= 0.95;
        objPos[i].sx *= 0.95; objPos[i].sy *= 0.95;
    }
    glDisable(GL_TEXTURE_2D);  //テクスチャ有効化

    //虫表示
    glDisable(GL_LIGHTING);
    for (int i=0; i<MUSHIMAX; i++) {
        
        if (mushiPos[i].status>0) {
            glEnable(GL_TEXTURE_2D);
            glBindTexture(GL_TEXTURE_2D, (mushiID[i]+1)*10+mushiPos[i].cnt);  //テクスチャオブジェクト生成
            glColor4d(1.0, 1.0, 1.0, 1.0);
            glPushMatrix();
            glTranslated(mushiPos[i].x, mushiPos[i].y, mushiPos[i].z);
            glRotated(mushiPos[i].theta, 0.0, 0.0, 1.0);
            glScaled(mushiPos[i].sx*mushiPos[i].sd, mushiPos[i].sy*mushiPos[i].sd, mushiPos[i].sz*mushiPos[i].sd);
            glBegin(GL_QUADS);
            glTexCoord2d(0.0, 0.0);
            glVertex3d(0.5, 0.5, 0.0);
            glTexCoord2d(1.0*mushiPos[i].id, 0.0);
            glVertex3d(-0.5, 0.5, 0.0);
            glTexCoord2d(1.0*mushiPos[i].id, 1.0);
            glVertex3d(-0.5, -0.5, 0.0);
            glTexCoord2d(0.0, 1.0);
            glVertex3d(0.5, -0.5, 0.0);
            glEnd();
            glPopMatrix();
            glDisable(GL_TEXTURE_2D);
            
            if (mushiID[i]==1) {
                glColor4d(1.0, 1.0, 1.0, 1.0);
                glLineWidth(1.0);
                glBegin(GL_LINES);
                glVertex3d(mushiPos[i].x, mushiPos[i].y, mushiPos[i].z);
                glVertex3d(mushiPos[i].rx, mushiPos[i].ry, mushiPos[i].rz);
                glEnd();
            }
            
            mushiPos[i].cnt++;
            if (mushiPos[i].cnt==animeNum[mushiID[i]]) mushiPos[i].cnt = 0;
        }
    }
    
    cvShowImage("DEPTH", depthDispImage);
    
    if (gameMode>0) {
        //舌と虫の状態
        for (int j=0; j<MIN(playerNum,playerNumTemp); j++) {
            for (int i=0; i<MUSHIMAX; i++) {
                double len = sqrt(pow(piroPoint[j][3].x-mushiPos[i].x,2.0)+pow(piroPoint[j][3].y-mushiPos[i].y,2.0)+pow(piroPoint[j][3].z-mushiPos[i].z,2.0));
                
                if (len<150.0 && (mushiPos[i].status==1 || mushiPos[i].status==3)) {
                    //舌に捕まった
                    mushiPos[i].status = 2;
                    mushiPos[i].status2 = j;
                    alSourcePlay(sourceEat);
                    for (int i=0; i<30; i++) {
                        objPos[objNum].x = mushiPos[i].x; objPos[objNum].y = mushiPos[i].y; objPos[objNum].z = catchArea.z*1.5;
                        objPos[objNum].sx = 15.0*rand()/RAND_MAX+15.0; objPos[objNum].sy = 15.0*rand()/RAND_MAX+15.0; objPos[objNum].vz = 1.0;
                        double len = 100.0*rand()/RAND_MAX+100.0;
                        double theta = 2.0*M_PI*rand()/RAND_MAX;
                        objPos[objNum].vx = len*cos(theta); objPos[objNum].vy = len*sin(theta);
                        objPos[objNum].a = 1.0;
                        objNum++;
                        if (objNum==OBJMAX) objNum = 0;
                    }
                }
            }
        }
        
        //舌に捕まっている虫
        for (int i=0; i<MUSHIMAX; i++) {
            if (mushiPos[i].status==2) {
                //舌に捕まっている状態
                mushiPos[i].x = piroPoint[mushiPos[i].status2][3].x;
                mushiPos[i].y = piroPoint[mushiPos[i].status2][3].y;
                mushiPos[i].z = piroPoint[mushiPos[i].status2][3].z;
                mushiPos[i].sd = 2.0;
                mushiPos[i].vx = 0.0;
                mushiPos[i].vy = 0.0;
                mushiPos[i].vz = 0.0;
                mushiPos[i].theta += 5.0;
                mushiPos[i].status = 2;
                if (piroLen<100) {
                    //食べられた
                    if (gameMode==1) score[mushiPos[i].status2] += mushiPos[i].score;
                    //mushiPos[i].status2 = -1;
                    mushiPos[i].status = 0;
                    alDeleteSources(1, &sourceMushi[i]);
                    resetInsect(i, 3);
                }
            }
        }

        //虫更新
        for (int i=0; i<MUSHIMAX; i++) {
            if (mushiPos[i].status==1) {
                mushiPos[i].x += mushiPos[i].vx;
                mushiPos[i].y += mushiPos[i].vy;
                mushiPos[i].z += mushiPos[i].vz;
                if (mushiID[i]==0) {
                    mushiPos[i].x += mushiPos[i].ax;
                    mushiPos[i].y -= mushiPos[i].ay;
                    mushiPos[i].vx += mushiPos[i].vy*mushiPos[i].az*mushiPos[i].id;
                    mushiPos[i].vy -= mushiPos[i].vx*mushiPos[i].az*mushiPos[i].id;
                    double v = sqrt(pow(mushiPos[i].vx,2)+pow(mushiPos[i].vy,2));
                    mushiPos[i].theta = 180.0*acos(mushiPos[i].vx/v)/M_PI;
                    if (mushiPos[i].vy<0) mushiPos[i].theta *= -1;
                    mushiPos[i].theta += 90;
                    if (mushiPos[i].y<-catchArea.h) {
                        mushiPos[i].x = 0.0;
                        mushiPos[i].y = catchArea.h;
                    }
                    if (rand()%10==0 && mushiPos[i].vy<0) {
                        mushiPos[i].id *= -1;
                        mushiPos[i].ax *= -1;
                        mushiPos[i].vx *= -1;
                    }
                    alSourcef(sourceMushi[i], AL_PITCH, 1.0-mushiPos[i].vy/20.0);  //ピッチ設定

                }
                if (mushiID[i]==1) {
                    double theta = asin((mushiPos[i].x-mushiPos[i].rx)/mushiPos[i].ax);
                    double vv = pow(mushiPos[i].vx,2)+pow(mushiPos[i].vy,2);
                    double a = vv/mushiPos[i].ax;
                    double ax = -a*sin(theta);
                    double ay = a*cos(theta)-sqrt(mushiPos[i].ax)*5.0;
                    mushiPos[i].vx += ax;
                    mushiPos[i].vy = ay;
                    mushiPos[i].y = mushiPos[i].ry-mushiPos[i].ax*cos(theta);
                    mushiPos[i].ax += mushiPos[i].ay;
                    if (mushiPos[i].ax>catchArea.h*1.5) mushiPos[i].ax = catchArea.h*1.5;
                    mushiPos[i].theta = 180.0*theta/M_PI;
                    
                    
                    //mushiPos[i].vx -= sin(theta);
                    //if (mushiPos[i].y>-0.5*catchArea.h)
                    //    mushiPos[i].y -= mushiPos[i].ay;
                    //mushiPos[i].ay += 10.0;
                }
                
                //フルーツ到達判定
                for (int j=0; j<FRUITMAX; j++) {
                    double len = sqrt(pow(mushiPos[i].x-fruitPos[j].x,2)+pow(mushiPos[i].y-fruitPos[j].y,2));
                    if (len<fruitPos[j].sx*0.5 && mushiPos[i].status == 1) {
                        mushiPos[i].status = 3;
                        mushiPos[i].fruit = j;
                        alDeleteSources(1, &sourceMushi[i]);
                        alGenSources(1, &sourceMushi[i]);
                        alSourcei(sourceMushi[i], AL_BUFFER, bufferMushi[3]);  //音源にバッファを結び付け
                        alSourcei(sourceMushi[i], AL_LOOPING, AL_TRUE);
                        alSourcef(sourceMushi[i], AL_PITCH, 1.0*rand()/RAND_MAX+3.0);  //ピッチ設定
                        alSourcef(sourceMushi[i], AL_GAIN, 1.0);  //音量設定(0〜1)
                        alSourcePlay(sourceMushi[i]);
                    }
                }
            }
            alSource3f(sourceMushi[i], AL_POSITION, -mushiPos[i].x/500.0, mushiPos[i].y/500.0, 1.0);
            double lenM = sqrt(pow(mushiPos[i].x,2)+pow(mushiPos[i].y,2));
        }
        
        //フルーツ食べられる
        for (int i=0; i<MUSHIMAX; i++) {
            if (mushiPos[i].status==3) {
                fruitPos[mushiPos[i].fruit].value -= 0.5;
                if (fruitPos[mushiPos[i].fruit].value<0)
                    fruitPos[mushiPos[i].fruit].value = 0;
            }
        }
        
        //タイマー
        int num = startTime/30;
        glPushMatrix();
        glTranslated(-500.0, 360.0, catchArea.z-20.0);
        dispNumber(num);
        glPopMatrix();
    }
    else if (gameMode==0) {
        //虫更新
        for (int i=0; i<MUSHIMAX; i++) {
            if (mushiPos[i].status==1) {
                mushiPos[i].x += mushiPos[i].vx;
                mushiPos[i].y += mushiPos[i].vy;
                mushiPos[i].z += mushiPos[i].vz;
                if (mushiID[i]==0) {
                    mushiPos[i].x += mushiPos[i].ax;
                    mushiPos[i].y -= mushiPos[i].ay;
                    mushiPos[i].vx += mushiPos[i].vy*mushiPos[i].az*mushiPos[i].id;
                    mushiPos[i].vy -= mushiPos[i].vx*mushiPos[i].az*mushiPos[i].id;
                    double v = sqrt(pow(mushiPos[i].vx,2)+pow(mushiPos[i].vy,2));
                    mushiPos[i].theta = 180.0*acos(mushiPos[i].vx/v)/M_PI;
                    if (mushiPos[i].vy<0) mushiPos[i].theta *= -1;
                    mushiPos[i].theta += 90;
                    if (mushiPos[i].y<-catchArea.h) {
                        mushiPos[i].x = 0.0;
                        mushiPos[i].y = catchArea.h;
                    }
                    if (rand()%10==0 && mushiPos[i].vy<0) {
                        mushiPos[i].id *= -1;
                        mushiPos[i].ax *= -1;
                        mushiPos[i].vx *= -1;
                    }
                }
                if (mushiID[i]==1) {
                    double theta = asin((mushiPos[i].x-mushiPos[i].rx)/mushiPos[i].ax);
                    double vv = pow(mushiPos[i].vx,2)+pow(mushiPos[i].vy,2);
                    double a = vv/mushiPos[i].ax;
                    double ax = -a*sin(theta);
                    double ay = a*cos(theta)-sqrt(mushiPos[i].ax)*5.0;
                    mushiPos[i].vx += ax;
                    mushiPos[i].vy = ay;
                    mushiPos[i].y = mushiPos[i].ry-mushiPos[i].ax*cos(theta);
                    mushiPos[i].ax += mushiPos[i].ay;
                    if (mushiPos[i].ax>catchArea.h*1.5) mushiPos[i].ax = catchArea.h*1.5;
                    mushiPos[i].theta = 180.0*theta/M_PI;
                    
                }
            }
        }
    }

    if (gameMode==0) {
        for (int j=0; j<MIN(playerNum,playerNumTemp); j++) {
            double len = sqrt(pow(piroPoint[j][3].x-startPos.x,2.0)+pow(piroPoint[j][3].y-startPos.y,2.0)+pow(piroPoint[j][3].z-startPos.z,2.0));
            
            if (len<200.0) {
                //舌に捕まった
                resetFruit();
                for (int i=0; i<MUSHIMAX; i++) {
                    resetInsect(i, 3);
                }
                gameMode = 1;
                startTime = TIMEMAX;
                score[0] = 0;
                piroCnt[0] = 0;
                score[1] = 0;
                piroCnt[1] = 0;
            }
        }

        //ロゴ
        glDisable(GL_LIGHTING);
        glColor4d(1.0, 1.0, 1.0, 1.0);
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, 4);  //テクスチャオブジェクト生成
        glPushMatrix();
        glTranslated(0.0, 30.0, catchArea.z-20.0);
        glRotated(10.0*sin(logoPhi), 0.0, 0.0, 1.0);
        glScaled(640.0, 480.0, 1.0);
        glBegin(GL_QUADS);
        glTexCoord2d(0.0, 0.0);
        glVertex3d(0.5, 0.5, 0.0);
        glTexCoord2d(1.0, 0.0);
        glVertex3d(-0.5, 0.5, 0.0);
        glTexCoord2d(1.0, 1.0);
        glVertex3d(-0.5, -0.5, 0.0);
        glTexCoord2d(0.0, 1.0);
        glVertex3d(0.5, -0.5, 0.0);
        glEnd();
        glPopMatrix();
        glDisable(GL_TEXTURE_2D);
        //
        glDisable(GL_LIGHTING);
        glColor4d(1.0, 1.0, 1.0, 1.0);
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, 9);  //テクスチャオブジェクト生成
        glPushMatrix();
        glTranslated(startPos.x, startPos.y, startPos.z);
        glScaled(240.0, 80.0, 1.0);
        glBegin(GL_QUADS);
        glTexCoord2d(0.0, 0.0);
        glVertex3d(0.5, 0.5, 0.0);
        glTexCoord2d(1.0, 0.0);
        glVertex3d(-0.5, 0.5, 0.0);
        glTexCoord2d(1.0, 1.0);
        glVertex3d(-0.5, -0.5, 0.0);
        glTexCoord2d(0.0, 1.0);
        glVertex3d(0.5, -0.5, 0.0);
        glEnd();
        glPopMatrix();
        glDisable(GL_TEXTURE_2D);
    }
    else if (gameMode==2) {
        //Game Over
        glDisable(GL_LIGHTING);
        glColor4d(1.0, 1.0, 1.0, 1.0);
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, 5);  //テクスチャオブジェクト生成
        glPushMatrix();
        glTranslated(0.0, 0.0, catchArea.z-20.0);
        glScaled(640.0, 120.0, 1.0);
        glBegin(GL_QUADS);
        glTexCoord2d(0.0, 0.0);
        glVertex3d(0.5, 0.5, 0.0);
        glTexCoord2d(1.0, 0.0);
        glVertex3d(-0.5, 0.5, 0.0);
        glTexCoord2d(1.0, 1.0);
        glVertex3d(-0.5, -0.5, 0.0);
        glTexCoord2d(0.0, 1.0);
        glVertex3d(0.5, -0.5, 0.0);
        glEnd();
        glPopMatrix();
        glDisable(GL_TEXTURE_2D);
    }

    for (int i=0; i<2; i++) {
        scoreX[i] = score[i]+piroCnt[i];
        if (highScore<scoreX[i]) highScore = scoreX[i];
    }
    
    //HighScore
    glDisable(GL_LIGHTING);
    glColor4d(1.0, 1.0, 1.0, 1.0);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 7);
    glPushMatrix();
    glTranslated(520.0, 360.0, catchArea.z-20.0);
    glScaled(150.0, 28.0, 1.0);
    glBegin(GL_QUADS);
    glTexCoord2d(0.0, 0.0);
    glVertex3d(0.5, 0.5, 0.0);
    glTexCoord2d(1.0, 0.0);
    glVertex3d(-0.5, 0.5, 0.0);
    glTexCoord2d(1.0, 1.0);
    glVertex3d(-0.5, -0.5, 0.0);
    glTexCoord2d(0.0, 1.0);
    glVertex3d(0.5, -0.5, 0.0);
    glEnd();
    glPopMatrix();
    glDisable(GL_TEXTURE_2D);
    glPushMatrix();
    glTranslated(420.0, 360.0, catchArea.z-20.0);
    dispNumber(highScore);
    glPopMatrix();

    //Score0
    glDisable(GL_LIGHTING);
    glColor4d(0.6, 1.0, 0.6, 1.0);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 6);
    glPushMatrix();
    glTranslated(520.0, 330.0, catchArea.z-20.0);
    glScaled(86.0, 28.0, 1.0);
    glBegin(GL_QUADS);
    glTexCoord2d(0.0, 0.0);
    glVertex3d(0.5, 0.5, 0.0);
    glTexCoord2d(1.0, 0.0);
    glVertex3d(-0.5, 0.5, 0.0);
    glTexCoord2d(1.0, 1.0);
    glVertex3d(-0.5, -0.5, 0.0);
    glTexCoord2d(0.0, 1.0);
    glVertex3d(0.5, -0.5, 0.0);
    glEnd();
    glPopMatrix();
    glDisable(GL_TEXTURE_2D);
    glPushMatrix();
    glTranslated(420.0, 330.0, catchArea.z-20.0);
    dispNumber(scoreX[0]);
    glPopMatrix();

    //Piro0
    glDisable(GL_LIGHTING);
    glColor4d(0.6, 1.0, 0.6, 1.0);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 90);
    glPushMatrix();
    glTranslated(520.0, 300.0, catchArea.z-20.0);
    glScaled(120.0, 28.0, 1.0);
    glBegin(GL_QUADS);
    glTexCoord2d(0.0, 0.0);
    glVertex3d(0.5, 0.5, 0.0);
    glTexCoord2d(1.0, 0.0);
    glVertex3d(-0.5, 0.5, 0.0);
    glTexCoord2d(1.0, 1.0);
    glVertex3d(-0.5, -0.5, 0.0);
    glTexCoord2d(0.0, 1.0);
    glVertex3d(0.5, -0.5, 0.0);
    glEnd();
    glPopMatrix();
    glDisable(GL_TEXTURE_2D);
    glPushMatrix();
    glTranslated(420.0, 300.0, catchArea.z-20.0);
    dispNumber(piroCnt[0]);
    glPopMatrix();

    //Score1
    glDisable(GL_LIGHTING);
    glColor4d(1.0, 0.3, 0.3, 1.0);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 6);
    glPushMatrix();
    glTranslated(520.0, 270.0, catchArea.z-20.0);
    glScaled(86.0, 28.0, 1.0);
    glBegin(GL_QUADS);
    glTexCoord2d(0.0, 0.0);
    glVertex3d(0.5, 0.5, 0.0);
    glTexCoord2d(1.0, 0.0);
    glVertex3d(-0.5, 0.5, 0.0);
    glTexCoord2d(1.0, 1.0);
    glVertex3d(-0.5, -0.5, 0.0);
    glTexCoord2d(0.0, 1.0);
    glVertex3d(0.5, -0.5, 0.0);
    glEnd();
    glPopMatrix();
    glDisable(GL_TEXTURE_2D);
    glPushMatrix();
    glTranslated(420.0, 270.0, catchArea.z-20.0);
    dispNumber(scoreX[1]);
    glPopMatrix();
    
    //Piro1
    glDisable(GL_LIGHTING);
    glColor4d(1.0, 0.3, 0.3, 1.0);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 90);
    glPushMatrix();
    glTranslated(520.0, 240.0, catchArea.z-20.0);
    glScaled(120.0, 28.0, 1.0);
    glBegin(GL_QUADS);
    glTexCoord2d(0.0, 0.0);
    glVertex3d(0.5, 0.5, 0.0);
    glTexCoord2d(1.0, 0.0);
    glVertex3d(-0.5, 0.5, 0.0);
    glTexCoord2d(1.0, 1.0);
    glVertex3d(-0.5, -0.5, 0.0);
    glTexCoord2d(0.0, 1.0);
    glVertex3d(0.5, -0.5, 0.0);
    glEnd();
    glPopMatrix();
    glDisable(GL_TEXTURE_2D);
    glPushMatrix();
    glTranslated(420.0, 240.0, catchArea.z-20.0);
    dispNumber(piroCnt[1]);
    glPopMatrix();

    if (gameMode==1) {
        startTime--;
        if (startTime<0) {
            gameMode = 2;
            startTime = 0;
            goTime = GOTIME;
        }
        for (int i=0; i<FRUITMAX; i++) {
            if (fruitPos[i].value==0) {
                gameMode = 2;
                goTime = GOTIME;
                //startTime = 0;
                break;
            }
        }
    }
    else if (gameMode==2) {
        goTime--;
        if (goTime<0) {
            gameMode = 0;
            for (int i=0; i<MUSHIMAX; i++) {
                resetInsect(i, 1);
            }
            resetFruit();
        }
    }
    
    //描画実行
    glutSwapBuffers();
    
    logoPhi+=0.1;
    if (logoPhi>2.0*M_PI) logoPhi = logoPhi-2.0*M_PI;
}

void dispNumber(int number)
{
    int num = number/1000;
    
    if (num>0) {
        glDisable(GL_LIGHTING);
        //glColor4d(1.0, 1.0, 1.0, 1.0);
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, 100+num);
        glPushMatrix();
        //glTranslated(-400.0, 330.0, catchArea.z-20.0);
        glScaled(40.0*28.0/48.0, 28.0, 1.0);
        glBegin(GL_QUADS);
        glTexCoord2d(0.0, 0.0);
        glVertex3d(0.5, 0.5, 0.0);
        glTexCoord2d(1.0, 0.0);
        glVertex3d(-0.5, 0.5, 0.0);
        glTexCoord2d(1.0, 1.0);
        glVertex3d(-0.5, -0.5, 0.0);
        glTexCoord2d(0.0, 1.0);
        glVertex3d(0.5, -0.5, 0.0);
        glEnd();
        glPopMatrix();
        glDisable(GL_TEXTURE_2D);
    }
    
    int num2 = (number-num*1000)/100;
    if (num2>0 || num>0) {
        glDisable(GL_LIGHTING);
        //glColor4d(1.0, 1.0, 1.0, 1.0);
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, 100+num2);
        glPushMatrix();
        glTranslated(-30.0, 0.0, 0.0);
        glScaled(40.0*28.0/48.0, 28.0, 1.0);
        glBegin(GL_QUADS);
        glTexCoord2d(0.0, 0.0);
        glVertex3d(0.5, 0.5, 0.0);
        glTexCoord2d(1.0, 0.0);
        glVertex3d(-0.5, 0.5, 0.0);
        glTexCoord2d(1.0, 1.0);
        glVertex3d(-0.5, -0.5, 0.0);
        glTexCoord2d(0.0, 1.0);
        glVertex3d(0.5, -0.5, 0.0);
        glEnd();
        glPopMatrix();
        glDisable(GL_TEXTURE_2D);
    }
    
    int num3 = (number-num*1000-num2*100)/10;
    if (num3>0 || num2>0 || num>0) {
        glDisable(GL_LIGHTING);
        //glColor4d(1.0, 1.0, 1.0, 1.0);
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, 100+num3);
        glPushMatrix();
        glTranslated(-60.0, 0.0, 0.0);
        glScaled(40.0*28.0/48.0, 28.0, 1.0);
        glBegin(GL_QUADS);
        glTexCoord2d(0.0, 0.0);
        glVertex3d(0.5, 0.5, 0.0);
        glTexCoord2d(1.0, 0.0);
        glVertex3d(-0.5, 0.5, 0.0);
        glTexCoord2d(1.0, 1.0);
        glVertex3d(-0.5, -0.5, 0.0);
        glTexCoord2d(0.0, 1.0);
        glVertex3d(0.5, -0.5, 0.0);
        glEnd();
        glPopMatrix();
        glDisable(GL_TEXTURE_2D);
    }
    
    int num4 = number-num*1000-num2*100-num3*10;
    glDisable(GL_LIGHTING);
    //glColor4d(1.0, 1.0, 1.0, 1.0);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 100+num4);
    glPushMatrix();
    glTranslated(-90.0, 0.0, 0.0);
    glScaled(40.0*28.0/48.0, 28.0, 1.0);
    glBegin(GL_QUADS);
    glTexCoord2d(0.0, 0.0);
    glVertex3d(0.5, 0.5, 0.0);
    glTexCoord2d(1.0, 0.0);
    glVertex3d(-0.5, 0.5, 0.0);
    glTexCoord2d(1.0, 1.0);
    glVertex3d(-0.5, -0.5, 0.0);
    glTexCoord2d(0.0, 1.0);
    glVertex3d(0.5, -0.5, 0.0);
    glEnd();
    glPopMatrix();
    glDisable(GL_TEXTURE_2D);
}

//リサイズコールバック理関数
void resize(int w, int h)
{
    //ビューポート設定
    glViewport(0, 0, w, h);
    
    winW = w; winH = h;
}

//マウスクリックコールバック関数
void mouse(int button, int state, int x, int y)
{
    GLfloat winX, winY, winZ;  //ウィンドウ座標
    GLdouble objX, objY, objZ;  //ワールド座標
    
    if (state==GLUT_DOWN) {
        //クリックしたマウス座標を(mX, mY)に格納
        mX = x; mY = y;
        
        //ボタン情報
        mButton = button; mState = state;
        
        //左クリックの場合，クリック点のワールド座標取得
        if (button==GLUT_LEFT_BUTTON) {
            //マウス座標をウィンドウ座標に変換
            winX = x; winY = winH-y;
            glReadPixels(winX, winY, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &winZ);  //デプスバッファ取り出し
            printf("winZ=%f\n", winZ);
            
            //モデルビュー変換行列・透視変換行列・ビューポート変換行列を取り出す
            glGetDoublev(GL_MODELVIEW_MATRIX, modelM);
            glGetDoublev(GL_PROJECTION_MATRIX, projM);
            glGetIntegerv(GL_VIEWPORT, viewM);
            //ウィンドウ座標をワールド座標に変換
            gluUnProject(winX, winY, winZ, modelM, projM, viewM, &objX, &objY, &objZ);
            
            printf("objX = %f, objY = %f, objZ = %f\n", objX, objY, objZ);
            
        }
    }
}

//マウスドラッグコールバック関数
void motion(int x, int y)
{
    if (mButton==GLUT_RIGHT_BUTTON) {
        //マウスのx方向の移動(mX-x)：水平角の変化
        eDegY = eDegY+(mX-x)*0.5;
        if (eDegY>360) eDegY-=360;
        if (eDegY<-0) eDegY+=360;
        
        //マウスのy方向の移動(y-mY)：垂直角の変化
        eDegX = eDegX+(y-mY)*0.5;
        if (eDegX>89) eDegX=89;
        if (eDegX<-89) eDegX=-89;
    }
    
    //現在のマウス座標を(mX, mY)に格納
    mX = x; mY = y;
}

//引数のベクトルを単位ベクトル化して戻す
Vec_3D vectorNormalize(Vec_3D v0)
{
    double len;  //ベクトル長
    Vec_3D v;  //戻り値用ベクトル
    
    //ベクトル長を計算
    len = sqrt(v0.x*v0.x+v0.y*v0.y+v0.z*v0.z);
    //ベクトル各成分をベクトル長で割って正規化
    if (len>0) {
        v.x = v0.x/len;
        v.y = v0.y/len;
        v.z = v0.z/len;
    }
    
    return v;  //正規化したベクトルを返す
}

//タイマーコールバック関数
void timer(int value)
{
    glutPostRedisplay();  //ディスプレイイベント強制発生
    glutTimerFunc(1000.0/fLate, timer, 0);
}

//キーボードコールバック関数
void keyboard(unsigned char key, int x, int y)
{
    switch (key) {
        case 27:
            CleanupExit();
            
        case 'f':
            glutFullScreen();
            break;
            
        case 'r':
            resetFruit();
            for (int i=0; i<MUSHIMAX; i++) {
                resetInsect(i, 3);
            }
            gameMode = 1;
            startTime = TIMEMAX;
            score[0] = 0;
            piroCnt[0] = 0;
            score[1] = 0;
            piroCnt[1] = 0;
            break;
            
        //視点関係
        case 'a':
            tg.x += 20.0;
            break;
        case 'd':
            tg.x -= 20.0;
            break;
        case 'w':
            tg.y += 20.0;
            break;
        case 'x':
            tg.y -= 20.0;
            break;
        case 's':
            tg.z += 20.0;
            break;
        case 'S':
            tg.z -= 20.0;
            break;
        case 'z':
            eDist -= 20.0;
            break;
        case 'Z':
            eDist += 20.0;
            break;
        case 'o':
            eDegY = 180.0; eDegX = 0.0; eDist = 2000.0;
            tg.x = 0.0; tg.y = 0.0; tg.z = 2000;
            break;
            
            
        default:
            break;
    }
}

//------------------------------------------------------- Kinect関係 --------------------------------------------------------
//Kinect初期化
int initKinect()
{
    XnStatus nRetVal = XN_STATUS_OK;
    
    EnumerationErrors errors;
    nRetVal = g_Context.InitFromXmlFile(SAMPLE_XML_PATH, g_scriptNode, &errors);
    if (nRetVal == XN_STATUS_NO_NODE_PRESENT)
    {
        XnChar strError[1024];
        errors.ToString(strError, 1024);
        printf("%s\n", strError);
        return (nRetVal);
    }
    else if (nRetVal != XN_STATUS_OK)
    {
        printf("Open failed: %s\n", xnGetStatusString(nRetVal));
        return (nRetVal);
    }
    
    nRetVal = g_Context.FindExistingNode(XN_NODE_TYPE_DEPTH, g_DepthGenerator);
    if (nRetVal != XN_STATUS_OK)
    {
        printf("No depth generator found. Using a default one...");
        MockDepthGenerator mockDepth;
        nRetVal = mockDepth.Create(g_Context);
        CHECK_RC(nRetVal, "Create mock depth");
        
        // set some defaults
        XnMapOutputMode defaultMode;
        defaultMode.nXRes = 320;
        defaultMode.nYRes = 240;
        defaultMode.nFPS = 30;
        nRetVal = mockDepth.SetMapOutputMode(defaultMode);
        CHECK_RC(nRetVal, "set default mode");
        
        // set FOV
        XnFieldOfView fov;
        fov.fHFOV = 1.0225999419141749;
        fov.fVFOV = 0.79661567681716894;
        nRetVal = mockDepth.SetGeneralProperty(XN_PROP_FIELD_OF_VIEW, sizeof(fov), &fov);
        CHECK_RC(nRetVal, "set FOV");
        
        XnUInt32 nDataSize = defaultMode.nXRes * defaultMode.nYRes * sizeof(XnDepthPixel);
        XnDepthPixel* pData = (XnDepthPixel*)xnOSCallocAligned(nDataSize, 1, XN_DEFAULT_MEM_ALIGN);
        
        nRetVal = mockDepth.SetData(1, 0, nDataSize, pData);
        CHECK_RC(nRetVal, "set empty depth map");
        
        g_DepthGenerator = mockDepth;
    }
    
    nRetVal = g_Context.FindExistingNode(XN_NODE_TYPE_USER, g_UserGenerator);
    if (nRetVal != XN_STATUS_OK)
    {
        nRetVal = g_UserGenerator.Create(g_Context);
        CHECK_RC(nRetVal, "Find user generator");
    }
    
    nRetVal = g_Context.FindExistingNode(XN_NODE_TYPE_IMAGE, g_ImageGenerator);
    if (nRetVal != XN_STATUS_OK)
    {
        printf("No image node exists! Check your XML.");
        return 1;
    }
    
    XnCallbackHandle hUserCallbacks, hCalibrationStart, hCalibrationComplete, hPoseDetected, hCalibrationInProgress, hPoseInProgress;
    if (!g_UserGenerator.IsCapabilitySupported(XN_CAPABILITY_SKELETON))
    {
        printf("Supplied user generator doesn't support skeleton\n");
        return 1;
    }
    nRetVal = g_UserGenerator.RegisterUserCallbacks(User_NewUser, User_LostUser, NULL, hUserCallbacks);
    CHECK_RC(nRetVal, "Register to user callbacks");
    nRetVal = g_UserGenerator.GetSkeletonCap().RegisterToCalibrationStart(UserCalibration_CalibrationStart, NULL, hCalibrationStart);
    CHECK_RC(nRetVal, "Register to calibration start");
    nRetVal = g_UserGenerator.GetSkeletonCap().RegisterToCalibrationComplete(UserCalibration_CalibrationComplete, NULL, hCalibrationComplete);
    CHECK_RC(nRetVal, "Register to calibration complete");
    
    if (g_UserGenerator.GetSkeletonCap().NeedPoseForCalibration())
    {
        g_bNeedPose = TRUE;
        if (!g_UserGenerator.IsCapabilitySupported(XN_CAPABILITY_POSE_DETECTION))
        {
            printf("Pose required, but not supported\n");
            return 1;
        }
        nRetVal = g_UserGenerator.GetPoseDetectionCap().RegisterToPoseDetected(UserPose_PoseDetected, NULL, hPoseDetected);
        CHECK_RC(nRetVal, "Register to Pose Detected");
        g_UserGenerator.GetSkeletonCap().GetCalibrationPose(g_strPose);
    }
    
    g_UserGenerator.GetSkeletonCap().SetSkeletonProfile(XN_SKEL_PROFILE_ALL);
    
    //nRetVal = g_UserGenerator.GetSkeletonCap().RegisterToCalibrationInProgress(MyCalibrationInProgress, NULL, hCalibrationInProgress);
    //CHECK_RC(nRetVal, "Register to calibration in progress");
    
    //nRetVal = g_UserGenerator.GetPoseDetectionCap().RegisterToPoseInProgress(MyPoseInProgress, NULL, hPoseInProgress);
    //CHECK_RC(nRetVal, "Register to pose in progress");
    
    nRetVal = g_Context.StartGeneratingAll();
    CHECK_RC(nRetVal, "StartGenerating");
    
    //テクスチャ設定
    g_DepthGenerator.GetAlternativeViewPointCap().SetViewPoint(g_ImageGenerator);
    g_nTexMapX = (((unsigned short)(depthMD.FullXRes()-1) / 512) + 1) * 512;
    g_nTexMapY = (((unsigned short)(depthMD.FullYRes()-1) / 512) + 1) * 512;
    g_pTexMap = (XnRGB24Pixel*)malloc(g_nTexMapX * g_nTexMapY * sizeof(XnRGB24Pixel));
    
    return 0;
}

//関節点"jointPos[]"取得
int getJoint()
{
    static bool bInitialized = false;
    static GLuint depthTexID;
    static unsigned char* pDepthTexBuf;
    static int texWidth, texHeight;
    
    float topLeftX;
    float topLeftY;
    float bottomRightY;
    float bottomRightX;
    float texXpos;
    float texYpos;
    
    char strLabel[50] = "";
    XnUserID aUsers[15];
    XnUInt16 nUsers = 15;
    g_UserGenerator.GetUsers(aUsers, nUsers);
    
    XnSkeletonJointPosition joint[24];
    
    //if (nUsers>0) {
    for (int i=0; i<nUsers; i++) {
        /*
         XN_SKEL_HEAD
         XN_SKEL_NECK
         XN_SKEL_TORSO
         XN_SKEL_WAIST
         XN_SKEL_LEFT_COLLAR
         XN_SKEL_LEFT_SHOULDER
         XN_SKEL_LEFT_ELBOW
         XN_SKEL_LEFT_WRIST
         XN_SKEL_LEFT_HAND
         XN_SKEL_LEFT_FINGERTIP
         XN_SKEL_RIGHT_COLLAR
         XN_SKEL_RIGHT_SHOULDER
         XN_SKEL_RIGHT_ELBOW
         XN_SKEL_RIGHT_WRIST
         XN_SKEL_RIGHT_HAND
         XN_SKEL_RIGHT_FINGERTIP
         XN_SKEL_LEFT_HIP
         XN_SKEL_LEFT_KNEE
         XN_SKEL_LEFT_ANKLE
         XN_SKEL_LEFT_FOOT
         XN_SKEL_RIGHT_HIP
         XN_SKEL_RIGHT_KNEE
         XN_SKEL_RIGHT_ANKLE
         XN_SKEL_RIGHT_FOOT
         */
        
        g_UserGenerator.GetSkeletonCap().GetSkeletonJointPosition(aUsers[i], XN_SKEL_HEAD, joint[0]);
        g_UserGenerator.GetSkeletonCap().GetSkeletonJointPosition(aUsers[i], XN_SKEL_NECK, joint[1]);
        g_UserGenerator.GetSkeletonCap().GetSkeletonJointPosition(aUsers[i], XN_SKEL_TORSO, joint[2]);
        //g_UserGenerator.GetSkeletonCap().GetSkeletonJointPosition(aUsers[0], XN_SKEL_WAIST, joint[2]);  //たぶん取れない
        //g_UserGenerator.GetSkeletonCap().GetSkeletonJointPosition(aUsers[0], XN_SKEL_LEFT_COLLAR, joint[3]);   //たぶん取れない
        g_UserGenerator.GetSkeletonCap().GetSkeletonJointPosition(aUsers[i], XN_SKEL_LEFT_SHOULDER, joint[3]);
        g_UserGenerator.GetSkeletonCap().GetSkeletonJointPosition(aUsers[i], XN_SKEL_LEFT_ELBOW, joint[4]);
        //g_UserGenerator.GetSkeletonCap().GetSkeletonJointPosition(aUsers[0], XN_SKEL_LEFT_WRIST, joint[5]);
        g_UserGenerator.GetSkeletonCap().GetSkeletonJointPosition(aUsers[i], XN_SKEL_LEFT_HAND, joint[5]);
        //g_UserGenerator.GetSkeletonCap().GetSkeletonJointPosition(aUsers[0], XN_SKEL_LEFT_FINGERTIP, joint[6]);
        //g_UserGenerator.GetSkeletonCap().GetSkeletonJointPosition(aUsers[0], XN_SKEL_RIGHT_COLLAR, joint[6]);
        g_UserGenerator.GetSkeletonCap().GetSkeletonJointPosition(aUsers[i], XN_SKEL_RIGHT_SHOULDER, joint[6]);
        g_UserGenerator.GetSkeletonCap().GetSkeletonJointPosition(aUsers[i], XN_SKEL_RIGHT_ELBOW, joint[7]);
        //g_UserGenerator.GetSkeletonCap().GetSkeletonJointPosition(aUsers[0], XN_SKEL_RIGHT_WRIST, joint[11]);
        g_UserGenerator.GetSkeletonCap().GetSkeletonJointPosition(aUsers[i], XN_SKEL_RIGHT_HAND, joint[8]);
        //g_UserGenerator.GetSkeletonCap().GetSkeletonJointPosition(aUsers[0], XN_SKEL_RIGHT_FINGERTIP, joint[10]);
        g_UserGenerator.GetSkeletonCap().GetSkeletonJointPosition(aUsers[i], XN_SKEL_LEFT_HIP, joint[9]);
        g_UserGenerator.GetSkeletonCap().GetSkeletonJointPosition(aUsers[i], XN_SKEL_LEFT_KNEE, joint[10]);
        //g_UserGenerator.GetSkeletonCap().GetSkeletonJointPosition(aUsers[0], XN_SKEL_LEFT_ANKLE, joint[11]);
        g_UserGenerator.GetSkeletonCap().GetSkeletonJointPosition(aUsers[i], XN_SKEL_LEFT_FOOT, joint[11]);
        g_UserGenerator.GetSkeletonCap().GetSkeletonJointPosition(aUsers[i], XN_SKEL_RIGHT_HIP, joint[12]);
        g_UserGenerator.GetSkeletonCap().GetSkeletonJointPosition(aUsers[i], XN_SKEL_RIGHT_KNEE, joint[13]);
        //g_UserGenerator.GetSkeletonCap().GetSkeletonJointPosition(aUsers[0], XN_SKEL_RIGHT_ANKLE, joint[15]);
        g_UserGenerator.GetSkeletonCap().GetSkeletonJointPosition(aUsers[i], XN_SKEL_RIGHT_FOOT, joint[14]);
        
        //信頼度が低い場合は採用しない
        //		for (int j=0; j<jointNum; j++) {
        //			if (joint[j].fConfidence<0.5)
        //				continue;
        //		}
        
        for (int j=0; j<jointNum; j++) {
            jointPos[i][j] = joint[j].position;
            jointConf[i][j] = joint[j].fConfidence;
        }
    }
    
    return nUsers;
}

//スキャン点"pointPos[]"と色"pointCol[]"取得
void getDepthImage()
{
    int cnt = 0;
    xnOSMemSet(g_pTexMap, 0, g_nTexMapX*g_nTexMapY*sizeof(XnRGB24Pixel));
    XnRGB24Pixel* pTexRow = g_pTexMap+imageMD.YOffset()*g_nTexMapX;
    const XnRGB24Pixel* pImageRow = imageMD.RGB24Data();
    const XnDepthPixel* pDepthRow = depthMD.Data();
    const XnLabel* pLabelRow = sceneMD.Data();
    
    for (XnUInt y=0; y<imageMD.YRes(); ++y)
    {
        const XnRGB24Pixel* pImage = pImageRow;
        const XnDepthPixel* pDepth = pDepthRow;
        const XnLabel* pLabel = pLabelRow;
        
        for (XnUInt x=0; x<imageMD.XRes(); ++x, ++pImage, ++pDepth, ++pLabel)
        {
            pointCol[cnt] = *pImage;
            pointLabel[cnt] = *pLabel;
            t1[cnt].X = x; t1[cnt].Y = y; t1[cnt].Z = *pDepth;
            cnt++;
        }
        pDepthRow += depthMD.XRes();
        pImageRow += imageMD.XRes();
        pLabelRow += sceneMD.XRes();
        pTexRow += g_nTexMapX;
    }
    g_DepthGenerator.ConvertProjectiveToRealWorld(GL_WIN_TOTAL, t1, pointPos);
}

//終了処理
void CleanupExit()
{
    g_scriptNode.Release();
    g_DepthGenerator.Release();
    g_UserGenerator.Release();
    g_Player.Release();
    g_Context.Release();
    
    exit (1);
}

/*
 void XN_CALLBACK_TYPE MyCalibrationInProgress(SkeletonCapability& capability, XnUserID id, XnCalibrationStatus calibrationError, void* pCookie)
 {
	m_Errors[id].first = calibrationError;
 }
 
 void XN_CALLBACK_TYPE MyPoseInProgress(PoseDetectionCapability& capability, const XnChar* strPose, XnUserID id, XnPoseDetectionStatus poseError, void* pCookie)
 {
	m_Errors[id].second = poseError;
 }
 */

// Callback: New user was detected
void XN_CALLBACK_TYPE User_NewUser(UserGenerator& generator, XnUserID nId, void* pCookie)
{
    XnUInt32 epochTime = 0;
    xnOSGetEpochTime(&epochTime);
    printf("%d New User %d\n", epochTime, nId);
    // New user found
    if (g_bNeedPose)
    {
        g_UserGenerator.GetPoseDetectionCap().StartPoseDetection(g_strPose, nId);
    }
    else
    {
        g_UserGenerator.GetSkeletonCap().RequestCalibration(nId, TRUE);
    }
}

// Callback: An existing user was lost
void XN_CALLBACK_TYPE User_LostUser(UserGenerator& generator, XnUserID nId, void* pCookie)
{
    XnUInt32 epochTime = 0;
    xnOSGetEpochTime(&epochTime);
    printf("%d Lost user %d\n", epochTime, nId);
    
    for (int j=0; j<15; j++) {
        jointPos[nId][j].X = 0.0;
        jointPos[nId][j].Y = 0.0;
        jointPos[nId][j].Z = 0.0;
    }
}

// Callback: Detected a pose
void XN_CALLBACK_TYPE UserPose_PoseDetected(PoseDetectionCapability& capability, const XnChar* strPose, XnUserID nId, void* pCookie)
{
    XnUInt32 epochTime = 0;
    xnOSGetEpochTime(&epochTime);
    printf("%d Pose %s detected for user %d\n", epochTime, strPose, nId);
    g_UserGenerator.GetPoseDetectionCap().StopPoseDetection(nId);
    g_UserGenerator.GetSkeletonCap().RequestCalibration(nId, TRUE);
}

// Callback: Started calibration
void XN_CALLBACK_TYPE UserCalibration_CalibrationStart(SkeletonCapability& capability, XnUserID nId, void* pCookie)
{
    XnUInt32 epochTime = 0;
    xnOSGetEpochTime(&epochTime);
    printf("%d Calibration started for user %d\n", epochTime, nId);
}

// Callback: Finished calibration
void XN_CALLBACK_TYPE UserCalibration_CalibrationComplete(SkeletonCapability& capability, XnUserID nId, XnCalibrationStatus eStatus, void* pCookie)
{
    XnUInt32 epochTime = 0;
    xnOSGetEpochTime(&epochTime);
    if (eStatus == XN_CALIBRATION_STATUS_OK)
    {
        // Calibration succeeded
        printf("%d Calibration complete, start tracking user %d\n", epochTime, nId);
        g_UserGenerator.GetSkeletonCap().StartTracking(nId);
    }
    else
    {
        // Calibration failed
        printf("%d Calibration failed for user %d\n", epochTime, nId);
        if(eStatus==XN_CALIBRATION_STATUS_MANUAL_ABORT)
        {
            printf("Manual abort occured, stop attempting to calibrate!");
            return;
        }
        if (g_bNeedPose)
        {
            g_UserGenerator.GetPoseDetectionCap().StartPoseDetection(g_strPose, nId);
        }
        else
        {
            g_UserGenerator.GetSkeletonCap().RequestCalibration(nId, TRUE);
        }
    }
}
//------------------------------------------------------- Kinect関係(終わり) --------------------------------------------------------
