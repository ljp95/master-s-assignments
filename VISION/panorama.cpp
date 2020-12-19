// Imagine++ project
// Project:  Panorama
// Author:   Pascal Monasse
// Date:     2013/10/08

#include <Imagine/Graphics.h>
#include <Imagine/Images.h>
#include <Imagine/LinAlg.h>
#include <vector>
#include <sstream>
using namespace Imagine;
using namespace std;

// Record clicks in two images, until right button click
void getClicks(Window w1, Window w2,
               vector<IntPoint2>& pts1, vector<IntPoint2>& pts2) {


    // ------------- TODO/A completer ----------
    int button = 0; // 1 for left, 2 for middle, 3 for right
    Event ev;
    while(button != 3){
         // get event
         getEvent(-1,ev);
         if(ev.type == EVT_BUT_ON){
             button = ev.button;
             // adding point according to window
             if(ev.win == w1){
                if(button != 3){
                    pts1.push_back(ev.pix);
                }
             }
             else{
                 if(ev.win == w2){
                     if(button != 3){
                        pts2.push_back(ev.pix);
                     }
                 }
             }
         }
    }
    // ------------- END TODO -----------------
}

// Return homography compatible with point matches
Matrix<float> getHomography(const vector<IntPoint2>& pts1,
                            const vector<IntPoint2>& pts2) {
    size_t n = min(pts1.size(), pts2.size());
    if(n<4) {
        cout << "Not enough correspondences: " << n << endl;
        return Matrix<float>::Identity(3);
    }
    Matrix<double> A(2*n,8);
    Vector<double> B(2*n);


    // ------------- TODO/A completer ----------
    // Complete matrix A according to each pair of matches points
    for(unsigned int j=0; j<n; j++){
        int i = 2*j;
        double x1 = pts1[j].x(); double x2 = pts2[j].x();
        double y1 = pts1[j].y(); double y2 = pts2[j].y();
        // matrix A
        A(i,0) = x1;  A(i,1) = y1;  A(i,2) = 1;   A(i,3) = 0;    A(i,4) = 0;    A(i,5) = 0;   A(i,6) = -x2*x1;   A(i,7) = -x2*y1;
        A(i+1,0) = 0; A(i+1,1) = 0; A(i+1,2) = 0; A(i+1,3) = x1; A(i+1,4) = y1; A(i+1,5) = 1; A(i+1,6) = -y2*x1; A(i+1,7) = -y2*y1;
        // vector B
        B[i] = x2; B[i+1] = y2;
    }
    // ------------- END TODO -----------------

    B = linSolve(A, B);
    Matrix<float> H(3, 3);
    H(0,0)=B[0]; H(0,1)=B[1]; H(0,2)=B[2];
    H(1,0)=B[3]; H(1,1)=B[4]; H(1,2)=B[5];
    H(2,0)=B[6]; H(2,1)=B[7]; H(2,2)=1;

    // Sanity check
    for(size_t i=0; i<n; i++) {
        float v1[]={(float)pts1[i].x(), (float)pts1[i].y(), 1.0f};
        float v2[]={(float)pts2[i].x(), (float)pts2[i].y(), 1.0f};
        Vector<float> x1(v1,3);
        Vector<float> x2(v2,3);
        x1 = H*x1;
        cout << x1[1]*x2[2]-x1[2]*x2[1] << ' '
             << x1[2]*x2[0]-x1[0]*x2[2] << ' '
             << x1[0]*x2[1]-x1[1]*x2[0] << endl;
    }
    return H;
}

// Grow rectangle of corners (x0,y0) and (x1,y1) to include (x,y)
void growTo(float& x0, float& y0, float& x1, float& y1, float x, float y) {
    if(x<x0) x0=x;
    if(x>x1) x1=x;
    if(y<y0) y0=y;
    if(y>y1) y1=y;    
}

// Panorama construction
void panorama(const Image<Color,2>& I1, const Image<Color,2>& I2,
              Matrix<float> H) {
    Vector<float> v(3);
    float x0=0, y0=0, x1=I2.width(), y1=I2.height();

    v[0]=0; v[1]=0; v[2]=1;
    v=H*v; v/=v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    v[0]=I1.width(); v[1]=0; v[2]=1;
    v=H*v; v/=v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    v[0]=I1.width(); v[1]=I1.height(); v[2]=1;
    v=H*v; v/=v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    v[0]=0; v[1]=I1.height(); v[2]=1;
    v=H*v; v/=v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    cout << "x0 x1 y0 y1=" << x0 << ' ' << x1 << ' ' << y0 << ' ' << y1<<endl;

    Image<Color> I(int(x1-x0), int(y1-y0));
    setActiveWindow( openWindow(I.width(), I.height()) );
    I.fill(WHITE);

    // ------------- TODO/A completer ----------
    // Calculate the shifts
    int shift_x = int(abs(x0));
    int shift_y = int(abs(y0));
    // Copy paste I2
    for(int i=0; i<I2.width(); i++){
        for(int j=0; j<I2.height(); j++){
            I(i+shift_x, j+shift_y) = I2(i,j);
        }
    }

    // Stitching I1 by pulling
    // Calculate inverse matrix : Matrix to FMaxtrix to Matrix
    FMatrix<float,3,3> H_inv2;
    for(int i=0; i<3; i++){
        for(int j=0; j<3; j++){
            H_inv2(i,j) = H(i,j);
        }
    }
    H_inv2 = inverseFMatrix(H_inv2);
    Matrix<float> H_inv(3,3);
    for(int i=0; i<3; i++){
        for(int j=0; j<3; j++){
            H_inv(i,j) = H_inv2(i,j);
        }
    }
    // Pull
    for(int i=int(x0); i<I1.width(); i++){
        for(int j=int(y0); j<I1.height(); j++){
            // Apply inverse of H on each pixel of the final image
            v[0]=i; v[1]=j; v[2]=1;
            v = H_inv*v; v/=v[2];
            // check if the point is inside I1 or not
            if(int(v[0])>=0 && int(v[1])>=0 && int(v[0]<I1.width() && v[1]<I1.height())){
                int x = i+shift_x;
                int y = j+shift_y;
                I(x,y) = I1(int(v[0]),int(v[1]));   // paste I1 pixel
                /*
                // To get average
                if(I(x,y)[0] == 255 && I(x,y)[1] == 255 && I(x,y)[2] == 255){ // if white pixel
                    I(x,y) = I1(int(v[0]),int(v[1]));   // paste I1 pixel
                }
                else{
                    I(x,y) = (I1(int(v[0]),int(v[1]))+I(x,y))/2; // else average
                }
                */
            }
        }
    }
    /*
    // Stitching by pushing
    for(int i=0; i<I1.width(); i++){
        for(int j=0; j<I1.height(); j++){
            v[0]=i;v[1]=j;v[2]=1;
            v=H*v; v/=v[2];
            int x = int(v[0])+shift_x;
            int y = int(v[1])+shift_y;
            I(x,y) = I1(i,j);
            // To get average
            if(I(x,y)[0] == 255 && I(x,y)[1] == 255 && I(x,y)[2] == 255){ // if white pixel
                I(x,y) = I1(i,j); // paste I1
            }
            else{
                I(x,y) = (I1(i,j)+I(x,y))/2; // average
            }

        }
    }
    */
    // ------------- END TODO -----------------

    display(I,0,0);
}

// Main function
int main(int argc, char* argv[]) {
    const char* s1 = argc>1? argv[1]: srcPath("image0006.jpg");
    const char* s2 = argc>2? argv[2]: srcPath("image0007.jpg");

    // Load and display images
    Image<Color> I1, I2;
    if( ! load(I1, s1) ||
        ! load(I2, s2) ) {
        cerr<< "Unable to load the images" << endl;
        return 1;
    }
    Window w1 = openWindow(I1.width(), I1.height(), s1);
    display(I1,0,0);
    Window w2 = openWindow(I2.width(), I2.height(), s2);
    setActiveWindow(w2);
    display(I2,0,0);

    // Get user's clicks in images
    vector<IntPoint2> pts1, pts2;
    getClicks(w1, w2, pts1, pts2);

    vector<IntPoint2>::const_iterator it;
    cout << "pts1="<<endl;
    for(it=pts1.begin(); it != pts1.end(); it++)
        cout << *it << endl;
    cout << "pts2="<<endl;
    for(it=pts2.begin(); it != pts2.end(); it++)
        cout << *it << endl;

    // Compute homography
    Matrix<float> H = getHomography(pts1, pts2);
    cout << "H=" << H/H(2,2);

    // Apply homography
    panorama(I1, I2, H);

    endGraphics();
    return 0;
}
