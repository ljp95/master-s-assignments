// Imagine++ project
// Project:  Fundamental
// Author:   Pascal Monasse
// Date:     2013/10/08

#include "./Imagine/Features.h"
#include <Imagine/Graphics.h>
#include <Imagine/LinAlg.h>
#include <vector>
#include <cstdlib>
#include <ctime>
using namespace Imagine;
using namespace std;

static const float BETA = 0.01f; // Probability of failure

struct Match {
    float x1, y1, x2, y2;
};

// Display SIFT points and fill vector of point correspondences
void algoSIFT(Image<Color,2> I1, Image<Color,2> I2,
              vector<Match>& matches) {
    // Find interest points
    SIFTDetector D;
    D.setFirstOctave(-1);
    Array<SIFTDetector::Feature> feats1 = D.run(I1);
    drawFeatures(feats1, Coords<2>(0,0));
    cout << "Im1: " << feats1.size() << flush;
    Array<SIFTDetector::Feature> feats2 = D.run(I2);
    drawFeatures(feats2, Coords<2>(I1.width(),0));
    cout << " Im2: " << feats2.size() << flush;

    const double MAX_DISTANCE = 100.0*100.0;
    for(size_t i=0; i < feats1.size(); i++) {
        SIFTDetector::Feature f1=feats1[i];
        for(size_t j=0; j < feats2.size(); j++) {
            double d = squaredDist(f1.desc, feats2[j].desc);
            if(d < MAX_DISTANCE) {
                Match m;
                m.x1 = f1.pos.x();
                m.y1 = f1.pos.y();
                m.x2 = feats2[j].pos.x();
                m.y2 = feats2[j].pos.y();
                matches.push_back(m);
            }
        }
    }
}

// RANSAC algorithm to compute F from point matches (8-point algorithm)
// Parameter matches is filtered to keep only inliers as output.
FMatrix<float,3,3> computeF(vector<Match>& matches) {
    const float distMax = 1.5f; // Pixel error for inlier/outlier discrimination
    int Niter=100000; // Adjusted dynamically
    FMatrix<float,3,3> bestF;
    vector<int> bestInliers;

    // --------------- TODO ------------
    // DO NOT FORGET NORMALIZATION OF POINTS
    // initialization
    vector<int> indices,inliers;
    Match p;
    int iter,n,m;
    FVector<float,9> S;
    FMatrix<float,9,9> A,U,V_t;
    FVector<float,3> Sf,Sf2;
    FMatrix<float,3,3> F,N,Uf,Uf2,Vf_t,Vf_t2;
    FMatrix<float,3,1> u;
    FMatrix<float,1,3> v;
    FMatrix <float,3,1> Ftu;
    float dist;

    // filling what we can
    // matrix for normalization
    N.fill(0); N(0,0) = 0.001; N(1,1) = 0.001; N(2,2) = 1;
    // we will shuffle indices to get random matches
    for(int i=0;i<matches.size();i++){
        indices.push_back(i);
    }
    n = matches.size();
    iter = 0;

    while(iter<Niter){
        // shuffle indices to select matches
        std::random_shuffle(indices.begin(),indices.end());

        // fill matrix A with normalized points
        A.fill(0);
        for(int i=0;i<8;i++){
           p = matches[indices[i]];
           p.x1 *= 0.001; p.y1 *= 0.001; p.x2 *= 0.001; p.y2 *= 0.001;
           A(i,0) = p.x1*p.x2; A(i,1) = p.y1*p.x2; A(i,2) = p.x2;
           A(i,3) = p.x1*p.y2; A(i,4) = p.y1*p.y2; A(i,5) = p.y2;
           A(i,6) = p.x1;      A(i,7) = p.y1;      A(i,8) = 1;
        }

        // SVD
        svd(A,U,S,V_t);

        // fill F with last column of V = last row of Vt
        for(int i=0;i<3;i++){
            for(int j=0;j<3;j++){
                F(i,j) = V_t(8,i*3+j);
            }
        }

        // transform F to be singular
        svd(F,Uf,Sf,Vf_t);
        Sf[2] = 0;
        F = Uf*Diagonal(Sf)*Vf_t;

        // un-normalized F
        F = transpose(N)*F*N;

        // testing the model F
        // first get inliers by computing distances d(v,Ft*u), pushback if < threshold
        inliers.clear();
        u[2] = 1.; v[2] = 1.;
        for(int i=0;i<n;i++){
            u[0] = matches[i].x1; u[1] = matches[i].y1;
            v[0] = matches[i].x2; v[1] = matches[i].y2;
            Ftu = transpose(F)*u;
            dist = (abs(Ftu[0]*v[0]+Ftu[1]*v[1]+Ftu[2]))/(sqrt(Ftu[0]*Ftu[0]+Ftu[1]*Ftu[1]));
            //cout <<dist<<endl;
            if(dist < distMax){
                inliers.push_back(i);
            }
        }

        // Always update for case of first F computed
        if(iter==0){
            m = inliers.size();
        }
        // Update m,bestF,inliers and Niter if more inliers found
        if(inliers.size()>=m){
            m = inliers.size();
            bestF = F;
            bestInliers = inliers;
            cout << "Update ! " << m<<" inliers at iteration "<<iter<<endl;
            if(float(m)/n > 0.1){ // Care of division by something too close to 0
                Niter = int(log(BETA)/log(1-pow(float(m)/n,8)));
            }
        }
        iter++;
    }
    cout <<"Fin de RANSAC en "<<iter<<" iterations"<<endl;

    // refine bestF by least square minimization on all inliers
    // Same as before !
    // initialization
    Matrix <float> B(m,9);
    Vector<float> S2(9);
    Matrix<float> V_t2(9,9);
    Matrix<float> U2(m,m);

    // fill matrix B with normalized points
    for(int i=0;i<m;i++){
        p = matches[indices[i]];
        p.x1 *= 0.001; p.y1 *= 0.001; p.x2 *= 0.001; p.y2 *= 0.001;
        B(i,0) = p.x1*p.x2; B(i,1) = p.x1*p.y2; B(i,2) = p.x1;
        B(i,3) = p.y1*p.x2; B(i,4) = p.y1*p.y2; B(i,5) = p.y1;
        B(i,6) = p.x2;      B(i,7) = p.y2;      B(i,8) = 1;
    }

    // SVD
    svd(B,U2,S2,V_t2);

    // fill bestF from last row of Vt
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            bestF(i,j) = V_t(8,i*3+j);
        }
    }

    // Transform bestF to be singular and un-normalized it
    svd(bestF,Uf2,Sf2,Vf_t2);
    Sf2[2] = 0;
    bestF = Uf2*Diagonal(Sf2)*Vf_t2;
    bestF = transpose(N)*bestF*N;

    // ------------------- FIN TODO -----------------
    // Updating matches with inliers only
    vector<Match> all=matches;
    matches.clear();
    for(size_t i=0; i<bestInliers.size(); i++)
        matches.push_back(all[bestInliers[i]]);
    return bestF;
}

// Expects clicks in one image and show corresponding line in other image.
// Stop at right-click.
void displayEpipolar(Image<Color> I1, Image<Color> I2,
                     const FMatrix<float,3,3>& F) {
    // --------------- TODO ------------
    // initialization
    FMatrix <float,3,1> u,v,Fu;
    float h = I1.height();
    float w = I1.width();
    float a,b,c;
    int x,y;
    bool draw_right = 0; // to know where to draw and to shift x if needed

    while(true) {
        Color color(rand()%256,rand()%256,rand()%256);
        if(getMouse(x,y) == 3) // quit if right click with the mouse
            break;
        drawCircle(x,y,1,color,2); // draw a circle to see the point clicked
        u[0] = x; u[1] = y; u[2] = 1;

        // compute epipolar line depending on click
        if(x>w){ // Click on the right image, draw on left
            u[0] -= w; // shift to get real x coordinate
            Fu = F*u;
            draw_right = 0;
        }
        else{ // Click on the left image, draw on right
            Fu = transpose(F)*u;
            draw_right = 1;
        }

        // epipolar coordinates
        a = Fu[0]; b = Fu[1]; c = Fu[2];

        // Solve ax+by+zc = 0 to get intersection between image borders and epipolar line
        // Order : left,right,up and down
        //       :  x=0, x=w ,y=0 and y=h
        vector <float> intersections_x = {0,w,-c/a, (-c-b*h)/a};
        vector <float> intersections_y = {-c/b,(-c-a*w)/b,0,h};

        // Keeping the points in the image
        vector <float> X;
        vector <float> Y;
        for(int i=0;i<4;i++){
            x = intersections_x[i];
            y = intersections_y[i];
            if(x>=0 && x<=w && y>=0 && y<=h){ // condition to be in
                X.push_back(x+w*draw_right);
                Y.push_back(y);
            }
        }

        //drawing the line
        drawLine(X[0],Y[0],X[1],Y[1],color);
    }
}

int main(int argc, char* argv[])
{
    srand((unsigned int)time(0));

    const char* s1 = argc>1? argv[1]: srcPath("im1.jpg");
    const char* s2 = argc>2? argv[2]: srcPath("im2.jpg");

    // Load and display images
    Image<Color,2> I1, I2;
    if( ! load(I1, s1) ||
        ! load(I2, s2) ) {
        cerr<< "Unable to load images" << endl;
        return 1;
    }
    int w = I1.width();
    openWindow(2*w, I1.height());
    display(I1,0,0);
    display(I2,w,0);

    vector<Match> matches;
    algoSIFT(I1, I2, matches);
    cout << " matches: " << matches.size() << endl;
    click();
    
    FMatrix<float,3,3> F = computeF(matches);
    cout << "F="<< endl << F;

    // Redisplay with matches
    display(I1,0,0);
    display(I2,w,0);
    for(size_t i=0; i<matches.size(); i++) {
        Color c(rand()%256,rand()%256,rand()%256);
        fillCircle(matches[i].x1+0, matches[i].y1, 2, c);
        fillCircle(matches[i].x2+w, matches[i].y2, 2, c);        
    }
    click();

    // Redisplay without SIFT points
    display(I1,0,0);
    display(I2,w,0);
    displayEpipolar(I1, I2, F);

    endGraphics();
    return 0;
}

