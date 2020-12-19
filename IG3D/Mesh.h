#ifndef MESH_H
#define MESH_H


#include <vector>
#include <string>
#include <math.h>
#include "Vec3.h"

#include <GL/glut.h>


// -------------------------------------------
// Basic Mesh class
// -------------------------------------------

struct MeshVertex {
    inline MeshVertex () {}
    inline MeshVertex (const Vec3 & _p, const Vec3 & _n) : p (_p), n (_n) {}
    inline MeshVertex (const MeshVertex & v) : p (v.p), n (v.n) {}
    inline virtual ~MeshVertex () {}
    inline MeshVertex & operator = (const MeshVertex & v) {
        p = v.p;
        n = v.n;
        return (*this);
    }
    // membres :
    Vec3 p; // une position
    Vec3 n; // une normale
};

struct MeshTriangle {
    inline MeshTriangle () {
        corners[0] = corners[1] = corners[2] = 0;
    }
    inline MeshTriangle (const MeshTriangle & t) {
        corners[0] = t[0];   corners[1] = t[1];   corners[2] = t[2];
    }
    inline MeshTriangle (unsigned int v0, unsigned int v1, unsigned int v2) {
        corners[0] = v0;   corners[1] = v1;   corners[2] = v2;
    }
    inline virtual ~MeshTriangle () {}
    inline MeshTriangle & operator = (const MeshTriangle & t) {
        corners[0] = t[0];   corners[1] = t[1];   corners[2] = t[2];
        return (*this);
    }

    unsigned int operator [] (unsigned int c) const { return corners[c]; }
    unsigned int & operator [] (unsigned int c) { return corners[c]; }

private:
    // membres :
    unsigned int corners[3];
};




class Mesh {
public:
    std::vector<MeshVertex>   vertices;
    std::vector<MeshTriangle> triangles;

    std::vector<float> positionArray;
    std::vector<unsigned int> triangleArray;
	std::vector<float> normalArray;
	std::vector<float> colorArray;

    void loadOFF (const std::string & filename);
    void recomputeNormals ();
    void centerAndScaleToUnit ();
    void scaleUnit ();
    void buildVertexArray(){
        positionArray.clear();
        triangleArray.clear();
        normalArray.clear();
		unsigned int i,j,k;
		for(i=0;i<vertices.size();i++){
			positionArray.push_back(vertices[i].p[0]);
			positionArray.push_back(vertices[i].p[1]);
			positionArray.push_back(vertices[i].p[2]);
		}
		for(j=0;j<triangles.size();j++){
			triangleArray.push_back(triangles[j][0]);	
			triangleArray.push_back(triangles[j][1]);
			triangleArray.push_back(triangles[j][2]);
		}
		for(k=0;k<vertices.size();k++){
			normalArray.push_back(vertices[k].n[0]);
			normalArray.push_back(vertices[k].n[1]);
			normalArray.push_back(vertices[k].n[2]);
		}
    }
	void buildColorArray(){
        colorArray.clear();
		unsigned int i;
		for(i=0;i<positionArray.size();i++){
			colorArray.push_back(double(rand())/(RAND_MAX));
		}
	}
	void setUnitSphere(int nX, int nY){
		vertices.clear();
		triangles.clear();
		int i,j;
		for(i=0;i<nX;i++){
			for(j=0;j<nY;j++){
				MeshVertex newVertex;
				float theta = (float(i)*2*M_PI)/float(nX-1);
				float phi = -M_PI/2 + float(j)*M_PI/float(nY-1);
				newVertex.p[0] = cos(theta)*cos(phi);	
				newVertex.p[1] = sin(theta)*cos(phi);	
				newVertex.p[2] = sin(phi);	
				newVertex.n[0] = cos(theta)*cos(phi);	
				newVertex.n[1] = sin(theta)*cos(phi);	
				newVertex.n[2] = sin(phi);
				vertices.push_back(newVertex);
			}
		}
		for(i=0;i<nX-1;i++){
			for(j=0;j<nY-1;j++){
				MeshTriangle newTriangle,newTriangle2;
				newTriangle[0] = i*nY+j;
				newTriangle[1] = (i+1)*nY+j;
				newTriangle[2] = (i+1)*nY+j+1;
				newTriangle2[0] = (i+1)*nY+j+1;
				newTriangle2[1] = i*nY+j+1;
				newTriangle2[2] = i*nY+j;
				triangles.push_back(newTriangle2);
				triangles.push_back(newTriangle);
				
			}
		}
		buildVertexArray();
		buildColorArray();
	}
	
/* void draw() const {
        // This code is deprecated. We will how to use vertex arrays and vertex buffer objects instead. (Exercice 1)
        glBegin (GL_TRIANGLES);
        for (unsigned int t = 0; t < triangles.size (); t++)
            for (unsigned int j = 0; j < 3; j++) {
                const MeshVertex & v = vertices[  triangles[t][j]  ];
                glNormal3f (v.n[0], v.n[1], v.n[2]);
                glVertex3f (v.p[0], v.p[1], v.p[2]);
            }
        glEnd ();
    }*/
	void draw() const{
		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_NORMAL_ARRAY);
		glEnableClientState(GL_COLOR_ARRAY);
		glEnable(GL_COLOR_MATERIAL);
		glVertexPointer(3,GL_FLOAT,3*sizeof(float),(GLvoid*)(&positionArray[0]));
		glNormalPointer(GL_FLOAT,3*sizeof(float),(GLvoid*)(&normalArray[0]));
		glColorPointer(3,GL_FLOAT,3*sizeof(float),(GLvoid*)(&colorArray[0]));
		glDrawElements(GL_TRIANGLES,triangleArray.size(),GL_UNSIGNED_INT, (GLvoid*)(&triangleArray[0]));
	}

};
#endif
