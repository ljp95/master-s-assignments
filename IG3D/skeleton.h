#ifndef MESH_H
#define MESH_H


#include <vector>
#include <string>
#include "Vec3.h"
#include "Skeleton.h"

#include <cmath>

#include <GL/glut.h>


// -------------------------------------------
// Basic Mesh class
// -------------------------------------------

struct MeshVertex {
    inline MeshVertex () {
        w.clear();
    }
    inline MeshVertex (const Vec3 & _p, const Vec3 & _n) : p (_p), n (_n) {
        w.clear();
    }
    inline MeshVertex (const MeshVertex & vertex) : p (vertex.p), n (vertex.n) , w(vertex.w) {
    }
    inline virtual ~MeshVertex () {}
    inline MeshVertex & operator = (const MeshVertex & vertex) {
        p = vertex.p;
        n = vertex.n;
        w = vertex.w;
        return (*this);
    }
    // membres :
    Vec3 p; // une position
    Vec3 n; // une normale
    std::vector< double > w; // skinning weights
};

struct MeshTriangle {
    inline MeshTriangle () {
        v[0] = v[1] = v[2] = 0;
    }
    inline MeshTriangle (const MeshTriangle & t) {
        v[0] = t.v[0];   v[1] = t.v[1];   v[2] = t.v[2];
    }
    inline MeshTriangle (unsigned int v0, unsigned int v1, unsigned int v2) {
        v[0] = v0;   v[1] = v1;   v[2] = v2;
    }
    inline virtual ~MeshTriangle () {}
    inline MeshTriangle & operator = (const MeshTriangle & t) {
        v[0] = t.v[0];   v[1] = t.v[1];   v[2] = t.v[2];
        return (*this);
    }
    // membres :
    unsigned int v[3];
};




class Mesh {
public:
    std::vector<MeshVertex> V;
    std::vector<MeshTriangle> T;

    void loadOFF (const std::string & filename);
    void recomputeNormals ();


	void computeSkinningWeights( Skeleton const & skeleton ) {
        //---------------------------------------------------//
        //---------------------------------------------------//
        // code to change :

        // Indications:
        // you should compute weights for each vertex w.r.t. the skeleton bones
        // so each vertex will have B weights (B = number of bones)
        // these weights shoud be stored in vertex.w:

        for( unsigned int i = 0 ; i < V.size() ; ++i ) {
            MeshVertex & vertex = V[ i ];

			float alphas[skeleton.bones.size()];
			float sum_alpha = 0;
			for (unsigned int j = 0 ;j<skeleton.bones.size() ; j++){
				Bone b = skeleton.bones[j];
				Vec3 p1 = skeleton.articulations[b.joints[0]].p;
				Vec3 p2 = skeleton.articulations[b.joints[1]].p;
				Vec3 vi = vertex.p;
				double lambda = Vec3::dot(vi-p1,p2-p1)/Vec3::dot(p2-p1,p2-p1);
				Vec3 pj = p1;
				if(lambda<0){
					Vec3 pj = p1;
				}
				else{
					if(lambda>1){
						Vec3 pj = p2;
					}
					else{
						Vec3 pj = p1 + lambda*(p2-p1);
					}
				}
				float alpha = 1/Vec3::dot(vi-pj,vi-pj);
				alphas[j] = alpha;
				sum_alpha += alpha;			
			}
			for (unsigned int k = 0;k<sizeof(alphas);k++){
				double alpha = alphas[k];
				vertex.w.push_back(alpha/sum_alpha);
			}
		
        }

	

        //---------------------------------------------------//
        //---------------------------------------------------//
        //---------------------------------------------------//
    }
   

    void draw() const {
        glEnable(GL_LIGHTING);
        glBegin (GL_TRIANGLES);
        for (unsigned int i = 0; i < T.size (); i++)
            for (unsigned int j = 0; j < 3; j++) {
                const MeshVertex & v = V[T[i].v[j]];
                glNormal3f (v.n[0], v.n[1], v.n[2]);
                glVertex3f (v.p[0], v.p[1], v.p[2]);
            }
        glEnd ();
    }

    void drawTransformedMesh( SkeletonTransformation const & transfo ) const {
        std::vector< Vec3 > newPositions( V.size() );

        //---------------------------------------------------//
        //---------------------------------------------------//
        // code to change :
        for( unsigned int i = 0 ; i < V.size() ; ++i ) {
            Vec3 p = V[i].p;

            // Indications:
            // you should use the skinning weights to blend the transformations of the vertex position by the bones.
			newPositions[ i ] = Vec3(0,0,0);
			for(unsigned int j=0;j<transfo.boneTransformations.size();j++){
				BoneTransformation b = transfo.boneTransformations[j];
				newPositions[ i ] += V[i].w[j]*(b.worldSpaceRotation*p + b.worldSpaceTranslation);
			}
            
        }

        //---------------------------------------------------//
        //---------------------------------------------------//
        //---------------------------------------------------//

        glEnable(GL_LIGHTING);
        glBegin (GL_TRIANGLES);
        for (unsigned int i = 0; i < T.size (); i++)
            for (unsigned int j = 0; j < 3; j++) {
                const MeshVertex & v = V[T[i].v[j]];
                Vec3 p = newPositions[ T[i].v[j] ];
                glNormal3f (v.n[0], v.n[1], v.n[2]);
                glVertex3f (p[0], p[1], p[2]);
            }
        glEnd ();
    }
};



#endif
