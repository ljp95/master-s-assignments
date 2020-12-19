#ifndef SCENE_H
#define SCENE_H

#include <vector>
#include <string>
#include "Mesh.h"

#include <GL/glut.h>

class Scene {
    // Mettez ici tout ce que vous souhaitez avoir dans votre scene 3D.
    // Pour l'instant, on a uniquement des maillages, mais par la suite on pourra rajouter des objets specialises comme des spheres, des cylindres ou des cones par ex...
    std::vector< Mesh > meshes;

public:
    Scene() {}

    void addMesh(std::string const & modelFilename) {
        meshes.resize( meshes.size() + 1 );
        Mesh & meshAjoute = meshes[ meshes.size() - 1 ];
        meshAjoute.loadOFF (modelFilename);
        meshAjoute.centerAndScaleToUnit ();
        meshAjoute.recomputeNormals ();
		meshAjoute.buildVertexArray();
		meshAjoute.buildColorArray();
    }

    void addSphere(std::string const & modelFilename) {
        meshes.resize( meshes.size() + 1 );
        Mesh & meshAjoute = meshes[ meshes.size() - 1 ];
        meshAjoute.setUnitSphere(10,10);
		meshAjoute.buildVertexArray();
		meshAjoute.buildColorArray();
    }

    void draw() const {

        // iterer sur l'ensemble des objets, et faire leur rendu.
        for( unsigned int mIt = 0 ; mIt < meshes.size() ; ++mIt ) {
            Mesh const & mesh = meshes[mIt];
            mesh.draw();

            
            // copies affichees : (Exercice 3)
	    	glPushMatrix();
            glTranslatef(2,  0,  0  ); glRotatef( 0  , 0  ,  0  ,  0  ); glScalef( 1 , 1  ,  1  );
            
			mesh.draw();
			glPopMatrix();
			glPushMatrix();
            glTranslatef(-2  ,  0  ,  0  ); glRotatef( 90  ,  0  ,  1  ,  0 ); glScalef( 0.5,  0.5 ,0.5 );
			mesh.draw();
			glPopMatrix();
            
        }
    }
};



#endif
