#include "Mesh.h"
#include <iostream>
#include <fstream>

void Mesh::loadOFF (const std::string & filename) {
    std::ifstream in (filename.c_str ());
    if (!in)
        exit (EXIT_FAILURE);
    std::string offString;
    unsigned int sizeV, sizeT, tmp;
    in >> offString >> sizeV >> sizeT >> tmp;
    vertices.resize (sizeV);
    triangles.resize (sizeT);
    for (unsigned int i = 0; i < sizeV; i++)
        in >> vertices[i].p;
    int s;
    for (unsigned int t = 0; t < sizeT; t++) {
        in >> s;
        for (unsigned int j = 0; j < 3; j++)
            in >> triangles[t][j];
    }
    in.close ();
}

void Mesh::recomputeNormals () {
    for (unsigned int i = 0; i < vertices.size (); i++)
        vertices[i].n = Vec3 (0.0, 0.0, 0.0);
    for (unsigned int t = 0; t < triangles.size (); t++) {
        Vec3 e01 = vertices[  triangles[t][1]  ].p -  vertices[  triangles[t][0]  ].p;
        Vec3 e02 = vertices[  triangles[t][2]  ].p -  vertices[  triangles[t][0]  ].p;
        Vec3 n = Vec3::cross (e01, e02);
        n.normalize ();
        for (unsigned int j = 0; j < 3; j++)
            vertices[  triangles[t][j]  ].n += n;
    }
    for (unsigned int i = 0; i < vertices.size (); i++)
        vertices[i].n.normalize ();
}

void Mesh::centerAndScaleToUnit () {
    Vec3 c(0,0,0);
    for  (unsigned int i = 0; i < vertices.size (); i++)
        c += vertices[i].p;
    c /= vertices.size ();
    float maxD = (vertices[0].p - c).length();
    for (unsigned int i = 0; i < vertices.size (); i++){
        float m = (vertices[i].p - c).length();
        if (m > maxD)
            maxD = m;
    }
    for  (unsigned int i = 0; i < vertices.size (); i++)
        vertices[i].p = (vertices[i].p - c) / maxD;
}

void Mesh::buildVertexArray()
{
    positionArray.clear();
    positionArray.reserve(3*vertices.size());
    for(unsigned int i=0; i<vertices.size();i++)
    {
        MeshVertex v = vertices[i];
        positionArray.push_back(v.p[0]);
        positionArray.push_back(v.p[1]);
        positionArray.push_back(v.p[2]);
    }
}

void Mesh::buildTriangleArray()
{
    triangleArray.clear();
    triangleArray.reserve(3*triangles.size());
    for(unsigned int i=0; i<triangles.size();i++)
    {
        MeshTriangle t = triangles[i];
        triangleArray.push_back(t[0]);
        triangleArray.push_back(t[1]);
        triangleArray.push_back(t[2]);
    }
}

void Mesh::buildNormalArray()
{
    normalArray.clear();
    normalArray.reserve(3*vertices.size());
    for(unsigned int i=0; i<vertices.size();i++)
    {
        MeshVertex v = vertices[i];
        normalArray.push_back(v.n[0]);
        normalArray.push_back(v.n[1]);
        normalArray.push_back(v.n[2]);
    }
}

void Mesh::builduvArray()
{
    uvArray.clear();
    uvArray.reserve(2*vertices.size());
    for(unsigned int i=0; i<vertices.size();i++)
    {
        MeshVertex vert = vertices[i];
        uvArray.push_back(vert.u);
        uvArray.push_back(vert.v);
    }
}

void Mesh::buildColorArray()
{
    //Couleur aléatoire
    colorArray.clear();
    colorArray.reserve(3*vertices.size());
    for(unsigned int i=0; i<vertices.size();i++)
    {
        MeshVertex & v = vertices[i];
        colorArray.push_back( v.c[0] );
        colorArray.push_back( v.c[1] );
        colorArray.push_back( v.c[2] );
    }
}

void Mesh::setUnitSphere(int nX, int nY)
{
    vertices.clear();
    triangles.clear();
    float thetaStep = 2*M_PI/(nX-1);
    float phiStep = M_PI/(nY-1);
    for(int i=0; i<nX;i++)
    {
        for(int j=0;j<nY;j++)
        {
            float t = thetaStep*i;
            float p = phiStep*j - M_PI/2;

            Vec3 position(cos(t)*cos(p), sin(t)*cos(p), sin(p));
            Vec3 normal(position[0],position[1],position[2]);

            normal.normalize();
            vertices.push_back(MeshVertex(position, normal));
        }
    }
    for(int i=0; i<nX-1;i++)
    {
        for(int j=0;j<nY-1;j++)
        {
            triangles.push_back(MeshTriangle(i*nY+j, (i+1)*nY+j, (i+1)*nY+j+1));
            triangles.push_back(MeshTriangle(i*nY+j, (i+1)*nY+j+1, i*nY+j+1));
        }
    }

}


void Mesh::setSquare(Vec3 p0, Vec3 p1, Vec3 p2, Vec3 p3)
{	
	vertices.resize( 4 );
	triangles.resize( 2 );

    vertices[0].p = p0;  vertices[0].u = 0; vertices[0].v = 0;
    vertices[1].p = p1;  vertices[1].u = 0; vertices[1].v = 1;	 
    vertices[2].p = p2;  vertices[2].u = 1; vertices[2].v = 1;
    vertices[3].p = p3;  vertices[3].u = 1; vertices[3].v = 0;
	
	triangles[0] = MeshTriangle(0,1,2);
    triangles[1] = MeshTriangle(0,2,3);
	
}
