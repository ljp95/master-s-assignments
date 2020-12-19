#include "Mesh.h"
#include <iostream>
#include <fstream>

void Mesh::phongSubdivision(int k) {
	std::vector <MeshTriangle> newTris;
	std::vector <MeshVertex> newVerts;
	
	for(unsigned int i =0;i<triangles.size();i++){
		MeshTriangle t = triangles[i];
		MeshVertex v0 = vertices[t[0]];
		MeshVertex v1 = vertices[t[1]];
		MeshVertex v2 = vertices[t[2]];

		Vec3 p0 = v0.p;
		Vec3 n0 = v0.n;
		Vec3 p1 = v1.p;
		Vec3 n1 = v1.n;
		Vec3 p2 = v2.p;
		Vec3 n2 = v2.n;	

		for(int ku = 0;ku<=k+1;ku++){
			double u = double(ku)/(k+1);
			for(int kv = 0;kv<=k+1-ku;kv++){
				double v = double(kv)/(k+1);
				
				Vec3 p = (1-u-v)*p0 + u*p1 + v*p2;
				Vec3 proj0 = p + (Vec3::dot(p0-p,n0)*n0)/ n0.squareLength();			
				Vec3 proj1 = p + (Vec3::dot(p1-p,n1)*n1)/ n1.squareLength();				
				Vec3 proj2 = p + (Vec3::dot(p2-p,n2)*n2)/ n2.squareLength();	
				
				MeshVertex vertex;
				vertex.n = (1-u-v)*n0 + u*(n1-n0) + v*(n2-n0);
				vertex.p = (1-u-v)*proj0 + u*(proj1-proj0) + v*(proj2-proj0);
				vertex.n.normalize();
				newVerts.push_back(vertex);
			} 
		}
		int indStart = 0;
		for(int ku = 0;ku<k+1;ku++){
			for(int kv = 0;kv<k+1-ku;kv++){
				int ind0 = indStart+kv;
				int ind1 = ind0+1;
				int ind2 = ind0+k+2-ku;
				int ind3 = ind2+1;
				newTris.push_back(MeshTriangle(ind0, ind1, ind2));
				if(kv < k-ku){
            		newTris.push_back(MeshTriangle(ind1, ind3, ind2));
				}
			indStart = indStart + k+2-ku;
			}
		}
	}
	triangles = newTris;
	vertices = newVerts;
	buildVertexArray();
	buildTriangleArray();
	buildNormalArray();
}

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



