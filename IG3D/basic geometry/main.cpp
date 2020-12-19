
#include <iostream>
#include "Vec3.h"
#include "Line.h"
#include "Ray.h"
#include "Plane.h"
#include "Triangle.h"
int main() {
        Vec3 v1(0,1,-1), v2(4,5,6), v3(3,0,0);
        std::cout << Vec3::dot(v1 , v2) << "   " << Vec3::dot(v1,v3) << "   " << Vec3::cross(v1,v2) << "   " << v3-v2 << "   " << 5*v1 << std::endl;
        Line L(Vec3(0,0,0) , Vec3(0,10,0));
        std::cout << L.project(v2) << "   " << L.distance(v3) << std::endl;
        Ray R(Vec3(0.5,0.1,0) , Vec3(0,10,0));
        std::cout << R.project(v2) << "   " << R.distance(v3) << std::endl;
        Plane P(Vec3(0,0,0) , Vec3(0,10,0));
        std::cout << P.project(v2) << "    " << P.getIntersectionPoint(R) << std::endl;
        Triangle T( Vec3(0,0,0) , Vec3(1,0,0) , Vec3(0,1,0));
        Ray R2(Vec3(0.5,0.1,10) , Vec3(0,0,20));
        RayTriangleIntersection intersection = T.getIntersection(R2);
        std::cout << intersection.intersectionExists << "   " << intersection.lambda << "   " << intersection.intersection << std::endl;
        Ray R3(Vec3(0.5,0.1,10) , Vec3(0,0,-20));
        intersection = T.getIntersection(R3);
        std::cout << intersection.intersectionExists << "   " << intersection.lambda << "   " << intersection.intersection << std::endl;
        return 0;
}
