#ifndef TRIANGLE_H
#define TRIANGLE_H
#include "Vec3.h"
#include "Ray.h"
#include "Plane.h"

struct RayTriangleIntersection{
  bool intersectionExists;
  float lambda;
  float u,v;
  Vec3 intersection;
};

class Triangle {
private:
  Vec3 m_c[3] , m_normal;
  float area;
public:
  Triangle() {}
  Triangle( Vec3 const & c0 , Vec3 const & c1 , Vec3 const & c2 ) {
    m_c[0] = c0;
    m_c[1] = c1;
    m_c[2] = c2;
    updateAreaAndNormal();
  }
  void updateAreaAndNormal() {
    Vec3 nNotNormalized = Vec3::cross( m_c[1] - m_c[0] , m_c[2] - m_c[0] );
    float norm = nNotNormalized.length();
    m_normal = nNotNormalized / norm;
    area = norm / 2.f;
  }
  void setC0( Vec3 const & c0 ) { m_c[0] = c0; } // remember to update the area and normal afterwards!
  void setC1( Vec3 const & c1 ) { m_c[1] = c1; } // remember to update the area and normal afterwards!
  void setC2( Vec3 const & c2 ) { m_c[2] = c2; } // remember to update the area and normal afterwards!
  Vec3 const & normal() const { return m_normal; }
  Vec3 projectOnSupportPlane( Vec3 const & p ) const {
    return p - Vec3::dot( p - m_c[0] , m_normal )*m_normal;
  }
  float squareDistanceToSupportPlane( Vec3 const & p ) const { return (projectOnSupportPlane(p) - p).squareLength(); }
  float distanceToSupportPlane( Vec3 const & p ) const { return sqrt( squareDistanceToSupportPlane(p) ); }
  bool isParallelTo( Line const & L ) const {
    return Vec3::dot( L.direction() , m_normal ) == 0.f;
  }
  Vec3 getIntersectionPointWithSupportPlane( Line const & L ) const {
    // you should check first that the line is not parallel to the plane!
    float lambda = Vec3::dot( m_c[0] - L.origin() , normal() ) / Vec3::dot( L.direction() , normal() );
    return L.origin() + lambda * L.direction();
  }
  void computeBarycentricCoordinates( Vec3 const & p , float & u0 , float & u1 , float & u2 ) const {
    float denominator = Vec3::dot( Vec3::cross(m_c[1]-m_c[0] , m_c[2]-m_c[0]) , m_normal );
    u0 = Vec3::dot( Vec3::cross(p-m_c[0] , m_c[2]-m_c[0]) , m_normal ) / denominator;
    u1 = Vec3::dot( Vec3::cross(p-m_c[1] , m_c[0]-m_c[1]) , m_normal ) / denominator;
    u2 = Vec3::dot( Vec3::cross(p-m_c[2] , m_c[1]-m_c[2]) , m_normal ) / denominator;
  }

  RayTriangleIntersection getIntersection( Ray const & ray ) const {
    RayTriangleIntersection result;
    // 1) check that the ray is not parallel to the triangle:
    float dotDirectionAndNormal = Vec3::dot( ray.direction() , normal() );
    if( dotDirectionAndNormal == 0.f ) {
      // then the ray is parralel to the triangle:
      result.intersectionExists = false;
      return result;
    }
    // 2) check that the triangle is "in front of" the ray:
    result.lambda = Vec3::dot( m_c[0] - ray.origin() , normal() ) / dotDirectionAndNormal;
    if(result.lambda < 0.f) {
      // then the triangle is "behind" the ray:
      result.intersectionExists = false;
      return result;
    }
    result.intersection = ray.origin() + result.lambda * ray.direction();
    // 3) check that the intersection point is inside the triangle:
    // CONVENTION: compute u,v such that p = u*c0 + v*c1 + (1-u-v)*c2, check that 0 <= u,v <= 1
    float dummyFloat;
    computeBarycentricCoordinates( result.intersection , result.u , result.v , dummyFloat );
    if( result.u < 0.f || result.u > 1.f || result.v < 0.f || result.v > 1.f ) {
      // then the point is outside the triangle:
      result.intersectionExists = false;
      return result;
    }
    // 4) Finally, if all conditions were met, then there is an intersection! :
    result.intersectionExists = true;
    return result;
  }
};
#endif
