#ifndef PLANE_H
#define PLANE_H
#include "Vec3.h"
#include "Line.h"
class Plane {
private:
    Vec3 m_center , m_normal;
public:
    Plane() {}
    Plane( Vec3 const & c , Vec3 const & n ) {
        m_center = c;
        m_normal = n; m_normal.normalize();
    }
    void setCenter( Vec3 const & c ) { m_center = c; }
    void setNormal( Vec3 const & n ) { m_normal = n; m_normal.normalize(); }
    Vec3 const & center() const { return m_center; }
    Vec3 const & normal() const { return m_normal; }
    Vec3 project( Vec3 const & p ) const {
        return p - Vec3::dot( p - m_center , m_normal )*m_normal;
    }
    float squareDistance( Vec3 const & p ) const { return (project(p) - p).squareLength(); }
    float distance( Vec3 const & p ) const { return sqrt( squareDistance(p) ); }
    bool isParallelTo( Line const & L ) const {
        return Vec3::dot( L.direction() , m_normal ) == 0.f;
    }
    Vec3 getIntersectionPoint( Line const & L ) const {
        // you should check first that the line is not parallel to the plane!
        float lambda = Vec3::dot( center() - L.origin() , normal() ) / Vec3::dot( L.direction() , normal() );
        return L.origin() + lambda * L.direction();
    }
};
#endif
