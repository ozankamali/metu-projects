#ifndef __HW1__PARSER__
#define __HW1__PARSER__

#include <string>
#include <vector>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <limits>
#include <pthread.h>

namespace parser
{
    struct Vec3f
    {
        float x, y, z;
    };

    struct Vec3i
    {
        int x, y, z;
    };

    struct Vec4f
    {
        float x, y, z, w;
    };

    struct Camera
    {
        Vec3f position;
        Vec3f gaze;
        Vec3f up;
        Vec4f near_plane;
        float near_distance;
        int image_width, image_height;
        std::string image_name;
    };

    struct PointLight
    {
        Vec3f position;
        Vec3f intensity;
    };

    struct Material
    {
        bool is_mirror;
        Vec3f ambient;
        Vec3f diffuse;
        Vec3f specular;
        Vec3f mirror;
        float phong_exponent;
    };

    struct Face
    {
        int v0_id;
        int v1_id;
        int v2_id;
        Vec3f v0;
        Vec3f v1;
        Vec3f v2;
        Vec3f normal;
    };

    struct Mesh
    {
        int material_id;
        std::vector<Face> faces;
    };

    struct Triangle
    {
        int material_id;
        Face indices;
        Vec3f v0;
        Vec3f v1;
        Vec3f v2;
        Vec3f normal; 
    };

    struct Sphere
    {
        int material_id;
        int center_vertex_id;
        float radius;
        Vec3f center; 
    };

    struct Scene
    {
        Vec3i background_color;
        float shadow_ray_epsilon;
        int max_recursion_depth;
        std::vector<Camera> cameras;
        Vec3f ambient_light;
        std::vector<PointLight> point_lights;
        std::vector<Material> materials;
        std::vector<Vec3f> vertex_data;
        std::vector<Mesh> meshes;
        std::vector<Triangle> triangles;
        std::vector<Sphere> spheres;

        void loadFromXml(const std::string &filepath);
    };

}

// Function declarations
parser::Vec3f mult(parser::Vec3f v, float s);
parser::Vec3f add(parser::Vec3f v1, parser::Vec3f v2);
parser::Vec3f subtract(parser::Vec3f v1, parser::Vec3f v2);
float dot(parser::Vec3f v1, parser::Vec3f v2);
parser::Vec3f get_normal(parser::Vec3f v);
parser::Vec3f cross(parser::Vec3f v1, parser::Vec3f v2);

#endif
